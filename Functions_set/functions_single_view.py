import random
import os
import numpy as np
import torch.nn as nn
import torch
import tqdm
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

scaler = GradScaler()
from torch.distributions import Categorical


class MyConfusionMatrix(object):
    def __init__(self, class_num):
        self.class_num = class_num
        self.mat = np.zeros((class_num, class_num), dtype=np.int64)

    def reset(self):
        self.mat = np.zeros((self.class_num, self.class_num))

    def update(self, predictions, targets):
        """
        Update confusion matrix.

        Args:
            predictions (torch.Tensor): Model outputs (2D logits or 1D class indices)
            targets (torch.Tensor): Ground truth labels (1D class indices)
        """
        # Convert logits to class indices if needed
        if predictions.ndim > 1:
            pred_indices = torch.argmax(predictions, dim=1)
        else:
            pred_indices = predictions

        # Move to CPU for numpy compatibility
        pred_indices = pred_indices.cpu()
        targets = targets.cpu()

        # Update matrix
        for p, t in zip(pred_indices, targets):
            if 0 <= t < self.class_num and 0 <= p < self.class_num:
                self.mat[t, p] += 1
            else:
                print(f"Warning: Label out of bounds! True: {t}, Pred: {p}. Matrix size: {self.class_num}. Skipped.")


def fix_randomness(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def adjust_learning_rate(initial_lr, optimizer, weight_decay, epoch, steps):
    """Decay learning rate by 0.3 at specified steps"""
    power = sum([epoch >= step for step in steps])
    multiplier = 0.3 ** power
    lr = initial_lr * multiplier
    print('current learning rate', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = weight_decay


def save_checkpoint(model, epoch, prefix='./checkpoints'):
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    filename = os.path.join(prefix, f'epoch_{epoch}.pth')
    torch.save(model.state_dict(), filename)
    print('saved checkpoint to {}'.format(filename))
    return filename


class AverageMeter(object):
    """Compute and store running average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_weighted_acc(conf, weight=None):
    if weight is None:
        ws = [1.0 / conf.shape[0]] * conf.shape[0]
    else:
        ws = [w / sum(weight) for w in weight]

    true_sum = np.sum(conf, axis=1)
    true_pre_num = np.diag(conf)
    acc = 0.0

    for tpn, ts, w in zip(true_pre_num, true_sum, ws):
        if ts > 0:
            acc += (tpn / ts) * w
    return acc * 100


# Denormalization parameters
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


def denormalize(tensor, mean, std):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def visualize_quadruplet_batch(
    videos,
    paths,
    quadruplets_per_batch,
    epoch,
    save_dir='/mnt/data1/zyh/CHD_1001/CHD_classify/train/quadruplet_verification'
):
    """
    Visualize quadruplet training samples for verification.
    Args:
        videos: Batch tensor (B, T, C, H, W)
        paths: List of file paths
        quadruplets_per_batch: Number of quadruplet groups
        epoch: Current epoch
        save_dir: Save path
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"  -> Verifying quadruplet data for epoch {epoch}...")

    fig, axes = plt.subplots(quadruplets_per_batch, 4, figsize=(20, 5 * quadruplets_per_batch), squeeze=False)
    fig.suptitle(f'Quadruplet Verification - Epoch {epoch}', fontsize=20)

    for i in range(quadruplets_per_batch):
        roles = {
            'Anchor': i * 4,
            'Positive': i * 4 + 1,
            'Negative 1': i * 4 + 2,
            'Negative 2': i * 4 + 3
        }

        for col_idx, (role_name, sample_idx) in enumerate(roles.items()):
            middle_frame = videos[sample_idx, videos.size(1) // 2]
            img = denormalize(middle_frame, MEAN, STD)
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)

            filename = os.path.basename(paths[sample_idx])
            ax = axes[i, col_idx]
            ax.imshow(img_np)
            ax.set_title(f"Group {i+1}: {role_name}", fontsize=12)
            ax.set_xlabel(filename, fontsize=8, wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f'epoch_{epoch}_quadruplets.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Quadruplet verification image saved to {save_path}")


POSITIVE_TO_NEGATIVE_MAP = {
    10: 0, 11: 1, 12: 2, 13: 3, 14: 4,
    15: 5, 16: 6, 8: 1, 9: 4,
    17: 7,
}


class UnifiedGuidanceLoss(nn.Module):
    """
    Final Version - Symmetric Gumbel-Softmax Selection
    1. Class token decoupling (contrastive cross-entropy)
    2. Positive token attention sparsity (entropy minimization)
    3. Negative token attention smoothing (KL divergence)
    4. Positive token variance maximization (Gumbel-Softmax selection)
    5. Negative token variance minimization
    """
    def __init__(self, num_classes=18, num_string_tokens=25,
                 decouple_weight=1.0, sparsity_weight=0.5, smoothness_weight=0.5,
                 variance_p_weight=1.0, variance_n_weight=1.0,
                 gumbel_tau=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_string_tokens = num_string_tokens
        self.decouple_weight = decouple_weight
        self.sparsity_weight = sparsity_weight
        self.smoothness_weight = smoothness_weight
        self.variance_p_weight = variance_p_weight
        self.variance_n_weight = variance_n_weight
        self.gumbel_tau = gumbel_tau

        self.positive_token_indices = torch.arange(8, self.num_classes)
        self.negative_token_indices = torch.arange(0, 8)

    def forward(self, output_cls_tokens, all_layer_attention_scores, raw_strings, baseline, total_decouple_loss, labels_18cls):
        b = output_cls_tokens.size(0)
        device = output_cls_tokens.device

        loss_decouple = total_decouple_loss
        attention_scores = all_layer_attention_scores[-1]

        p_scores_all = attention_scores[:, self.positive_token_indices, :]
        n_scores_all = attention_scores[:, self.negative_token_indices, :]

        sample_labels = torch.argmax(labels_18cls, dim=1)

        loss_sparsity_total = torch.tensor(0.0, device=device)
        loss_smoothness_total = torch.tensor(0.0, device=device)
        loss_variance_p_total = torch.tensor(0.0, device=device)
        loss_variance_n_total = torch.tensor(0.0, device=device)

        num_positive = (sample_labels > 7).sum()
        num_negative = (sample_labels <= 7).sum()

        if num_positive > 0:
            positive_indices = torch.where(sample_labels > 7)[0]
            p_token_indices_in_group = sample_labels[positive_indices] - 8
            n_token_indices_in_group = torch.tensor(
                [POSITIVE_TO_NEGATIVE_MAP.get(l.item()) for l in sample_labels[positive_indices]],
                device=device
            )

            p_scores_pos = p_scores_all[positive_indices, p_token_indices_in_group, :]
            p_probs_pos = F.softmax(p_scores_pos, dim=-1)
            loss_sparsity_total = -torch.sum(p_probs_pos * torch.log(p_probs_pos + 1e-8), dim=-1).sum()

            residuals_pos = raw_strings[positive_indices]
            batch_baselines_pos = baseline[n_token_indices_in_group]
            residuals_pos = residuals_pos - batch_baselines_pos.unsqueeze(2)

            p_selectors = F.gumbel_softmax(p_scores_pos, tau=self.gumbel_tau, hard=True, dim=-1)
            key_residual_sequences = torch.einsum('bs,bstc->btc', p_selectors.float(), residuals_pos.float())
            loss_variance_p_total = -key_residual_sequences.var(dim=1).mean(dim=-1).sum()

            key_string_indices = torch.argmax(p_selectors, dim=1)
            n_scores_pos = n_scores_all[positive_indices, n_token_indices_in_group, :]

            for i in range(len(positive_indices)):
                key_idx = key_string_indices[i].item()
                non_key_indices = [j for j in range(self.num_string_tokens) if j != key_idx]
                non_key_n_scores = n_scores_pos[i, non_key_indices]

                uniform_dist = torch.full_like(non_key_n_scores, 1.0 / (self.num_string_tokens - 1))
                loss_smoothness_total += F.kl_div(
                    F.log_softmax(non_key_n_scores, dim=-1),
                    uniform_dist,
                    reduction='sum'
                )

                non_key_residuals = residuals_pos[i, non_key_indices, :, :]
                loss_variance_n_total += non_key_residuals.float().var(dim=1).mean()

        if num_negative > 0:
            negative_indices = torch.where(sample_labels <= 7)[0]
            n_token_indices_in_group = sample_labels[negative_indices]
            n_scores_neg = n_scores_all[negative_indices, n_token_indices_in_group, :]

            uniform_dist = torch.full_like(n_scores_neg, 1.0 / self.num_string_tokens)
            loss_smoothness_total += F.kl_div(
                F.log_softmax(n_scores_neg, dim=-1),
                uniform_dist,
                reduction='sum'
            )

            residuals_neg = raw_strings[negative_indices]
            batch_baselines_neg = baseline[n_token_indices_in_group]
            residuals_neg = residuals_neg - batch_baselines_neg.unsqueeze(2)
            loss_variance_n_total += residuals_neg.float().var(dim=2).mean()

        # Normalize losses
        loss_sparsity = loss_sparsity_total / num_positive if num_positive > 0 else torch.tensor(0.0, device=device)
        loss_smoothness = loss_smoothness_total / b
        loss_variance_p = loss_variance_p_total / num_positive if num_positive > 0 else torch.tensor(0.0, device=device)

        num_n_variances = (self.num_string_tokens - 1) * num_positive + self.num_string_tokens * num_negative
        loss_variance_n = loss_variance_n_total / max(num_n_variances, 1)

        # Total loss
        total_loss = (
            self.decouple_weight * loss_decouple
            + self.sparsity_weight * loss_sparsity
            + self.smoothness_weight * loss_smoothness
            + self.variance_p_weight * loss_variance_p
            + self.variance_n_weight * loss_variance_n
        )

        return total_loss, {
            "loss_decouple": loss_decouple.detach(),
            "loss_sparsity": loss_sparsity.detach(),
            "loss_smoothness": loss_smoothness.detach(),
            "loss_variance_p": loss_variance_p.detach(),
            "loss_variance_n": loss_variance_n.detach(),
        }


def train(
    train_loader,
    model,
    criterion_cls,
    aux_loss_fn,
    optimizer,
    epoch,
    baseline_features,
    device,
    debug=True
):
    """
    Final training loop with full metrics
    """
    print(f'Training Epoch {epoch}...')
    model.train()

    # Metric trackers
    total_losses = AverageMeter()
    cls_losses = AverageMeter()
    decouple_losses = AverageMeter()
    sparsity_losses = AverageMeter()
    smoothness_losses = AverageMeter()
    var_p_losses = AverageMeter()
    var_n_losses = AverageMeter()
    accuracy = AverageMeter()

    scaler = GradScaler()
    tbar = tqdm(train_loader)

    for i, data in enumerate(tbar):
        if not data:
            if debug:
                print(f"DEBUG (Batch {i}): Empty batch. Skipping.")
            continue

        # Load data
        videos, masks, labels_one_hot, labels_int, _ = data
        videos = videos.to(device, non_blocking=True)
        labels_one_hot = labels_one_hot.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(videos, labels_one_hot, masks)
            predicted_scores = outputs["predicted_scores"]
            loss_cls = criterion_cls(predicted_scores, labels_one_hot)

            loss_aux, aux_metrics = aux_loss_fn(
                output_cls_tokens=outputs["final_cls_tokens"],
                all_layer_attention_scores=outputs["all_layer_attention_scores"],
                raw_strings=outputs["raw_strings_for_loss"],
                baseline=baseline_features,
                total_decouple_loss=outputs["total_decouple_loss"],
                labels_18cls=labels_one_hot
            )

        total_loss = 10 * loss_cls + 0.1 * loss_aux

        if not torch.isfinite(total_loss):
            if debug:
                print(f"\nDEBUG (Batch {i}): Invalid loss detected. Skipping.")
            optimizer.zero_grad()
            continue

        # Backward pass
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        bs = videos.size(0)
        total_losses.update(total_loss.item(), bs)
        cls_losses.update(loss_cls.item(), bs)
        decouple_losses.update(aux_metrics['loss_decouple'].item(), bs)
        sparsity_losses.update(aux_metrics['loss_sparsity'].item(), bs)
        smoothness_losses.update(aux_metrics['loss_smoothness'].item(), bs)

        if aux_metrics['loss_variance_p'].abs() > 1e-6:
            var_p_losses.update(aux_metrics['loss_variance_p'].item(), bs)
        if aux_metrics['loss_variance_n'].abs() > 1e-6:
            var_n_losses.update(aux_metrics['loss_variance_n'].item(), bs)

        # Compute accuracy
        preds_int = torch.argmax(predicted_scores.detach(), dim=1)
        gt_int = torch.argmax(labels_one_hot, dim=1)
        acc = accuracy_score(gt_int.cpu().numpy(), preds_int.cpu().numpy())
        accuracy.update(acc, bs)

        # Update progress bar
        tbar.set_description(
            f'E:{epoch}, L:{total_losses.avg:.3f} '
            f'(Cls:{cls_losses.avg:.3f}, Aux:{loss_aux.item():.3f}) | '
            f'Acc:{accuracy.avg*100:.2f}%'
        )

    # Print epoch summary
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Total Loss:       {total_losses.avg:.4f}")
    print(f"  Cls Loss:         {cls_losses.avg:.4f}")
    print(f"  Decouple Loss:    {decouple_losses.avg:.4f}")
    print(f"  Sparsity Loss:    {sparsity_losses.avg:.4f}")
    print(f"  Smoothness Loss:  {smoothness_losses.avg:.4f}")
    print(f"  Var P Loss:       {var_p_losses.avg:.4f}")
    print(f"  Var N Loss:       {var_n_losses.avg:.4f}")
    print(f"  Accuracy:         {accuracy.avg*100:.2f}%\n")

    return total_losses.avg, accuracy.avg


def validate(
    valid_loader,
    model,
    criterion_cls,
    num_classes,
    epoch,
    device
):
    print('Evaluating...')
    model.eval()

    cls_losses = AverageMeter()
    conf = MyConfusionMatrix(num_classes)

    with torch.no_grad():
        tbar = tqdm(valid_loader)
        for data in tbar:
            if not data:
                continue

            videos, masks, labels_one_hot, labels_int, _ = data
            videos = videos.to(device, non_blocking=True)
            labels_one_hot = labels_one_hot.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(videos, labels_one_hot, masks)
                predicted_scores = outputs["predicted_scores"]
                loss = criterion_cls(predicted_scores, labels_one_hot)

            cls_losses.update(loss.item(), videos.size(0))
            conf.update(predicted_scores.data, labels_int)

            tbar.set_description(f'Validation E:{epoch}, Loss:{cls_losses.avg:.4f}')

    conf_mat = conf.mat
    tp = np.diag(conf_mat)
    total_per_class = conf_mat.sum(axis=1)

    # Balanced accuracy
    valid_mask = total_per_class > 0
    per_class_recall = tp[valid_mask] / total_per_class[valid_mask]
    balanced_acc = per_class_recall.mean() if valid_mask.any() else 0.0
    overall_acc = tp.sum() / conf_mat.sum() if conf_mat.sum() > 0 else 0.0

    print(f'\nValidation Epoch {epoch}:')
    print(f'  Loss:         {cls_losses.avg:.4f}')
    print(f'  Overall Acc:  {overall_acc*100:.2f}%')
    print(f'  Balanced Acc: {balanced_acc*100:.2f}%')
    print('Confusion Matrix:\n', conf_mat)

    return cls_losses.avg, balanced_acc, conf_mat