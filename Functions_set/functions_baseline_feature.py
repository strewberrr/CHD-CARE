import os
import random
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.distributions import Categorical

scaler = GradScaler()

# ==============================================================================
# 1. Utility Classes & Functions
# ==============================================================================

class ConfusionMatrix(object):
    def __init__(self, class_num):
        self.class_num = class_num
        self.mat = np.zeros((class_num, class_num))

    def reset(self):
        self.mat = np.zeros((self.class_num, self.class_num))

    def update(self, predictions, targets):
        """
        Update the confusion matrix.
        
        Args:
            predictions (torch.Tensor): Model predictions. 
                                        [Can be 2D logits or 1D class indices]
            targets (torch.Tensor): Ground truth labels (1D class indices).
        """
        # --- [Core Correction: Unified Variable Names] ---
        # 1. Check the dimensions of the input predictions
        if predictions.ndim > 1:
            # If 2D logits, calculate class indices via argmax
            pred_indices = torch.argmax(predictions, dim=1)
        else:
            # If already 1D class indices, use directly
            pred_indices = predictions
        
        # 2. Ensure indices and labels are on CPU and converted to correct types
        pred_indices = pred_indices.cpu()
        targets = targets.cpu()
        
        # 3. Iterate and update the confusion matrix
        for p, t in zip(pred_indices, targets):
            # Safety check to prevent label out-of-bounds
            if t < self.class_num and p < self.class_num:
                self.mat[t, p] += 1
            else:
                print(f"Warning: Label out of bounds! True: {t}, Pred: {p}. Matrix size: {self.class_num}x{self.class_num}. Skipped.")

def fix_randomness(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def adjust_learning_rate(initial_lr, optimizer, weight_decay, epoch, steps):
    """Sets the learning rate to the initial LR decayed by 0.3 every stage."""
    power = sum([epoch >= step for step in steps])
    multiplier = 0.3 ** power
    lr = initial_lr * multiplier
    print('Current learning rate:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 
        param_group['weight_decay'] = weight_decay

def save_checkpoint(model, epoch, prefix='./checkpoints'):
    filename = os.path.join(prefix, 'epoch_' + str(epoch) + '.pth')
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    torch.save(model.state_dict(), filename)
    print('Saved checkpoint to {}'.format(filename))
    return filename

class AverageMeter(object):
    """Computes and stores the average and current value."""
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
        ws = [1 / conf.shape[0]] * conf.shape[0]
    else:
        ws = [w / sum(weight) for w in weight]
    
    true_sum = np.sum(conf, axis=1)
    true_pre_num = [conf[i, i] for i in range(conf.shape[0])]
    acc = 0
    
    for tpn, ts, w in zip(true_pre_num, true_sum, ws):
        if ts == 0:
            continue
        acc += (tpn / ts) * w
    return acc * 100

# Normalization constants
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
    Visualize the quadruplet data in a batch to verify its composition.

    Args:
        videos (torch.Tensor): Video clips in the batch (B, T, C, H, W).
        paths (list of str): Original paths of each video in the batch.
        quadruplets_per_batch (int): Number of quadruplets contained in each batch.
        epoch (int): Current training epoch, used for naming files.
        save_dir (str): Directory to save visualization results.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"  -> Verifying quadruplet data for epoch {epoch}...")

    # Each quadruplet has 4 samples, thus 4 columns per plot
    fig, axes = plt.subplots(quadruplets_per_batch, 4, figsize=(20, 5 * quadruplets_per_batch), squeeze=False)
    fig.suptitle(f'Quadruplet Verification - Epoch {epoch}', fontsize=20)
    
    # Iterate through each quadruplet
    for i in range(quadruplets_per_batch):
        # Define indices and names for each role
        roles = {
            'Anchor': i * 4,
            'Positive': i * 4 + 1,
            'Negative 1': i * 4 + 2,
            'Negative 2': i * 4 + 3
        }
        
        col_idx = 0
        for role_name, sample_idx in roles.items():
            # Extract the middle frame
            middle_frame = videos[sample_idx, videos.size(1) // 2]
            
            # Denormalize and convert to displayable Numpy array
            img = denormalize(middle_frame, MEAN, STD)
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)

            # Get filename
            filename = os.path.basename(paths[sample_idx])
            
            # Plotting
            ax = axes[i, col_idx]
            ax.imshow(img_np)
            ax.set_title(f"Group {i+1}: {role_name}", fontsize=12)
            # Display full filename below image using xlabel
            ax.set_xlabel(filename, fontsize=8, wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
            
            col_idx += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f'epoch_{epoch}_quadruplets.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Quadruplet verification image saved to {save_path}")

# ==============================================================================
# 2. Training & Validation Pipelines
# ==============================================================================

def train_with_contrastive(
    train_loader, model, criterion_18cls, triplet_loss_fn, optimizer, epoch, cuda_device, contrastive_loss_weight=0.5):
    """
    [Adapted Version]
    Train using a single 18-class head combined with 'anatomical dual' supervised contrastive learning.
    """
    print(f'Training Epoch {epoch} (CE + Hierarchical Self-Supervised ST-CL)...')
    model.train()
    
    total_losses = AverageMeter()
    ce_losses = AverageMeter()
    s_cl_losses = AverageMeter()
    t_cl_losses = AverageMeter()
    acc_18cls = AverageMeter()
    
    scaler = GradScaler()
    tbar = tqdm(train_loader)
    
    for i, data in enumerate(tbar):
        if data is None: 
            continue
        videos, masks, labels_18cls, _, paths = data
        
        videos = videos.to(cuda_device, non_blocking=True)
        masks = masks.to(cuda_device, non_blocking=True)
        labels_18cls = labels_18cls.to(cuda_device, non_blocking=True)


        optimizer.zero_grad()
        mask_to_use = masks if epoch >= 50 else None

        with autocast(device_type='cuda', dtype=torch.float16):
            # 1. Forward pass (Receiving targeted outputs)
            outputs = model(videos, doppler_mask_7x7=masks, return_attention=True) 
            
            logits_temporal, temporal_weights, frame_t_outputs = outputs
            logits_18cls = logits_temporal 

            # 2. 18-class Cross Entropy Loss
            loss_ce = criterion_18cls(logits_18cls, labels_18cls)

            # 3. Temporal Contrastive Learning Loss
            loss_temporal_cl = calculate_contrastive_loss(
                mode='temporal',
                features=frame_t_outputs,
                weights=temporal_weights,
                triplet_loss_fn=triplet_loss_fn,
                quadruplets_per_batch=train_loader.batch_sampler.quadruplets_per_batch,
                device=cuda_device
            )
            
            # 5. Total Loss Calculation
            total_loss = loss_ce + contrastive_loss_weight * loss_temporal_cl
            
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics update and logging
        batch_size = videos.size(0)
        total_losses.update(total_loss.item(), batch_size)
        ce_losses.update(loss_ce.item(), batch_size)
        if loss_temporal_cl.item() > 0: 
            t_cl_losses.update(loss_temporal_cl.item(), batch_size)
        
        preds_18cls = torch.argmax(logits_18cls.detach(), dim=1)
        acc_18cls.update(accuracy_score(labels_18cls.cpu().numpy(), preds_18cls.cpu().numpy()), batch_size)
        
        tbar.set_description(
            f'E: {epoch}, L: {total_losses.avg:.3f} '
            f'(CE:{ce_losses.avg:.3f}, T-CL:{t_cl_losses.avg:.3f}), '
            f'Acc:{acc_18cls.avg:.2f}'
        )

    print(f'Epoch: {epoch}, Epoch End. Avg Loss: {total_losses.avg:.4f}, Avg Acc: {acc_18cls.avg:.4f}')
    return total_losses.avg, acc_18cls.avg

def calculate_contrastive_loss(
    mode, features, weights, 
    triplet_loss_fn, quadruplets_per_batch, device,
    num_pairs=16, top_k=2
):
    """
    [Final Unified Version]
    Executes hierarchical, self-supervised spatial or temporal contrastive learning on a structured batch.
    """
    if features is None or weights is None: 
        return torch.tensor(0.0, device=device)
        
    total_loss = torch.tensor(0.0, device=device)
    
    # Counter for valid quadruplets to avoid division by zero
    valid_quads = 0
    
    for i in range(quadruplets_per_batch):
        # Logical indexing
        anchor_idx, pos_idx, neg1_idx, neg2_idx = i*4, i*4+1, i*4+2, i*4+3
        positive_pool, negative_pool = None, None

        if mode == 'temporal':
            # --- Temporal Mode (Frame vs. Random Frame) ---
            frame_features = features
            temporal_weights = weights
            
            anchor_feats, pos_feats = frame_features[anchor_idx], frame_features[pos_idx]
            neg1_feats, neg2_feats = frame_features[neg1_idx], frame_features[neg2_idx]
            anchor_weights, pos_weights = temporal_weights[anchor_idx], temporal_weights[pos_idx]
            
            # Positive Pool: Top-K key frames from positive videos
            positive_pool = torch.cat([
                anchor_feats[torch.topk(anchor_weights, k=min(top_k, 16)).indices],
                pos_feats[torch.topk(pos_weights, k=min(top_k, 16)).indices]
            ], dim=0)

            # Negative Pool: Random frames from negative videos
            negative_pool = torch.cat([
                neg1_feats[torch.randperm(16, device=device)[:top_k]],
                neg2_feats[torch.randperm(16, device=device)[:top_k]]
            ], dim=0)

        elif mode == 'spatial':
            # --- Spatial Mode (Key-Frame vs. Hard-Frame, Hierarchical approximation) ---
            frame_features = features
            temporal_weights = weights
            
            # 1. Positive Pool: Identical to temporal mode, key frames selected by temporal attention
            anchor_feats, pos_feats = frame_features[anchor_idx], frame_features[pos_idx]
            anchor_weights, pos_weights = temporal_weights[anchor_idx], temporal_weights[pos_idx]
            positive_pool = torch.cat([
                anchor_feats[torch.topk(anchor_weights, k=min(top_k, 16)).indices],
                pos_feats[torch.topk(pos_weights, k=min(top_k, 16)).indices]
            ], dim=0)

            # 2. Negative Pool: "Most attended" hard negative frames selected by temporal attention from negative videos
            neg1_feats, neg2_feats = frame_features[neg1_idx], frame_features[neg2_idx]
            neg1_weights, neg2_weights = temporal_weights[neg1_idx], temporal_weights[neg2_idx]
            negative_pool = torch.cat([
                neg1_feats[torch.topk(neg1_weights, k=min(top_k, 16)).indices],
                neg2_feats[torch.topk(neg2_weights, k=min(top_k, 16)).indices]
            ], dim=0)

        # --- Generic Triplet Loss Calculation ---
        if positive_pool is not None and negative_pool is not None and \
           positive_pool.shape[0] >= 2 and negative_pool.shape[0] >= 1:
            
            quad_loss = torch.tensor(0.0, device=device)
            num_possible_pairs = positive_pool.shape[0] * (positive_pool.shape[0] - 1)
            actual_num_pairs = min(num_pairs, num_possible_pairs)
            
            if actual_num_pairs > 0:
                valid_quads += 1
                for _ in range(actual_num_pairs):
                    anchor_idx_pool, pos_idx_pool = torch.randperm(positive_pool.shape[0])[:2]
                    neg_idx_pool = torch.randint(0, negative_pool.shape[0], (1,)).item()
                    
                    anchor = positive_pool[anchor_idx_pool]
                    pos = positive_pool[pos_idx_pool]
                    neg = negative_pool[neg_idx_pool]
                    
                    quad_loss += triplet_loss_fn(anchor, pos, neg)
                
                total_loss += (quad_loss / actual_num_pairs)

    return total_loss / valid_quads if valid_quads > 0 else torch.tensor(0.0, device=device)

def validate_with_contrastive(valid_loader, model, criterion_18cls, num_classes, epoch, cuda_device):
    """
    [Adapted Version]
    Evaluate a single 18-class head and contrastive learning loss on the validation set.
    """
    print('Evaluating...')
    model.eval()
    
    losses = AverageMeter()
    conf = ConfusionMatrix(num_classes)
    
    # Disable gradient computation for validation
    with torch.no_grad():
        tbar = tqdm(valid_loader)
        for i, data in enumerate(tbar):
            if data is None: 
                continue
            
            # Ensure DataLoader returns consistent data structure as during training
            videos, masks, labels_18cls, _, _ = data
            
            videos = videos.to(cuda_device, non_blocking=True)
            labels_18cls = labels_18cls.to(cuda_device, non_blocking=True)
            masks = masks.to(cuda_device, non_blocking=True)
            
            mask_to_use = masks if epoch >= 50 else None

            with autocast(device_type='cuda', dtype=torch.float16):
                # 1. Forward Pass
                outputs = model(
                    videos, 
                    doppler_mask_7x7=masks,
                    return_attention=True
                )
                
                # 2. Unpack model outputs to retrieve required 18-class logits
                logits_18cls, _, _ = outputs

                if logits_18cls is None:
                    raise ValueError("Model did not return expected logits for 18-class classification.")

                # 3. Calculate Loss
                loss = criterion_18cls(logits_18cls, labels_18cls)

            # --- Update Metrics (Outside autocast scope) ---
            losses.update(loss.item(), videos.size(0))
            conf.update(logits_18cls.data, labels_18cls)
            
            tbar.set_description(f'Validation E:{epoch}, Loss:{losses.avg:.4f}')

    # --- Post-loop Metric Calculation ---
    epoch_acc = cal_weighted_acc(conf.mat)
    
    print(f'Epoch: {epoch}, Validation End.')
    print(f'  Loss: {losses.avg:.4f}, Valid Accuracy: {epoch_acc:.4f}')
    print('  Confusion Matrix: ')
    print(conf.mat)
    
    return losses.avg, epoch_acc, conf.mat