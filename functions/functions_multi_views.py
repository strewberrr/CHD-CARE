import os
import random
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend, suitable for servers
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
        # 1. Check the dimensions of the passed predictions
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
                print(f"Warning: Label out of bounds! True: {t}, Pred: {p}. Matrix size: {self.class_num}x{self.class_num}. Skipped this sample.")

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

# Denormalization constants
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
    Visualize a batch of quadruplet data to verify its composition.

    Args:
        videos (torch.Tensor): Video clips in the batch (B, T, C, H, W).
        paths (list of str): Original paths for each video in the batch.
        quadruplets_per_batch (int): Number of quadruplets contained in each batch.
        epoch (int): Current training epoch, used for naming files.
        save_dir (str): Directory to save visualization results.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"  -> Verifying quadruplet data for epoch {epoch}...")

    # Each quadruplet has 4 samples, so 4 columns per plot
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
# 2. Core Training, Validation, and Testing Loops
# ==============================================================================

def train(train_loader, model, criterion_4cls, optimizer, epoch, cuda_device, ensemble_loss_weight=0.5):
    """
    [Final Version] Training function for the multi-view model.
    """
    print(f'Training Epoch {epoch} (Multi-View)...')
    model.train()

    # --- Meter Initialization ---
    total_losses = AverageMeter()
    ce_losses = AverageMeter()
    acc_temporal = AverageMeter()

    scaler = GradScaler()
    tbar = tqdm(train_loader)
    
    for i, data in enumerate(tbar):
        if data is None: 
            continue
            
        # Unpack data generated by pad_collate_fn
        videos, masks, labels_4cls, num_views = data

        # --- Data Loading ---
        # videos shape: (B, max_views, T, C, H, W)
        videos = videos.to(cuda_device, non_blocking=True)
        masks = masks.to(cuda_device, non_blocking=True)
        labels_4cls = labels_4cls.to(cuda_device, non_blocking=True)
        num_views = num_views.to(cuda_device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=torch.float16):
            # 1. Forward pass, inputting multi-view data
            logits_temporal = model(videos, num_views, masks)
            
            # --- [Core: Loss Calculation] ---
            # 2. Calculate independent classification loss
            loss_temporal = criterion_4cls(logits_temporal, labels_4cls)
            total_loss = loss_temporal 

        # --- Backpropagation ---
        scaler.scale(total_loss).backward()
        # Optional: Gradient clipping
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # --- Update Metrics ---
        batch_size = videos.size(0)
        total_losses.update(total_loss.item(), batch_size)
        
        # a. Get independent predictions
        preds_temporal = torch.argmax(logits_temporal.detach(), dim=1)
        
        # b. Update independent accuracy
        acc_temporal.update(accuracy_score(labels_4cls.cpu().numpy(), preds_temporal.cpu().numpy()), batch_size)

        # d. Update progress bar
        tbar.set_description(
            f'Epoch: {epoch}, Loss: {total_losses.avg:.3f} '
            f'Acc(Temporal): {acc_temporal.avg:.2f}'
        )

    # --- Epoch End ---
    print(f'Epoch: {epoch}, Epoch End. Avg Loss: {total_losses.avg:.4f}')
    print(f'  - Temporal Head Acc: {acc_temporal.avg:.4f}')
    
    # Return key metrics
    return total_losses.avg, acc_temporal.avg

def validate(valid_loader, model, criterion_4cls, epoch, cuda_device):
    """
    [Modified Version] Validate the multi-view model.
    - Primary evaluation metric changed to Balanced Accuracy (avgacc).
    - Overall Accuracy (alloveracc) is also reported for reference.
    """
    print(f'Evaluating Epoch {epoch} (Multi-View)...')
    model.eval()  # Switch to evaluation mode

    # --- Meter Initialization ---
    total_losses = AverageMeter()
    acc_overall = AverageMeter()
    conf_temporal = ConfusionMatrix(4)  # 4 classes

    tbar = tqdm(valid_loader)
    with torch.no_grad():
        for i, data in enumerate(tbar):
            if data is None: 
                continue
                
            # Unpack data generated by pad_collate_fn
            videos, masks, labels_4cls, num_views = data

            # --- Data Loading ---
            videos = videos.to(cuda_device, non_blocking=True)
            masks = masks.to(cuda_device, non_blocking=True)
            labels_4cls = labels_4cls.to(cuda_device, non_blocking=True)
            num_views = num_views.to(cuda_device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                # 1. Forward pass
                logits_temporal = model(videos, num_views, masks)
                
                # 2. Calculate loss (for monitoring, not backpropagation)
                loss_temporal = criterion_4cls(logits_temporal, labels_4cls)
                total_loss = loss_temporal

            # --- [Core: Voting and Evaluation] ---
            # 3. Get independent predictions
            preds_temporal = torch.argmax(logits_temporal, dim=1)

            # --- Update Metrics ---
            batch_size = videos.size(0)
            total_losses.update(total_loss.item(), batch_size)
            
            # a. Update overall accuracy
            acc_overall.update(accuracy_score(labels_4cls.cpu().numpy(), preds_temporal.cpu().numpy()), batch_size)
            conf_temporal.update(preds_temporal.data, labels_4cls)
    
    # 1. Get final confusion matrix
    conf_mat = conf_temporal.mat
    
    # 2. Calculate Balanced Accuracy (avgacc)
    #    a. Calculate per-class accuracy (recall). Add 1e-8 to prevent division by zero.
    per_class_acc = conf_mat.diagonal() / (conf_mat.sum(axis=1) + 1e-8)
    #    b. Average the accuracy across all classes
    avg_acc = np.mean(per_class_acc)

    # 3. Overall Accuracy is already calculated by acc_overall.avg
    all_over_acc = acc_overall.avg

    # --- Print Final Report ---
    print(f'Epoch: {epoch}, Validation End.')
    print(f'  - Avg Validation Loss: {total_losses.avg:.4f}')
    print(f'  --- Accuracy Metrics ---')
    print(f'  - Overall Accuracy:  {all_over_acc:.4f}')
    print(f'  - Balanced Accuracy: {avg_acc:.4f}')
    print('  - Final Temporal Confusion Matrix:')
    print(conf_mat)
    
    # Return key metrics
    return total_losses.avg, avg_acc, conf_mat

def test(test_loader, model, num_classes, cuda_device):
    """
    [Final Test Version] Test the multi-view model outputting a single temporal logit.
    - Calculates and reports Overall Accuracy and Balanced Accuracy.
    - Calculates detailed evaluation metrics including F1, AUC, and Confusion Matrix.
    """
    print(f'--- Starting Final Test for Temporal Fusion Model ---')
    model.eval()  # Switch to evaluation mode

    # --- Initialize lists to collect results across the entire test set ---
    all_true_labels = []
    all_preds = []
    all_probs = []
    all_case_names = []

    tbar = tqdm(test_loader)
    with torch.no_grad():
        for i, data in enumerate(tbar):
            if data is None: 
                continue
            
            # Assuming test_loader returns (videos, masks, labels, num_views, case_name)
            videos, masks, labels_4cls, num_views, case_name = data

            # --- Data Loading ---
            videos = videos.to(cuda_device, non_blocking=True)
            masks = masks.to(cuda_device, non_blocking=True)
            labels_4cls = labels_4cls.to(cuda_device, non_blocking=True)
            num_views = num_views.to(cuda_device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                # 1. Forward pass to get logits
                logits_temporal = model(videos, num_views, masks)
            
            # --- 2. Get predictions and probabilities ---
            preds_temporal = torch.argmax(logits_temporal, dim=1)
            probs_temporal = F.softmax(logits_temporal, dim=1)

            # --- 3. Collect batch results ---
            all_true_labels.extend(labels_4cls.cpu().numpy())
            all_preds.extend(preds_temporal.cpu().numpy())
            all_probs.extend(probs_temporal.cpu().numpy())
            all_case_names.extend(case_name)

    # --- [Core: Calculate Final Metrics on Entire Test Set] ---
    print("\n--- Final Test Report ---")
    
    true_labels = np.array(all_true_labels)
    pred_labels = np.array(all_preds)
    pred_probs = np.array(all_probs)
    
    # --- 1. Calculate Evaluation Metrics ---
    overall_acc = accuracy_score(true_labels, pred_labels) * 100
    all_possible_labels = list(range(num_classes))
    conf_mat = confusion_matrix(true_labels, pred_labels, labels=all_possible_labels)
    per_class_recall = conf_mat.diagonal() / (conf_mat.sum(axis=1) + 1e-8)
    balanced_acc = np.mean(per_class_recall) * 100
    f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    
    # Calculate F1 score for each class (average=None returns an array)
    per_class_f1 = f1_score(true_labels, pred_labels, average=None, zero_division=0, labels=all_possible_labels)

    # Pre-initialize dictionaries
    fpr, tpr, roc_auc_per_class = {}, {}, {}
    
    # Calculate Macro AUC
    try:
        # Convert true labels to one-hot encoding for OVR calculation
        true_labels_one_hot = np.eye(num_classes)[true_labels]
        auc_roc_macro = roc_auc_score(true_labels_one_hot, pred_probs, multi_class='ovr', average='macro')
    except ValueError as e:
        print(f"\n[Warning]: Error calculating Macro AUC: {e}. Likely due to missing classes in the test set.")
        auc_roc_macro = float('nan')

    # Calculate ROC/AUC for each class
    for c in range(num_classes):
        y_true_binary = (true_labels == c)
        
        # Calculate only if both positive and negative samples exist
        if len(np.unique(y_true_binary)) > 1:
            try:
                fpr[c], tpr[c], _ = roc_curve(y_true_binary, pred_probs[:, c])
                roc_auc_per_class[c] = auc(fpr[c], tpr[c])
            except ValueError as e:
                print(f"Warning: Error calculating ROC for class {c}: {e}")
                fpr[c], tpr[c], roc_auc_per_class[c] = [], [], float('nan')
        else:
            fpr[c], tpr[c], roc_auc_per_class[c] = [], [], float('nan')
            
    # --- 2. Organize info for all cases and misclassified cases ---
    all_case_info = {}
    error_case_info = {}
    
    for i in range(len(all_case_names)):
        case_id = all_case_names[i]
        true_label = int(true_labels[i])
        pred_label = int(pred_labels[i])
        probabilities = pred_probs[i].tolist()
        
        case_info = {
            "true_label": true_label,
            "predicted_label": pred_label,
            "probabilities": [round(p, 4) for p in probabilities]  # Round to 4 decimal places
        }
        
        all_case_info[case_id] = case_info
        
        # Add to error dictionary if misclassified
        if true_label != pred_label:
            error_case_info[case_id] = case_info

    # --- 4. Print Final Report ---
    print(f'\n--- Final Performance Metrics ---')
    print(f'  - Overall Accuracy:  {overall_acc:.2f}%')
    print(f'  - Balanced Accuracy: {balanced_acc:.2f}%')
    print(f'  - Macro F1-Score:    {f1_macro:.4f}')
    print(f'  - Macro AUC-ROC:     {auc_roc_macro:.4f}')
    
    print('\n--- Per-class Recall (Accuracy) And F1 Scores ---')
    for i in range(num_classes):
        print(f'  - Class {i}: Recall (Acc) = {per_class_recall[i] * 100:.2f}%,  F1-Score = {per_class_f1[i]:.4f}')
        
    print('\n--- Final Confusion Matrix ---')
    print(conf_mat)
    
    # 5. Return Results Dictionary
    results = {
        'overall_accuracy': overall_acc,
        'balanced_accuracy': balanced_acc,
        'confusion_matrix': conf_mat.tolist(),
        'f1_macro': f1_macro,
        'auc_roc_macro': auc_roc_macro,
        'per_class_recall': (per_class_recall * 100).tolist(),
        'roc_fpr': {str(k): (v.tolist() if hasattr(v, 'tolist') else list(v)) for k, v in fpr.items()},
        'roc_tpr': {str(k): (v.tolist() if hasattr(v, 'tolist') else list(v)) for k, v in tpr.items()},
        'roc_auc_per_class': {str(k): v for k, v in roc_auc_per_class.items()},
        'all_cases': all_case_info,
        'error_cases': error_case_info
    }
    
    return results