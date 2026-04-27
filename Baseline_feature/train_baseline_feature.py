import os
import sys
import argparse
import random
from collections import Counter

import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
from timm.scheduler import CosineLRScheduler

# Append parent directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nets_set.model_baseline_feature import ResNetTransformerDualClsEntmaxWithDopplerMaskTemporal
from functions_set.functions_baseline_feature import *
from datasets.dataset_baseline_feature import ContrastivePairedDataset, StructuredPairedBatchSampler
from utils import *

# ==============================================================================
# 1. Configuration Loading & Hyperparameters
# ==============================================================================
config_path = '.Config/train_config_baseline_feature.yaml'  
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

for c in config:
    print(c, config[c])

# Hardware & Training parameters
cuda_device = config['cuda_device']
num_classes = config['num_classes']
batch_size = config['batch_size']
learning_rate = float(config['learning_rate'])
total_epoch = config['total_epoch']
seed = config['seed']
num_workers = config['num_workers']
momentum = config['momentum']
weight_decay = config['weight_decay']

view_dropout = config['view_dropout']
drop_rate = config['drop_rate']
add_bias = config['add_bias']
num_segments = config['num_segments']
segment_len = config['segment_len']
train_tag = config['train_tag']

# ResNetTransformer specific parameters
d_model = config.get('d_model', 512)
temporal_nhead = config.get('temporal_nhead')
temporal_layers = config.get('temporal_layers')
spatial_nhead = config.get('spatial_nhead')
spatial_layers = config.get('spatial_layers')
dim_feedforward = config.get('dim_feedforward', 2048)
dropout = config.get('dropout', 0.1)
temporal_entmax_alpha = config['temporal_entmax_alpha']
spatial_entmax_alpha = config['spatial_entmax_alpha']

# Scheduler parameters
unfreeze_epoch = config['unfreeze_epoch']
use_scheduler = config['unfreeze_epoch'] # Note: Inherited logic from original code
warmup_epochs = config['warmup_epochs']
warmup_lr = config['warmup_lr']
lr_min = config['lr_min']

# Paths & Weights
description = 'CHD_1views_{}'.format(train_tag)
print('Description: {}'.format(description))
save_dir = '/mnt/data1/zyh/CHD_1001/CHD_classify/output/checkpoint/' + description
vis_dir = '/mnt/data1/zyh/CHD_1001/CHD_classify/output/visualize/' + description
train_class_weight = config['train_class_weight']
print('Train class weight:', train_class_weight)
valid_class_weight = config['valid_class_weight']
print('Valid class weight:', valid_class_weight)

# Initialization
fix_randomness(seed)
best_valid_acc = 0
best_valid_acc_epoch = -1
best_loss_epoch = -1
best_loss = 100.0

# ==============================================================================
# 2. Data Loading & Collation Functions
# ==============================================================================

class CollateFnWithPadding:
    def __init__(self, batch_size):
        self.expected_batch_size = batch_size
        print(f"--- CollateFnWithPadding initialized with expected_batch_size: {self.expected_batch_size} ---")

    def __call__(self, batch):
        """
        A robust collate_fn.
        1. Filter out None samples in the batch.
        2. If the batch is incomplete, randomly pad with valid samples to the expected size.
        """
        # a. Filter out None
        batch = [item for item in batch if item is not None]
        
        # b. If empty after filtering, return None to let the main loop skip
        if not batch:
            return None
        
        # c. If the batch size is insufficient, perform padding
        current_size = len(batch)
        if current_size < self.expected_batch_size:
            num_to_add = self.expected_batch_size - current_size
            # Randomly sample with replacement from existing valid samples
            padding = random.choices(batch, k=num_to_add)
            batch.extend(padding)
        
        # d. Batch size is guaranteed to be the expected size, safe to collate
        return torch.utils.data.dataloader.default_collate(batch)

def collate_fn_skip_none(batch):
    """
    Custom collate_fn.
    Filters out all None values in the batch and pads if necessary.
    """
    EXPECTED_BATCH_SIZE = 16 

    batch = [item for item in batch if item is not None]
    
    if not batch:
        return None
        
    current_size = len(batch)
    if current_size < EXPECTED_BATCH_SIZE:
        num_to_add = EXPECTED_BATCH_SIZE - current_size
        padding = random.choices(batch, k=num_to_add)
        batch.extend(padding)
        
    return torch.utils.data.dataloader.default_collate(batch)

# Transforms
train_transform = v2.Compose([
    v2.RandomZoomOut(fill=0, side_range=(1., 1.5), p=0.5),
    v2.RandomCrop(size=(224, 224)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=(-20, 20)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

valid_transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

pickle_path = config['pickle_path'] 
split_json_path = config['split_json_path']

# 18 target fine-grained categories for balanced sampling
TARGET_LABELS_FOR_SAMPLING = [
    11, 12, 13, 14, 15, # VSD
    16, 17, 9, 10,      # ASD
    18,                 # PDA
    1, 2, 3, 4, 5, 6, 7, 8 # Normal
]

# Dataset & DataLoader instantiation
train_dataset = ContrastivePairedDataset(pickle_path, split_json_path, 'train', train_transform, segment_len=segment_len)
train_batch_sampler = StructuredPairedBatchSampler(dataset=train_dataset, batch_size=batch_size)

collate_fn = CollateFnWithPadding(batch_size=batch_size)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_sampler=train_batch_sampler, 
    num_workers=num_workers, 
    pin_memory=True,
    collate_fn=collate_fn
)

valid_dataset = ContrastivePairedDataset(pickle_path, split_json_path, 'val', valid_transform, segment_len=segment_len)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=batch_size, 
    shuffle=False,
    num_workers=num_workers, 
    pin_memory=True,
    drop_last=False,
    collate_fn=collate_fn
)

print(f'Data size: training {len(train_dataset)}, validation {len(valid_dataset)}')

# ==============================================================================
# 3. Model Initialization
# ==============================================================================
torch.cuda.set_device(cuda_device)

model_name = config['model_name']
model = ResNetTransformerDualClsEntmaxWithDopplerMaskTemporal(
    num_classes=num_classes, d_model=d_model, 
    temporal_nhead=temporal_nhead, temporal_layers=temporal_layers, 
    spatial_nhead=spatial_nhead, spatial_layers=spatial_layers,
    dim_feedforward=dim_feedforward, dropout=dropout,
    temporal_entmax_alpha=temporal_entmax_alpha, 
    spatial_entmax_alpha=spatial_entmax_alpha
)

# Load pre-trained weights if specified
resume_path = config['resume_path']
if resume_path != '': 
    resume_model_dict = torch.load(resume_path, map_location='cpu')
    resume_model_filter_dict = {k: v for k, v in resume_model_dict.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(resume_model_filter_dict, strict=False)
    print('Loaded model from {}'.format(resume_path))

model = model.cuda()       

# Loss functions
train_criterion = nn.CrossEntropyLoss()
valid_criterion = nn.CrossEntropyLoss()
triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2).to(cuda_device)

# ==============================================================================
# 4. Phased Training Setup
# ==============================================================================

print("\n--- Phase 1: Configure frozen training (Spatial module and Classification head only) ---")

# a. Freeze all model parameters initially
for param in model.parameters():
    param.requires_grad = False
print("All model parameters have been frozen.")

# b. Unfreeze spatial-related modules and classification head
modules_to_unfreeze_stage1 = [
    model.cls_token_temporal,
    model.pos_encoder_temporal,
    model.temporal_transformer,
    model.temporal_classifier 
]

params_unfrozen_count = 0
for module in modules_to_unfreeze_stage1:
    if isinstance(module, nn.Parameter):
        module.requires_grad = True
        params_unfrozen_count += 1
    else:
        for param in module.parameters():
            param.requires_grad = True
            params_unfrozen_count += 1
print(f"Unfrozen {params_unfrozen_count} parameter groups for Phase 1 training.")

# c. Configure Optimizer for Phase 1
print("\n--- Configure Optimizer (Phase 1) ---")
trainable_params_stage1 = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(
    trainable_params_stage1, 
    lr=learning_rate, 
    weight_decay=weight_decay
)
print(f"Optimizer configured, targeting {len(trainable_params_stage1)} parameter groups.")

# d. Configure Learning Rate Scheduler
lr_scheduler = None
if config.get('use_scheduler', False):
    warmup_epochs = config.get('warmup_epochs', 10)
    warmup_lr = float(config.get('warmup_lr', 1e-6))
    lr_min = float(config.get('lr_min', 1e-6))

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=total_epoch, 
        lr_min=lr_min,
        warmup_t=warmup_epochs,
        warmup_lr_init=warmup_lr,
        warmup_prefix=True
    )
    print("Cosine annealing learning rate scheduler enabled.")
else:
    print(f"Fixed learning rate ({learning_rate}) enabled.")

# Initialize plotting configurations
plot_dir = os.path.join(vis_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)
train_losses, train_accs = [], []
valid_losses, valid_accs = [], []
UNFREEZE_EPOCH = 50

# ==============================================================================
# 5. Training Loop
# ==============================================================================

print("\n--- Training Started ---")
for epoch in range(total_epoch):
    print(f"\nEpoch {epoch}/{total_epoch-1}")
    
    # Phased unfreezing logic
    if epoch == UNFREEZE_EPOCH:
        print("\n--- !!! Unfreezing all parameters !!! ---")
        print(f"--- Epoch {epoch}: Unfreezing CNN Backbone and Temporal Transformer ---")
        
        # 1. Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        
        # 2. Recreate optimizer to include all parameters with scaled down learning rate
        new_lr = learning_rate / 10 
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=new_lr, 
            weight_decay=weight_decay
        )
        print(f"Optimizer recreated, containing all model parameters. New LR: {new_lr}")
        
    train_loss, train_acc = train_with_contrastive(train_loader, model, train_criterion, triplet_loss_fn, optimizer, epoch, cuda_device)
    valid_loss, valid_acc, conf = validate_with_contrastive(valid_loader, model, valid_criterion, num_classes, epoch, cuda_device)
    
    # Learning rate scheduler step
    if lr_scheduler is not None:
        lr_scheduler.step(epoch + 1)
        
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch} ended. Current LR = {current_lr:.8f}")

    train_acc = train_acc * 100
    
    # Checkpointing logic
    if epoch == 0:
        best_loss = valid_loss
        best_valid_acc = 10
        best_train_acc = 10 
        best_valid_acc_epoch = epoch
        best_train_acc_epoch = epoch
        best_loss_epoch = epoch
        save_checkpoint(model, epoch, prefix=save_dir+'/train')
    else:
        if valid_loss <= best_loss:
            print(f'Loss decreased from {best_loss:.3f} to {valid_loss:.3f} from epoch {best_loss_epoch}')
            best_loss = valid_loss
            best_loss_epoch = epoch
            save_checkpoint(model, best_loss, prefix=save_dir+'/best_loss')
        else:
            print(f'Loss did not decrease from {best_loss:.3f} from epoch {best_loss_epoch}')
            
        if valid_acc >= best_valid_acc:
            print(f'Valid accuracy increased from {best_valid_acc:.3f} to {valid_acc:.3f} from epoch {best_valid_acc_epoch}')
            best_valid_acc = valid_acc
            best_valid_acc_epoch = epoch
            save_checkpoint(model, best_valid_acc, prefix=save_dir+'/best_valid_acc')
        else:
            print(f'Valid accuracy did not increase from {best_valid_acc:.3f} from epoch {best_valid_acc_epoch}')
            
        if train_acc >= best_train_acc:
            print(f'Train accuracy increased from {best_train_acc:.3f} to {train_acc:.3f} from epoch {best_train_acc_epoch}')
            best_train_acc = train_acc
            best_train_acc_epoch = epoch
            save_checkpoint(model, best_train_acc, prefix=save_dir+'/best_train_acc')
        else:
            print(f'Train accuracy did not increase from {best_train_acc:.3f} from epoch {best_train_acc_epoch}')

    # Metric tracking
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    valid_losses.append(valid_loss)
    valid_accs.append(valid_acc)

    # Plot metrics
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(valid_losses, label='Valid Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy', marker='o')
    plt.plot(valid_accs, label='Valid Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    
    plt.savefig(os.path.join(plot_dir, 'metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(conf, display_labels=range(num_classes)).plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f'Validation Confusion Matrix (Epoch {epoch})')
    plt.savefig(os.path.join(plot_dir, 'confusion.png'), dpi=300, bbox_inches='tight')
    plt.close()