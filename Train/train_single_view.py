import argparse
import yaml
config_path = './Config/train_config_single_view.yaml'
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
for c in config:
    print(c, config[c])
cuda_device = config['cuda_device']

import os
import numpy as np
import torch
import torchvision
import sys
import random
from collections import OrderedDict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nets.model_single_view import ResnetTransformerDualTokensTemporalSpatialDecouplesize
from functions.functions_single_view import *
from Datasets.dataset_single_view import ContrastivePairedDataset
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from utils import *
from timm.scheduler import CosineLRScheduler
from torch.nn import TripletMarginLoss

# Hyper-parameters
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

# Model parameters from config
d_model_transformer = config['d_model_transformer']
d_model_cnn = config['d_model_cnn']
compressed_dim = config['compressed_dim']
nhead = config['nhead']
num_layers = config['num_layers']
dropout = config['dropout']

unfreeze_epoch = config['unfreeze_epoch']
use_scheduler = config.get('use_scheduler', False)
warmup_epochs = config['warmup_epochs']
warmup_lr = config['warmup_lr']
lr_min = config['lr_min']

description = 'CHD_1views_{}'.format(train_tag)
print('Description： {}'.format(description))
save_dir = '/mnt/data1/xxx/CHD_1001/CHD_classify/output/checkpoint/' + description
vis_dir = '/mnt/data1/xxx/CHD_1001/CHD_classify/output/visualize/' + description
train_class_weight = config['train_class_weight']
print('Train class weight', train_class_weight)
valid_class_weight = config['valid_class_weight']
print('Valid class weight', valid_class_weight)

# Fix random seed
fix_randomness(seed)
best_valid_acc = 0
best_valid_acc_epoch = -1
best_loss_epoch = -1
best_loss = 100.0

spatial_size = 5

class CollateFnWithPadding:
    def __init__(self, batch_size):
        self.expected_batch_size = batch_size
        print(f"--- CollateFnWithPadding initialized with expected_batch_size: {self.expected_batch_size} ---")

    def __call__(self, batch):
        original_batch = batch
        batch = [item for item in batch if item is not None]

        if not batch:
            return None

        current_size = len(batch)
        if current_size < self.expected_batch_size:
            num_to_add = self.expected_batch_size - current_size
            padding = random.choices(batch, k=num_to_add)
            batch.extend(padding)

        return torch.utils.data.dataloader.default_collate(batch)


def transfer_weights_ultimate_precise(new_model, resume_path):
    """
    Final diagnostic version:
    Strict shape matching before loading weights,
    and print all shape mismatches.
    """
    if not (resume_path and os.path.exists(resume_path)):
        print("Warning: Invalid weight path.")
        return

    print(f"--- Starting weight transfer with shape check from {os.path.basename(resume_path)} ---")
    old_model_dict = torch.load(resume_path, map_location='cpu')
    new_model_dict = new_model.state_dict()

    weights_to_load = OrderedDict()
    shape_mismatch_keys = []

    # 1. Directly transfer backbone and projection
    for key, value in old_model_dict.items():
        if key.startswith('cnn_backbone.') or key.startswith('projection.'):
            if key in new_model_dict:
                if new_model_dict[key].shape == value.shape:
                    weights_to_load[key] = value
                else:
                    shape_mismatch_keys.append((key, value.shape, new_model_dict[key].shape))

    # 2. Precisely transfer Transformer
    for i in range(new_model.num_layers):
        rules = {
            f'temporal_transformer.layers.{i}.self_attn.': f'transformer_blocks.{i}.temporal_transformer.attn.',
            f'temporal_transformer.layers.{i}.linear1.':  f'transformer_blocks.{i}.temporal_transformer.linear1.',
            f'temporal_transformer.layers.{i}.linear2.':  f'transformer_blocks.{i}.temporal_transformer.linear2.',
            f'temporal_transformer.layers.{i}.norm1.':    f'transformer_blocks.{i}.temporal_transformer.norm1.',
            f'temporal_transformer.layers.{i}.norm2.':    f'transformer_blocks.{i}.temporal_transformer.norm2.',
        }

        for old_key, value in old_model_dict.items():
            for old_prefix, new_prefix in rules.items():
                if old_key.startswith(old_prefix):
                    new_key = old_key.replace(old_prefix, new_prefix)
                    if new_key in new_model_dict:
                        if new_model_dict[new_key].shape == value.shape:
                            weights_to_load[new_key] = value
                        else:
                            shape_mismatch_keys.append((new_key, value.shape, new_model_dict[new_key].shape))
                    break

    # Load weights and generate report
    missing, unexpected = new_model.load_state_dict(weights_to_load, strict=False)

    print("\n--- Weight Transfer Final Report ---")
    print(f"Successfully loaded {len(weights_to_load)} layers.")

    if shape_mismatch_keys:
        print("\n" + "!" * 20)
        print(f"[ERROR] Found {len(shape_mismatch_keys)} shape mismatches! Loading failed:")
        for key, old_shape, new_shape in shape_mismatch_keys:
            print(f"  - Key: {key}")
            print(f"    -> Weight shape: {old_shape}")
            print(f"    -> Model shape: {new_shape}")
        print("!" * 20)

    if missing:
        print(f"\n{len(missing)} layers remain randomly initialized:")
        for key in sorted(list(missing))[:10]:
            print(f"  - {key}")

    print("------------------\n")


train_transform = v2.Compose([
    v2.RandomZoomOut(fill=0, side_range=(1., 1.5), p=0.5),
    v2.RandomCrop(size=(224, 224)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=(-20, 20)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                 std=[0.26862954, 0.26130258, 0.27577711])
])

valid_transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                 std=[0.26862954, 0.26130258, 0.27577711])
])

pickle_path = config['pickle_path']
split_json_path = config['split_json_path']

# 18 fine-grained target labels
TARGET_LABELS_FOR_SAMPLING = [
    11, 12, 13, 14, 15,    # VSD
    16, 17, 9, 10,         # ASD
    18,                    # PDA
    1, 2, 3, 4, 5, 6, 7, 8 # Normal
]

train_dataset = ContrastivePairedDataset(
    pickle_path, split_json_path, 'train', train_transform,
    segment_len=segment_len, spatial_resolution=spatial_size
)

collate_fn = CollateFnWithPadding(batch_size=batch_size)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True,
    collate_fn=collate_fn
)

valid_dataset = ContrastivePairedDataset(
    pickle_path, split_json_path, 'val', valid_transform,
    segment_len=segment_len, spatial_resolution=spatial_size
)
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

### Model Initialization ###
torch.cuda.set_device(cuda_device)

model_name = config['model_name']
model = ResnetTransformerDualTokensTemporalSpatialDecouplesize(
    num_classes=num_classes,
    d_model_cnn=d_model_cnn,
    nhead=nhead,
    num_layers=num_layers,
    dropout=dropout,
    spatial_resolution=spatial_size
)

# Load baseline features
baseline_path = "./final_baseline_features_5x5_512dim_0212.pt"
if not os.path.exists(baseline_path):
    raise FileNotFoundError(f"Baseline feature file not found: {baseline_path}. Please run baseline script first.")
baseline_features = torch.load(baseline_path).to(cuda_device)
print(f"Successfully loaded baseline features, shape: {baseline_features.shape}")

# Load pretrained weights
resume_path = config['resume_path']
transfer_weights_ultimate_precise(model, resume_path)
model = model.cuda()

# Loss functions
train_criterion = torch.nn.CrossEntropyLoss()
valid_criterion = torch.nn.CrossEntropyLoss()

aux_loss_fn = UnifiedGuidanceLoss(
    decouple_weight=1.0, sparsity_weight=0.5, smoothness_weight=0.5,
    variance_p_weight=1.0, variance_n_weight=1.0,
    gumbel_tau=1.0
).to(cuda_device)

print("\n--- Configuring Optimizer and LR Scheduler ---")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"AdamW configured. Optimizing {trainable_params} parameters.")
print(f"Base LR: {learning_rate}")

# LR scheduler
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
    print("Cosine annealing LR scheduler enabled.")
else:
    print(f"Fixed learning rate ({learning_rate}) enabled.")

print(f"  - Warmup: {warmup_epochs} epochs ({warmup_lr} → {learning_rate})")
print(f"  - Cosine Annealing: {total_epoch - warmup_epochs} epochs ({learning_rate} → {lr_min})")

# Visualization setup
plot_dir = os.path.join(vis_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)
train_losses = []
train_accs = []
valid_losses = []
valid_accs = []
UNFREEZE_EPOCH = 1

print("\n--- Starting Training ---")
for epoch in range(total_epoch):
    print(f"\nEpoch {epoch}/{total_epoch-1}")

    train_loss, train_acc = train(
        train_loader, model, train_criterion, aux_loss_fn,
        optimizer, epoch, baseline_features, cuda_device
    )
    valid_loss, valid_acc, conf = validate(
        valid_loader, model, valid_criterion, num_classes, epoch, cuda_device
    )

    if lr_scheduler is not None:
        lr_scheduler.step(epoch + 1)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch} finished. Next LR: {current_lr:.8f}")

    train_acc *= 100
    valid_acc *= 100

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
            print(f'Loss decreased from {best_loss:.3f} to {valid_loss:.3f} (epoch {best_loss_epoch})')
            best_loss = valid_loss
            best_loss_epoch = epoch
            best_loss_path = save_checkpoint(model, best_loss, prefix=save_dir+'/best_loss')
        else:
            print(f'Loss did not improve (best: {best_loss:.3f} at epoch {best_loss_epoch})')

        if valid_acc >= best_valid_acc:
            print(f'Val accuracy increased from {best_valid_acc:.3f} to {valid_acc:.3f} (epoch {best_valid_acc_epoch})')
            best_valid_acc = valid_acc
            best_valid_acc_epoch = epoch
            best_valid_acc_path = save_checkpoint(model, best_valid_acc, prefix=save_dir+'/best_valid_acc')
        else:
            print(f'Val accuracy did not improve (best: {best_valid_acc:.3f} at epoch {best_valid_acc_epoch})')

        if train_acc >= best_train_acc:
            print(f'Train accuracy increased from {best_train_acc:.3f} to {train_acc:.3f} (epoch {best_train_acc_epoch})')
            best_train_acc = train_acc
            best_train_acc_epoch = epoch
            best_train_acc_path = save_checkpoint(model, best_train_acc, prefix=save_dir+'/best_train_acc')
        else:
            print(f'Train accuracy did not improve (best: {best_train_acc:.3f} at epoch {best_train_acc_epoch})')

    # Save metrics
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    valid_losses.append(valid_loss)
    valid_accs.append(valid_acc)

    # Plot loss & accuracy curves
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
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.savefig(os.path.join(plot_dir, 'metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(conf, display_labels=range(num_classes)).plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f'Validation Confusion Matrix (Epoch {epoch})')
    plt.savefig(os.path.join(plot_dir, 'confusion.png'), dpi=300, bbox_inches='tight')
    plt.close()