import argparse
import yaml
config_path = './Config/train_config_mutil_views.yaml'
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nets.model_multi_views import MultiViewDualTokensFusionSize
from functions.functions_multi_views import *
from Datasets.dataset_mutil_views import MultiViewDataset
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from utils import *
from timm.scheduler import CosineLRScheduler

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
clip_size = config['clip_size']
train_tag = config['train_tag']

# Model parameters from config
d_model = config.get('d_model', 512)
spatial_size = config.get('spatial_size')
view_nhead = config.get('view_nhead')
view_encoder_layers = config.get('view_encoder_layers')
fusion_nhead = config.get('fusion_nhead')
fusion_layers = config.get('fusion_layers')
dropout = config.get('dropout', 0.1)

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
    v2.Resize(size=(224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                 std=[0.26862954, 0.26130258, 0.27577711])
])

pickle_path = config['pickle_path']
split_json_path = config['split_json_path']

train_dataset = MultiViewDataset(
    pickle_path, split_json_path, 'train', train_transform,
    clip_size=clip_size, random_clip=True, max_views=5, spatial_resolution=spatial_size
)

# Weighted random sampler for class balance
case_labels = train_dataset.get_labels()
class_counts = np.bincount(case_labels, minlength=num_classes)
print(f"Training set case distribution: {class_counts}")
class_weights = 1. / (class_counts + 1e-8)
print(f"Calculated class weights: {class_weights}")

sample_weights = [class_weights[label] for label in case_labels]
train_sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True
)
print("Weighted random sampler created successfully.")

collate_fn = CollateFnWithPadding(batch_size=batch_size)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
    collate_fn=collate_fn
)

valid_dataset = MultiViewDataset(
    pickle_path, split_json_path, 'val', valid_transform,
    clip_size=clip_size, max_views=5, spatial_resolution=spatial_size
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
model = MultiViewDualTokensFusionSize(
    view_num_classes=18, d_model=d_model,
    view_nhead=view_nhead, view_encoder_layers=view_encoder_layers,
    case_num_classes=num_classes,
    fusion_layers=fusion_layers, fusion_nhead=fusion_nhead,
    max_views=5, dropout=dropout,
    spatial_size=spatial_size
)

print(f"Model spatial size: {spatial_size}")

# Load pretrained weights
resume_path = config['resume_path']
if resume_path != '':
    resume_model_dict = torch.load(resume_path, map_location='cpu')
    resume_model_filter_dict = {
        k: v for k, v in resume_model_dict.items()
        if k in model.state_dict() and v.shape == model.state_dict()[k].shape
    }
    model.load_state_dict(resume_model_filter_dict, strict=False)
    print('Loaded model from {}'.format(resume_path))

model = model.cuda()

# Loss functions
train_criterion = torch.nn.CrossEntropyLoss()
valid_criterion = torch.nn.CrossEntropyLoss()

# Optimizer & LR scheduler
for param in model.parameters():
    param.requires_grad = True

all_params = model.parameters()
optimizer = torch.optim.AdamW(
    all_params,
    lr=learning_rate,
    weight_decay=weight_decay
)
trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Optimizer configured. Peak LR: {learning_rate}, Trainable params: {trainable_param_count}")

lr_scheduler = None
if config.get('use_scheduler', False):
    decay_epoch = config.get('decay_epoch')
    warmup_epochs = config.get('warmup_epochs', 10)
    warmup_lr = float(config.get('warmup_lr', 3e-6))
    lr_min = float(config.get('lr_min', 1e-6))

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=decay_epoch,
        lr_min=lr_min,
        warmup_t=warmup_epochs,
        warmup_lr_init=warmup_lr,
        warmup_prefix=True
    )
    print("Cosine annealing LR scheduler enabled.")
    print(f"  - Warmup: {warmup_epochs} epochs")
    print(f"  - Decay: {decay_epoch} epochs")
else:
    print(f"Fixed learning rate ({learning_rate}) enabled.")

# Visualization setup
plot_dir = os.path.join(vis_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)
train_losses = []
train_accs = []
valid_losses = []
valid_accs = []

# Training loop
for epoch in range(total_epoch):
    print(f"\nEpoch {epoch}/{total_epoch - 1}")

    train_loss, train_acc = train(train_loader, model, train_criterion, optimizer, epoch, cuda_device)
    valid_loss, valid_acc, conf = validate(valid_loader, model, valid_criterion, epoch, cuda_device)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch} ended. Current LR = {current_lr:.8f}")

    train_acc *= 100
    valid_acc *= 100

    if epoch == 0:
        best_loss = valid_loss
        best_valid_acc = 10.0
        best_train_acc = train_acc
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