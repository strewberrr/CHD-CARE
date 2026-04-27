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
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nets_set.model_multi_views import MultiViewDualTokensFusionSize
from functions_set.functions_multi_views import *
from datasets.dataset_mutil_views import MultiViewDataset
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
clip_size = config['clip_size']
train_tag = config['train_tag']

# Transformer parameters from config
d_model = config.get('d_model', 512)
spatial_size = config.get('spatial_size')
view_nhead = config.get('view_nhead')
view_encoder_layers = config.get('view_encoder_layers')
fusion_nhead = config.get('fusion_nhead')
fusion_layers = config.get('fusion_layers')
dropout = config.get('dropout', 0.1)

unfreeze_epoch = config['unfreeze_epoch']
use_scheduler = config['unfreeze_epoch']
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
max_views = 5

def collate_fn_skip_none(batch):
    """
    Custom collate function that filters None samples
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

test_transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                 std=[0.26862954, 0.26130258, 0.27577711])
])

pickle_path = config['pickle_path']
split_json_path = config['split_json_path']

# 18 fine-grained target labels for balanced sampling
TARGET_LABELS_FOR_SAMPLING = [
    11, 12, 13, 14, 15,    # VSD
    16, 17, 9, 10,         # ASD
    18,                    # PDA
    1, 2, 3, 4, 5, 6, 7, 8 # Normal
]

test_dataset = MultiViewDataset(
    pickle_path, split_json_path, 'test', test_transform,
    clip_size=clip_size, spatial_resolution=spatial_size
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False,
    drop_last=False,
    collate_fn=collate_fn_skip_none
)

print('Data size: test {}.'.format(len(test_dataset)))

### Model initialization ###
torch.cuda.set_device(cuda_device)

model_name = config['model_name']
model = MultiViewDualTokensFusionSize(
    view_num_classes=18, d_model=d_model,
    view_nhead=view_nhead, view_encoder_layers=view_encoder_layers,
    case_num_classes=num_classes,
    fusion_layers=fusion_layers, fusion_nhead=fusion_nhead,
    max_views=max_views, dropout=dropout,
    spatial_size=spatial_size
)

print(f"Model receptive field: {spatial_size}x{spatial_size}, max_views: {max_views}")

# Load pretrained weights
resume_path = config['resume_path']
if resume_path != '':
    resume_model_dict = torch.load(resume_path, map_location='cpu')
    resume_model_filter_dict = {
        k: v for k, v in resume_model_dict.items()
        if k in model.state_dict() and v.shape == model.state_dict()[k].shape
    }
    model.load_state_dict(resume_model_filter_dict, strict=True)
    print('Loaded model from {}'.format(resume_path))

model = model.cuda().eval()

results = test(test_loader, model, num_classes, cuda_device)

# ==============================================================================
#                      Result Post-processing and Visualization
# ==============================================================================
print("\n--- Generating Visualizations and Reports ---")
plot_dir = os.path.join(vis_dir, 'plots_test')
os.makedirs(plot_dir, exist_ok=True)

# Extract results
conf_mat = np.array(results['confusion_matrix'])
all_case_info = results['all_cases']
error_case_info = results['error_cases']

roc_fpr = results['roc_fpr']
roc_tpr = results['roc_tpr']
roc_auc_per_class = results['roc_auc_per_class']

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
class_names_4 = ['Normal', 'VSD', 'ASD', 'PDA']
ConfusionMatrixDisplay(
    confusion_matrix=conf_mat, display_labels=class_names_4
).plot(cmap=plt.cm.Blues, ax=ax)

ax.set_title(f'Test Set Confusion Matrix\nOverall Accuracy: {results["overall_accuracy"]:.2f}%')

plt.savefig(os.path.join(plot_dir, 'confusion_matrix_test.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print("Confusion matrix saved.")

# Plot multi-class ROC curve
print("Drawing ROC curve...")
plt.figure(figsize=(10, 8))
colors = ['aqua', 'darkorange', 'cornflowerblue', 'deeppink']

for class_idx_str, color in zip(roc_auc_per_class.keys(), colors):
    i = int(class_idx_str)
    if not np.isnan(roc_auc_per_class[class_idx_str]):
        plt.plot(
            roc_fpr[class_idx_str], roc_tpr[class_idx_str],
            color=color, lw=2,
            label=f'ROC curve of class {class_names_4[i]} (area = {roc_auc_per_class[class_idx_str]:.3f})'
        )

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title(f'Multi-class ROC Curve (Macro AUC = {results["auc_roc_macro"]:.3f})')

plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, 'roc_curve_test.png'), dpi=300, bbox_inches='tight')
plt.close()
print("ROC curve saved.")

# Save detailed predictions to JSON
all_info_path = os.path.join(vis_dir, 'test_all_case_predictions.json')
with open(all_info_path, 'w', encoding='utf-8') as f:
    json.dump(all_case_info, f, indent=4, ensure_ascii=False)
print(f"All case predictions saved to: {all_info_path}")

error_info_path = os.path.join(vis_dir, 'test_error_case_predictions.json')
with open(error_info_path, 'w', encoding='utf-8') as f:
    json.dump(error_case_info, f, indent=4, ensure_ascii=False)
print(f"Error case predictions saved to: {error_info_path}")

print("\n--- Test script finished successfully! ---")