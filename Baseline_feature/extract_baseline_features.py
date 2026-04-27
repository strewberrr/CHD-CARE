import os
import sys
import re
import json
import argparse
from collections import Counter

import yaml
import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import v2
from einops import rearrange

# Add parent directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nets_set.model_baseline_feature import ResNetEncoder
from functions_set.functions_single_view import *
from utils import *

# ==============================================================================
# 1. Configuration Loading & Hyperparameters
# ==============================================================================
config_path = './Config/train_config_baseline_feature.yaml'
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
weight_decay = config['weight_decay']
clip_size = config['clip_size']
drop_rate = config['drop_rate']
train_tag = config['train_tag']

# Model specific parameters
d_model = config.get('d_model', 512)
temporal_nhead = config.get('temporal_nhead')
temporal_layers = config.get('temporal_layers')
spatial_nhead = config.get('spatial_nhead')
spatial_layers = config.get('spatial_layers')
dim_feedforward = config.get('dim_feedforward', 2048)
dropout = config.get('dropout', 0.1)
temporal_entmax_alpha = config['temporal_entmax_alpha']
spatial_entmax_alpha = config['spatial_entmax_alpha']

# Paths & Weights
description = 'CHD_1views_{}'.format(train_tag)
print('Description: {}'.format(description))
save_dir = '/mnt/data1/xxx/CHD_1001/CHD_classify/output/checkpoint/' + description
vis_dir = '/mnt/data1/xxx/CHD_1001/CHD_classify/output/visualize/' + description
train_class_weight = config['train_class_weight']
print('Train class weight:', train_class_weight)
valid_class_weight = config['valid_class_weight']
print('Valid class weight:', valid_class_weight)

fix_randomness(seed)

# ==============================================================================
# 2. Label and Path Configuration
# ==============================================================================
VIEW_TO_NORMAL_LABEL_ID = {
    "normal_view_1": 0, "normal_view_2": 1, "normal_view_3": 2, "normal_view_4": 3,
    "normal_view_5": 4, "normal_view_6": 5, "normal_view_7": 6, "normal_view_8": 7,
}

ROI_AREA_THRESHOLD = 150
ROI_PADDING = 20

# ==============================================================================
# 3. Helper Functions
# ==============================================================================

def parse_view_from_filename(filename):
    """Parse 'normal_view_X' from the filename."""
    match = re.search(r'(normal_view_\d+)', filename)
    if match: 
        return match.group(1)
    return None

def calculate_final_normal_baseline(
    negative_video_paths,
    cnn_backbone,
    projection,
    pool,
    transform,
    device,
    save_dir,
    spatial_size, 
    num_normal_views=8,
    feature_dim=512 
):
    """
    [Final Version - Full Implementation]
    Use fine-tuned CNN components to calculate 'spatially specific, temporally invariant' 
    normal baseline for each negative view.
    """
    print(f"Starting calculation of 'spatially specific, temporally invariant' normal baseline features... Feature size: {spatial_size}x{spatial_size}")

    # 1. Group video paths by view
    videos_by_view = {i: [] for i in range(num_normal_views)}
    for video_path in negative_video_paths:
        view_name = parse_view_from_filename(os.path.basename(video_path))
        if view_name and view_name in VIEW_TO_NORMAL_LABEL_ID:
            videos_by_view[VIEW_TO_NORMAL_LABEL_ID[view_name]].append(video_path)

    all_baseline_features = []
    
    # Move model components to GPU and set to evaluation mode
    cnn_backbone.to(device).eval()
    projection.to(device).eval()
    pool.to(device).eval()

    num_patch_tokens = spatial_size * spatial_size

    # 2. Calculate baseline independently for each view
    for view_id in sorted(videos_by_view.keys()):
        view_paths = videos_by_view[view_id]
        if not view_paths:
            print(f"Warning: View ID {view_id} has no videos, generating zero baseline.")
            all_baseline_features.append(torch.zeros(32, feature_dim))
            continue

        print(f"\n--- Processing View ID {view_id} ({len(view_paths)} videos) ---")
        
        # [[[Core Modification 3]]] Initialize accumulator using dynamically calculated count
        view_strings_sum = torch.zeros(num_patch_tokens, feature_dim, device=device)
        num_videos = 0

        with torch.no_grad():
            for video_path in tqdm(view_paths, desc=f"View {view_id}"):
                try:
                    # a. Video loading and full preprocessing
                    original_frames = video_loader(video_path)
                    if original_frames is None or len(original_frames) < 1: 
                        continue
                    
                    indices = np.arange(len(original_frames))
                    if len(original_frames) < clip_size: 
                        indices = np.tile(indices, (clip_size + len(original_frames) - 1) // len(original_frames))
                    processed_16_frames = original_frames[indices[:clip_size]]
                    
                    roi_bbox = find_doppler_roi_from_video(processed_16_frames, ROI_AREA_THRESHOLD, ROI_PADDING)
                    if roi_bbox is None: 
                        continue
                    
                    frames_for_model_np = np.array([cv2.resize(frame[roi_bbox[1]:roi_bbox[3], roi_bbox[0]:roi_bbox[2]], (224, 224)) for frame in processed_16_frames])
                    
                    # Convert to model Tensor
                    video_tensor_uint8 = torch.from_numpy(frames_for_model_np.copy()).permute(0, 3, 1, 2)
                    input_tensor = transform(video_tensor_uint8).unsqueeze(0).to(device)
                    
                    # b. Feature extraction (B=1)
                    x_reshaped = rearrange(input_tensor, 'b t c h w -> (b t) c h w')
                    feature_maps = cnn_backbone(x_reshaped)   # (T, 2048, 7, 7)
                    projected_maps = projection(feature_maps) # (T, 512, 7, 7)
                    pooled_maps = pool(projected_maps)        # (T, 512, 4, 4)
                    
                    # c. Construct "string" Token
                    raw_strings = rearrange(pooled_maps, 't c h w -> (h w) t c') # (16, 16, 128)
                    
                    # d. Average over temporal dimension
                    time_averaged_strings = raw_strings.mean(dim=1) # -> (16, 128)
                    
                    # e. Accumulate
                    view_strings_sum += time_averaged_strings
                    num_videos += 1
                except Exception as e:
                    print(f"Error processing video {video_path}: {e}")
                    import traceback
                    traceback.print_exc()

        if num_videos > 0:
            view_baseline = view_strings_sum / num_videos
            all_baseline_features.append(view_baseline.cpu())
        else:
            all_baseline_features.append(torch.zeros(32, feature_dim))
    
    # 3. Merge and save
    final_baseline_tensor = torch.stack(all_baseline_features, dim=0)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"final_baseline_features_{spatial_size}x{spatial_size}_{feature_dim}dim_0212.pt")
    torch.save(final_baseline_tensor, save_path)
    
    print("\n" + "="*50)
    print(f"Baseline feature calculation for all views completed, saved to: {save_path}")
    print(f"Final baseline tensor shape: {final_baseline_tensor.shape}")
    print("="*50)
    
    return final_baseline_tensor

# ==============================================================================
# 4. Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    # 1. Prepare paths for all negative videos
    NEGATIVE_FOLDERS = [
        "ASD_normal", 
        "PDA_normal",
        "PDA_normal_0809",
        "VSD_normal"
    ]
    
    BASE_INPUT_DIR = "/mnt/data1/zyh/CHD_1001/Comparison_data_sets_0916"
    negative_paths = []
    
    for folder in NEGATIVE_FOLDERS:
        folder_path = os.path.join(BASE_INPUT_DIR, folder)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.avi', '.mp4')):
                negative_paths.append(os.path.join(folder_path, filename))

    # 2. Setup Transform & Model
    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    model = ResNetEncoder(num_classes=num_classes, d_model=d_model, projection_dim=128)
    
    resume_path = config['resume_path']
    if resume_path != '': 
        resume_model_dict = torch.load(resume_path, map_location='cpu')
        resume_model_filter_dict = {k: v for k, v in resume_model_dict.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(resume_model_filter_dict, strict=False)
        print('Loaded model from {}'.format(resume_path))

    # 3. Extract necessary components from loaded model
    cnn_backbone = model.cnn_backbone
    projection = model.projection  # Conv2d(2048, 512)

    TARGET_SPATIAL_SIZE = 5 

    # 4. Instantiate pool layer based on target dimensions
    pool = nn.AdaptiveAvgPool2d((TARGET_SPATIAL_SIZE, TARGET_SPATIAL_SIZE))

    # 5. Run baseline calculation
    calculate_final_normal_baseline(
        negative_video_paths=negative_paths,
        cnn_backbone=cnn_backbone,
        projection=projection,
        pool=pool,
        transform=transform,
        device=cuda_device,
        spatial_size=TARGET_SPATIAL_SIZE,
        save_dir=save_dir,
        feature_dim=d_model
    )