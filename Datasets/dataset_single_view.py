import os
import sys
import re
import json
import pickle
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from collections import defaultdict
from tqdm import tqdm
from utils import *

# Append parent directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ==============================================================================
# 1. Global Fine-grained Label Mapping
# ==============================================================================

FINE_GRAINED_LABEL_MAP = {
    # Abnormal
    "VSD_view_1": 10, "VSD_view_2": 11, "VSD_view_3": 12, "VSD_view_4": 13, "VSD_view_5": 14,
    "ASD_view_6": 15, "ASD_view_7": 16, "ASD_view_2": 8, "ASD_view_5": 9,
    "PDA_view_8": 17,
    # Normal (Includes spelling correction fallback)
    "normal_view_1": 0, "normal_view_2": 1, "normal_view_3": 2, "normal_view_4": 3,
    "normal_view_5": 4, "normal_view_6": 5, "normal_view_7": 6, "normal_view_8": 7,
}


def get_fine_grained_label_from_filename(filename, label_map):
    """
    Extracts the fine-grained label integer from the given filename.
    """
    fn_lower = filename.lower()
    match = re.search(r'_label_(\w+)_view_(\d+)', fn_lower)
    
    if match:
        disease_lower = match.group(1)
        view_num = match.group(2)
        
        # Handle typo robustness and normalization
        if disease_lower in ['normal', 'noraml']:
            disease_upper = 'normal'
        else:
            disease_upper = disease_lower.upper()
        
        key = f"{disease_upper}_view_{view_num}"
        return label_map.get(key, -1)
    
    return -1

# ==============================================================================
# 2. Dataset Pipeline
# ==============================================================================

class ContrastivePairedDataset(data.Dataset):
    """
    Dataset class for processing contrastive paired samples with fine-grained labels 
    and corresponding Doppler attention masks.
    """
    def __init__(self, pickle_path, split_path, split, transform=None, segment_len=32, spatial_resolution=5):
        self.pickle_path = pickle_path
        self.split_path = split_path
        self.num_classes = 18
        self.split = split
        self.transform = transform
        self.segment_len = segment_len
        self.spatial_resolution = spatial_resolution
        
        with open(split_path, 'r') as f:
            dataset_splits = json.load(f)
        self.samples = list(dataset_splits[split].items())
        
        self.fine_grained_labels = []
        for data_path, _ in self.samples:
            base_filename = os.path.basename(data_path).rsplit('.', 1)[0]
            label = get_fine_grained_label_from_filename(base_filename, FINE_GRAINED_LABEL_MAP)
            self.fine_grained_labels.append(label)

    def __getitem__(self, index):
        data_path, _ = self.samples[index]
        base_filename = os.path.basename(data_path).rsplit('.', 1)[0]
        pkl_path = os.path.join(self.pickle_path, self.split, f"{base_filename}.pkl")

        try:
            with open(pkl_path, 'rb') as f:
                pickle_data = pickle.load(f)
            video_full = pickle_data['data']
            
            # Frame sampling and padding
            num_frames = len(video_full)
            if num_frames >= self.segment_len:
                segment_np = video_full[:self.segment_len]
            else:
                indices = np.arange(num_frames)
                indices = np.tile(indices, (self.segment_len + num_frames - 1) // num_frames)
                segment_np = video_full[indices[:self.segment_len]]

            # Extract spatial mask for string tokens
            doppler_mask_tensor = torch.from_numpy(
                create_string_token_mask_size(segment_np, self.spatial_resolution)
            )

            # Preprocessing and formatting
            segment_tensor_TCHW = torch.from_numpy(segment_np.copy()).permute(0, 3, 1, 2)
            segment_tensor_resized = F.interpolate(segment_tensor_TCHW, size=(224, 224), mode='bilinear', align_corners=True)
           
            if self.transform:
                segment_tensor = self.transform(segment_tensor_resized)
            else:
                segment_tensor = F.interpolate(segment_tensor_TCHW.float() / 255.0, size=(224, 224), mode='bilinear', align_corners=True)

            # Label parsing (Target One-hot encoding and Raw integer)
            fine_grained_label = self.fine_grained_labels[index]
            label_one_hot = torch.zeros(self.num_classes)
            if fine_grained_label != -1: 
                label_one_hot[fine_grained_label] = 1.0
            
            fine_grained_label_tensor = torch.tensor(fine_grained_label, dtype=torch.long)

            return segment_tensor, doppler_mask_tensor, label_one_hot, fine_grained_label_tensor, data_path
            
        except Exception:
            # Silently handle exceptions and utilize collate_fn drop mechanisms
            return None

    def __len__(self):
        return len(self.samples)