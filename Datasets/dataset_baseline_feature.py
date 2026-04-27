import os
import sys
import re
import json
import random
from collections import defaultdict
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Sampler
import pickle

# Append parent directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
# Assuming create_aligned_mask_for_clip is imported from functions_baseline_feature or similar
from functions_set.functions_baseline_feature import create_aligned_mask_for_clip

# ==============================================================================
# 1. Global Fine-grained Label Mapping and Rules
# ==============================================================================

FINE_GRAINED_LABEL_MAP = {
    # Abnormal
    "VSD_view_1": 11, "VSD_view_2": 12, "VSD_view_3": 13, "VSD_view_4": 14, "VSD_view_5": 15,
    "ASD_view_6": 16, "ASD_view_7": 17, "ASD_view_2": 9, "ASD_view_5": 10,
    "PDA_view_8": 18,
    # Normal (Includes spelling correction fallback)
    "normal_view_1": 1, "normal_view_2": 2, "normal_view_3": 3, "normal_view_4": 4,
    "normal_view_5": 5, "normal_view_6": 6, "normal_view_7": 7, "normal_view_8": 8,
}

POSITIVE_TO_NEGATIVE_MAP = {
    11: 1, 12: 2, 13: 3, 14: 4, 15: 5,
    16: 6, 17: 7, 9: 2, 10: 5,
    18: 8,
}

# ==============================================================================
# 2. Unified Label Parsing Function
# ==============================================================================

def get_fine_grained_label_from_filename(filename, label_map):
    """
    A unified, robust function to return the fine-grained label based on the filename.
    """
    fn_lower = filename.lower()
    match = re.search(r'_label_(\w+)_view_(\d+)', fn_lower)
    
    if match:
        disease_lower = match.group(1)
        view_num = match.group(2)
        
        # --- Final Core Correction ---
        # Before constructing the key, convert the disease part to uppercase to match dictionary keys.
        # Handle 'normal' specially for robustness against typos.
        if disease_lower in ['normal', 'noraml']:
            disease_upper = 'normal'
        else:
            disease_upper = disease_lower.upper()
        
        key = f"{disease_upper}_view_{view_num}"
        # --- End Correction ---

        return label_map.get(key, -1)
    return -1

# ==============================================================================
# 3. Dataset and Sampling Classes
# ==============================================================================

class StructuredPairedBatchSampler(Sampler):
    """
    [Final Version] A custom Batch Sampler.
    Every generated batch consists of N complete (A, P, N1, N2) quadruplets.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size 
        self.quadruplets_per_batch = batch_size // 4 # Number of quadruplets per batch

        print("--- Initializing Batch Sampler ---")
        
        # 1. Group indices using pre-calculated labels from the dataset
        indices_by_label = defaultdict(list)
        for idx, label in enumerate(dataset.fine_grained_labels):
            if label != -1:
                indices_by_label[label].append(idx)

        # 2. Pre-construct index quadruplets for all possible valid combinations
        self.quadruplets = []
        positive_labels = [l for l in indices_by_label.keys() if l > 8]
        
        for pos_label in tqdm(positive_labels, desc="Building Quadruplets"):
            neg_label = POSITIVE_TO_NEGATIVE_MAP.get(pos_label)
            
            # Ensure sufficient samples exist for both positive and negative pairs
            if (neg_label is not None and
                len(indices_by_label.get(pos_label, [])) >= 2 and
                len(indices_by_label.get(neg_label, [])) >= 2):
                
                positive_indices = indices_by_label[pos_label]
                negative_indices = indices_by_label[neg_label]
                
                # Construct quadruplets by cycling through positive anchors
                for i in range(len(positive_indices)):
                    anchor_idx = positive_indices[i]
                    positive_idx = positive_indices[(i + 1) % len(positive_indices)]
                    neg_idx1, neg_idx2 = random.sample(negative_indices, 2)
                    self.quadruplets.append((anchor_idx, positive_idx, neg_idx1, neg_idx2))

        self.num_quadruplets = len(self.quadruplets)
        self.num_batches = self.num_quadruplets // self.quadruplets_per_batch
        
        print(f"Preprocessing complete: Found {self.num_quadruplets} valid quadruplets.")
        print(f"Each batch has {self.batch_size} samples, containing {self.quadruplets_per_batch} quadruplets.")

    def __iter__(self):
        # Shuffle the order of pre-constructed quadruplets
        shuffled_quadruplets = np.random.permutation(self.quadruplets).tolist()
        
        for i in range(self.num_batches):
            batch_indices = []
            start = i * self.quadruplets_per_batch
            end = start + self.quadruplets_per_batch
            
            for quad in shuffled_quadruplets[start:end]:
                batch_indices.extend(quad)
            
            # Optional: np.random.shuffle(batch_indices) # Leave commented out to preserve quadruplet order within batch
            yield batch_indices

    def __len__(self):
        return self.num_batches


class ContrastivePairedDataset(data.Dataset):
    """
    Dataset for loading paired samples and extracting corresponding Doppler masks.
    """
    def __init__(self, pickle_path, split_path, split, transform=None, segment_len=32):
        self.pickle_path = pickle_path
        self.split_path = split_path
        self.split = split
        self.transform = transform
        self.segment_len = segment_len
        
        # 1. Load raw sample list (Path, coarse label)
        with open(split_path, 'r') as f:
            dataset_splits = json.load(f)
        self.samples = list(dataset_splits[split].items())
        
        # [Important] Pre-calculate all fine-grained labels during initialization
        print(f"Dataset '{split}': Pre-calculating all fine-grained labels...")
        self.fine_grained_labels = []
        for data_path, _ in self.samples:
            base_filename = os.path.basename(data_path).rsplit('.', 1)[0]
            label = get_fine_grained_label_from_filename(base_filename, FINE_GRAINED_LABEL_MAP)
            self.fine_grained_labels.append(label)
        print("Label mapping complete.")

    def __getitem__(self, index):
        data_path, _ = self.samples[index]
        base_filename = os.path.basename(data_path).rsplit('.', 1)[0]
        pkl_path = os.path.join(self.pickle_path, self.split, f"{base_filename}.pkl")

        try:
            with open(pkl_path, 'rb') as f:
                pickle_data = pickle.load(f)
            video_full = pickle_data['data']
            
            # Frame temporal handling (padding or truncation)
            num_frames = len(video_full)
            if num_frames >= self.segment_len:
                segment_np = video_full[:self.segment_len]
            else:
                indices = np.arange(num_frames)
                indices = np.tile(indices, (self.segment_len + num_frames - 1) // num_frames)
                segment_np = video_full[indices[:self.segment_len]]

            # Generate Doppler blood flow mask
            doppler_mask_tensor = torch.from_numpy(create_aligned_mask_for_clip(segment_np))

            # Video preprocessing and data augmentation
            segment_tensor_TCHW = torch.from_numpy(segment_np.copy()).permute(0, 3, 1, 2)
            segment_tensor_resized = F.interpolate(segment_tensor_TCHW, size=(224, 224), mode='bilinear', align_corners=True)
            
            if self.transform:
                segment_tensor = self.transform(segment_tensor_resized)
            else:
                segment_tensor = segment_tensor_resized
            
            # Label Extraction
            fine_grained_label = self.fine_grained_labels[index]
            fine_grained_label_tensor = torch.tensor(fine_grained_label, dtype=torch.long)
            
            # Broad classification label (Map -1 and all normal<9 to 0)
            classification_label = fine_grained_label - 1 if fine_grained_label != -1 else 0            
            label_tensor = torch.tensor(classification_label, dtype=torch.long)
            
            return segment_tensor, doppler_mask_tensor, label_tensor, fine_grained_label_tensor, data_path

        except Exception as e:
            # Silently handle exceptions and utilize collate_fn drop mechanisms
            # print(f"Warning: Failed to load item at index {index} ({pkl_path}). Error: {e}")
            return None


    def __len__(self):
        return len(self.samples)