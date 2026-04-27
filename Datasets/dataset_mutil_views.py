import os
import json
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import pickle
import random
import cv2  # Added missing import for cv2
import tqdm
from utils import *

class MultiViewDataset(data.Dataset):
    def __init__(self, pickle_path, split_path, split, transform, clip_size=64, random_clip=True, max_views=5, spatial_resolution=4, patch_mask_prob=0.2):
        """
        [Modified Version]
        Dataset class for multi-view models, supporting view count padding.

        Args:
            pickle_path (str): Root directory containing pickle files.
            split_path (str): Path to the multi-view JSON split file.
            split (str): Dataset partition ('train', 'val', or 'test').
            transform (callable, optional): Data augmentations applied to video frames.
            clip_size (int): Number of frames per video segment.
            max_views (int): Maximum number of views per case (used for padding and truncation).
            spatial_resolution (int): Resolution of the spatial grid for patch masking.
            patch_mask_prob (float): Probability of applying random patch masking during training.
        """

        with open(split_path, 'r') as f:
            # self.case_data is a dictionary keyed by case_id
            self.case_data = json.load(f)[split]
        
        # Convert dictionary keys (case_id) to a list for index-based access
        self.case_ids = list(self.case_data.keys())
        
        self.pickle_path = pickle_path
        self.split = split
        self.transform = transform
        self.clip_size = clip_size
        self.max_views = max_views
        self.spatial_resolution = spatial_resolution
        self.patch_mask_prob = patch_mask_prob

    def get_labels(self):
        """Retrieve a list of labels for all samples (utilized for calculating sampling weights)."""
        return [details['label'] for details in list(self.case_data.values())]
   
    def __getitem__(self, index):
        case_id = self.case_ids[index]
        case_info = self.case_data[case_id]

        # 1. Extract all video paths for this case from the JSON configuration
        view_paths = case_info['videos']

        ### [Core Modification] Random View Sampling ###
        # If the current case exceeds the view limit and is in training mode
        if len(view_paths) > self.max_views and self.split == 'train':
            # Randomly sample max_views without replacement from available views
            view_paths = random.sample(view_paths, self.max_views)
        else:
            # For validation/test modes, or if views are insufficient, truncate to the first max_views
            view_paths = view_paths[:self.max_views]
            
        view_data_pairs = []
        
        # Iterate over view paths, load, and process videos
        for video_path in view_paths:
            # Terminate if the maximum number of loaded views is reached
            if len(view_data_pairs) >= self.max_views:
                break
                
            base_filename = os.path.splitext(os.path.basename(video_path))[0]
            pkl_path = os.path.join(self.pickle_path, self.split, f"{base_filename}.pkl")
            
            try:
                with open(pkl_path, 'rb') as f:
                    pickle_data = pickle.load(f)
            
                video_full = pickle_data['data']
                
                # --- Video Clipping ---
                if video_full.shape[0] >= self.clip_size:
                    start_frame = random.randint(0, video_full.shape[0] - self.clip_size) if self.split == 'train' else 0
                    video_clipped = video_full[start_frame : start_frame + self.clip_size]
                else:
                    padding_needed = self.clip_size - video_full.shape[0]
                    padding_frames = np.repeat(video_full[0:1], padding_needed, axis=0)
                    video_clipped = np.concatenate([video_full, padding_frames], axis=0)

                # --- [Core Addition: Generate Mask for Processed Video] ---
                # Note: The mask is generated on raw, unnormalized images prior to transformations
                blood_flow_mask = create_string_token_mask_size(
                    video_clipped, 
                    spatial_resolution=self.spatial_resolution
                )

                # --- [Core Addition 2: Random Patch Masking (Data Augmentation)] ---
                # Execute strictly during training when probability conditions are met
                if self.split == 'train' and self.patch_mask_prob > 0 and random.random() < self.patch_mask_prob:
                    # Identify all 'valid' patch indices (where mask evaluates to False)
                    valid_indices = np.where(blood_flow_mask == False)[0]
                    
                    if len(valid_indices) > 0:
                        # Randomly designate a proportion of valid regions to mask (e.g., 20% - 40%)
                        mask_ratio = random.uniform(0.2, 0.4)
                        num_to_mask = int(len(valid_indices) * mask_ratio)
                        
                        if num_to_mask > 0:
                            # Randomly select indices for masking
                            indices_to_mask = np.random.choice(valid_indices, num_to_mask, replace=False)
                            # Set identified positions to True (masked state)
                            blood_flow_mask[indices_to_mask] = True

                # --- Tensor Conversion and Augmentation ---  
                # Transpose dimensions: (T, H, W, C) -> (T, C, H, W)
                video_tensor = torch.from_numpy(video_clipped.copy()).permute(0, 3, 1, 2)

                # Enforce size standardization prior to applying subsequent transformations.
                # Guarantees uniform input dimensions for augmentations regardless of original resolution.
                # Utilizes F.interpolate for optimal scaling efficiency.
                if video_tensor.shape[-2:] != (224, 224):
                    video_tensor = F.interpolate(video_tensor, size=(224, 224), mode='bilinear', align_corners=False)

                if self.transform:
                    video_tensor = self.transform(video_tensor)
                
                view_data_pairs.append((video_tensor, blood_flow_mask))

            except Exception as e:
                # Bypass current view and proceed if loading or processing fails
                continue
        
        if not view_data_pairs:
            # Return None if zero valid views are processed (requires collate_fn handling)
            return None
        
        # Execute padding to standardize the number of views per batch
        num_valid_views = len(view_data_pairs)
        padding_needed = self.max_views - num_valid_views
        
        if padding_needed > 0:
            # Extract dimensionality from an existing valid sample
            video_sample = view_data_pairs[0][0] # (T, C, H, W)
            mask_sample = view_data_pairs[0][1]  # (S,)
            
            # Construct zero-padded arrays
            zero_padding_view = torch.zeros_like(video_sample)
            
            # Pad mask uniformly with True (denoting full occlusion/ignorance)
            true_padding_mask = np.ones_like(mask_sample, dtype=bool)

            for _ in range(padding_needed):
                view_data_pairs.append((zero_padding_view, true_padding_mask))
        
        # --- Tensor Stacking ---
        # video_list: List[(T, C, H, W)] -> Stack -> (V, T, C, H, W)
        video_list = [item[0] for item in view_data_pairs]
        # mask_list: List[(S,)] -> Stack -> (V, S) -> Tensor
        mask_list = [torch.from_numpy(item[1]) for item in view_data_pairs]

        multi_view_video = torch.stack(video_list, dim=0)
        multi_view_mask = torch.stack(mask_list, dim=0)
        
        label = torch.tensor(case_info['label'], dtype=torch.long)
        num_valid_views_tensor = torch.tensor(num_valid_views, dtype=torch.long)
        
        # Condition outputs based on split partition (Test mode appends Case ID for inference tracking)
        if self.split == 'test':
            return multi_view_video, multi_view_mask, label, num_valid_views_tensor, case_id
        else:
            return multi_view_video, multi_view_mask, label, num_valid_views_tensor

    def __len__(self):
        return len(self.case_ids)