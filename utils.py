import numpy as np
import cv2
from PIL import Image
import PIL
import random
import torch
import json
import os

def pad_collate_fn(batch):
    """
    Custom collate function for multi-view data with variable view counts per case.
    Pads all samples in the batch to the maximum number of views in the batch.
    """
    # 1. Filter out None samples that failed to load
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # 2. Find maximum number of views in current batch
    max_views = max([item[0].shape[0] for item in batch])  # item[0] is videos_tensor

    # 3. Prepare lists for padded data
    padded_videos, padded_masks, labels, num_views_list, padded_orig_frames, case_ids = [], [], [], [], [], []

    # 4. Iterate each sample in the batch
    for videos, masks, label, num_views, orig_frames, case_id in batch:
        num_padding = max_views - num_views
        
        # a. Apply padding if needed
        if num_padding > 0:
            # Repeat the last valid view for padding
            video_pad = videos[-1].unsqueeze(0).repeat(num_padding, 1, 1, 1, 1)
            videos = torch.cat([videos, video_pad], dim=0)

            mask_pad = masks[-1].unsqueeze(0).repeat(num_padding, 1, 1, 1)
            masks = torch.cat([masks, mask_pad], dim=0)

            orig_frame_pad = orig_frames[-1].unsqueeze(0).repeat(num_padding, 1, 1, 1, 1)
            orig_frames = torch.cat([orig_frames, orig_frame_pad], dim=0)

        padded_videos.append(videos)
        padded_masks.append(masks)
        labels.append(label)
        num_views_list.append(num_views)  # Store REAL view count BEFORE padding
        padded_orig_frames.append(orig_frames)
        case_ids.append(case_id)

    # 5. Stack padded lists into final batch tensors
    final_videos = torch.stack(padded_videos)
    final_masks = torch.stack(padded_masks)
    final_labels = torch.stack(labels)
    final_num_views = torch.tensor(num_views_list, dtype=torch.long)
    final_orig_frames = torch.stack(padded_orig_frames)
    
    return final_videos, final_masks, final_labels, final_num_views, final_orig_frames, case_ids

def remove_info(array):
    """Remove static UI/info overlay from the ultrasound image."""
    video = array.copy()
    l = video.shape[0] // 2
    v1 = video[:l]
    v2 = video[-l:]
    mask = np.sum(v1 - v2, axis=0)
    video[:, mask == 0] = 0
    return video

def convertRGB(array):
    """Convert grayscale array to RGB format."""
    shape = array.shape
    assert len(shape) in [2, 3]
    if len(shape) == 2:  # Single image
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    else:  # Video clip
        array = np.array([cv2.cvtColor(a, cv2.COLOR_GRAY2RGB) for a in array])
    return array

def crop_resize_video(array, size=256, convert_RGB=True):
    """Make frames square then resize to target size."""
    imgs = [make_img_square(a) for a in array]
    video = np.array([resize_image(a, size, convert_RGB) for a in imgs])
    return video

def make_img_square(array):
    """Pad image to square shape using constant padding."""
    if array.shape[0] != array.shape[1]:
        diff = abs(array.shape[1] - array.shape[0]) // 2
        if len(array.shape) == 3:
            if array.shape[0] < array.shape[1]:
                array = np.pad(array, ((diff, diff), (0, 0), (0, 0)), mode='constant')
            else:
                array = np.pad(array, ((0, 0), (diff, diff), (0, 0)), mode='constant')
        else:
            if array.shape[0] < array.shape[1]:
                array = np.pad(array, ((diff, diff), (0, 0)), mode='constant')
            else:
                array = np.pad(array, ((0, 0), (diff, diff)), mode='constant')
    return array

def resize_image(array, size, convert_RGB=True):
    """Resize image using bilinear interpolation."""
    img = Image.fromarray(array.astype('uint8'))
    if convert_RGB:
        img = img.convert('RGB')
    img = img.resize((size, size), resample=PIL.Image.BILINEAR)
    return np.array(img)

def clip_video(data_path, video, size=64, random_clip=False):
    """
    Clip or loop video to fixed target length.
    Uses looping to fill frames if video is too short.
    """
    video0 = [v for v in video]
    video = [v for v in video]
    while len(video) < size:
        video += video0
    try:
        start = random.choice(range(len(video) - size))
    except:
        start = 0
    if not random_clip:
        start = 0
    video = video[start:start + size]
    video = np.array(video)
    return video

def create_attention_mask(batch_size, seq_len, indices_to_mask, device):
    """Create boolean attention mask (True positions are ignored/masked)."""
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for i in indices_to_mask:
        if 0 <= i < seq_len:
            mask[:, i] = True
    return mask

def video_loader(path):
    """
    Load video from path and return as RGB numpy array in (T, H, W, C) format.
    """
    if not os.path.exists(path):
        print(f"Error: Video file not found: {path}")
        return None
        
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        print(f"Error: OpenCV cannot open video: {path}")
        cap.release()
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()

    if not frames:
        print(f"Warning: Empty or corrupted video: {path}")
        return None
        
    return np.array(frames)

def process_video_for_inference(video_frames, target_len, resize_dim):
    """
    Process video for model inference.
    Returns processed frames and original frame index mapping.
    """
    # 1. Remove static UI information
    processed_frames = remove_info(video_frames.copy())

    # 2. Temporal processing (loop padding or truncation)
    original_indices = list(range(processed_frames.shape[0]))
    video_list = list(processed_frames)

    # Loop padding if frame count is insufficient
    if len(video_list) < target_len:
        print(f"Frames ({len(video_list)}) < {target_len}, applying loop padding...")
        original_video_copy = video_list.copy()
        original_indices_copy = original_indices.copy()
        while len(video_list) < target_len:
            video_list.extend(original_video_copy)
            original_indices.extend(original_indices_copy)

    # Truncate to target length
    final_frames_list = video_list[:target_len]
    final_index_map = original_indices[:target_len]
    
    final_frames_np = np.array(final_frames_list)
    
    # 3. Spatial processing (square pad + resize)
    print(f"Processing images (Pad-to-Square & Resize to {resize_dim}x{resize_dim})...")
    final_frames_np = crop_resize_video(final_frames_np, size=resize_dim, convert_RGB=False)

    return final_frames_np, final_index_map

def find_static_content_mask(frame1, frame2, diff_threshold=5):
    """Compare two consecutive frames to find identical (static) regions."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, static_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY_INV)
    return static_mask

def find_doppler_roi_from_video(video_frames, area_threshold, padding):
    """ROI extraction using color Doppler and motion analysis."""
    max_area = 0
    best_frame_idx = -1
    for i, frame in enumerate(video_frames):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        lower_blue = np.array([100, 70, 50])
        upper_blue = np.array([130, 255, 255])
        lower_turb = np.array([11, 70, 50])
        upper_turb = np.array([90, 255, 255])
        
        mask_r1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
        mask_r2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        mask_b = cv2.inRange(hsv_img, lower_blue, upper_blue)
        mask_t = cv2.inRange(hsv_img, lower_turb, upper_turb)
        current_area = np.sum((mask_r1 + mask_r2 + mask_b + mask_t) > 0)
        
        if current_area > max_area:
            max_area = current_area
            best_frame_idx = i

    if best_frame_idx == -1 or best_frame_idx + 1 >= len(video_frames):
        return None
    
    best_frame = video_frames[best_frame_idx]
    next_frame = video_frames[best_frame_idx + 1]
    static_mask = find_static_content_mask(best_frame, next_frame)
    hsv_best = cv2.cvtColor(best_frame, cv2.COLOR_BGR2HSV)
    
    mask_r1 = cv2.inRange(hsv_best, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv_best, lower_red2, upper_red2)
    mask_b = cv2.inRange(hsv_best, lower_blue, upper_blue)
    mask_t = cv2.inRange(hsv_best, lower_turb, upper_turb)
    color_mask = cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2), cv2.bitwise_or(mask_b, mask_t))
    cleaned_color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(static_mask))
    contours, _ = cv2.findContours(cleaned_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_threshold]

    if not final_contours:
        return None

    all_points = np.concatenate(final_contours)
    x, y, w, h = cv2.boundingRect(all_points)
    center_x, center_y = x + w // 2, y + h // 2
    max_side = max(w, h) + padding * 2
    x1 = max(0, center_x - max_side // 2)
    y1 = max(0, center_y - max_side // 2)
    x2 = min(best_frame.shape[1], center_x + max_side // 2)
    y2 = min(best_frame.shape[0], center_y + max_side // 2)
    return (x1, y1, x2, y2)

def create_aligned_mask_for_clip(segment_np, patch_size=32, threshold_ratio=0.5):
    """
    Generate aligned (16, 7, 7) boolean attention mask from video clip.
    True = patch should be ignored.
    """
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([100, 70, 50])
    upper_blue = np.array([130, 255, 255])
    lower_turb = np.array([11, 70, 50])
    upper_turb = np.array([90, 255, 255])
    
    area_threshold = (patch_size * patch_size) * threshold_ratio
    final_masks = []
    
    for frame_bgr in segment_np:
        frame_resized = cv2.resize(frame_bgr, (224, 224))
        num_patches_h = 224 // patch_size
        num_patches_w = 224 // patch_size
        frame_attn_mask = np.zeros((num_patches_h, num_patches_w), dtype=bool)
        
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch = frame_resized[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                
                mask_r1 = cv2.inRange(hsv_patch, lower_red1, upper_red1)
                mask_r2 = cv2.inRange(hsv_patch, lower_red2, upper_red2)
                mask_b = cv2.inRange(hsv_patch, lower_blue, upper_blue)
                mask_t = cv2.inRange(hsv_patch, lower_turb, upper_turb)
                color_mask_in_patch = cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2), cv2.bitwise_or(mask_b, mask_t))
                
                blood_flow_area = np.sum(color_mask_in_patch > 0)
                if blood_flow_area < area_threshold:
                    frame_attn_mask[i, j] = True
                    
        final_masks.append(frame_attn_mask)
        
    return np.array(final_masks, dtype=bool)

def create_string_token_mask_size(segment_np,
                                 spatial_resolution,
                                 model_input_size=(224, 224),
                                 vote_threshold=1,
                                 area_threshold_ratio=0.05):
    """Create spatial token mask based on Doppler color detection."""
    num_timesteps = segment_np.shape[0]
    num_spatial_locations = spatial_resolution * spatial_resolution
    
    patch_size_h = model_input_size[0] // spatial_resolution
    patch_size_w = model_input_size[1] // spatial_resolution
    
    area_threshold = (patch_size_h * patch_size_w) * area_threshold_ratio

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([100, 70, 50])
    upper_blue = np.array([130, 255, 255])
    lower_turb = np.array([11, 70, 50])
    upper_turb = np.array([90, 255, 255])

    vote_matrix = np.zeros((num_timesteps, num_spatial_locations), dtype=bool)
    
    for t, frame_bgr in enumerate(segment_np):
        frame_resized = cv2.resize(frame_bgr, model_input_size)
        hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        
        mask_r1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask_r2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        mask_b = cv2.inRange(hsv_frame, lower_blue, upper_blue)
        mask_t = cv2.inRange(hsv_frame, lower_turb, upper_turb)
        color_mask = cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2), cv2.bitwise_or(mask_b, mask_t))
        
        for s in range(num_spatial_locations):
            row = s // spatial_resolution
            col = s % spatial_resolution
            
            patch_mask = color_mask[
                row * patch_size_h : (row + 1) * patch_size_h,
                col * patch_size_w : (col + 1) * patch_size_w
            ]
            
            blood_flow_area = np.sum(patch_mask > 0)
            if blood_flow_area > area_threshold:
                vote_matrix[t, s] = True

    num_votes = np.sum(vote_matrix, axis=0)
    string_mask = (num_votes < vote_threshold)
            
    return string_mask
    
def parse_doctor_annotations(json_path, frame_limit=16):
    """Parse doctor's bounding box annotations from JSON file."""
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    annotations = {}
    bounding_box_model = data.get("Models", {}).get("BoundingBoxLabelModel", [])
    if not bounding_box_model:
        return annotations
    for item in bounding_box_model:
        frame_idx = item.get("FrameCount")
        if frame_idx is not None and frame_idx < frame_limit:
            p1, p2 = item.get("p1"), item.get("p2")
            if p1 and p2:
                annotations[frame_idx] = [
                    min(p1[0], p2[0]), min(p1[1], p2[1]),
                    max(p1[0], p2[0]), max(p1[1], p2[1])
                ]
    return annotations