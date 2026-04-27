import os
import json
import cv2
import numpy as np
import pickle  # Import pickle library
from tqdm import tqdm
from utils import *

# Corrected variable name from SPILT_PATH to SPLIT_PATH
SPLIT_PATH = './dataset_split_0303_WoFN_comparison_final.json'
OUTPUT_DIR = './pickle_cropped_data_0303_WoFN_comparsion_final'

# Hyperparameters for Region of Interest (ROI) extraction
CLIP_SIZE = 32
ROI_AREA_THRESHOLD = 150 # Tuned area threshold
ROI_PADDING = 20


def video_to_npy(video_path):
    """ Convert video file to a Numpy array of shape (T, H, W, C) """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    return np.array(frames)

def process_split(split_data, output_dir, split_name):
    """
    Modified version: Process a single data split, perform ROI cropping, and save as a pickle file.
    """
    output_split_dir = os.path.join(output_dir, split_name)
    os.makedirs(output_split_dir, exist_ok=True)
    
    for video_path, label in tqdm(split_data.items(), desc=f"Processing {split_name} split"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        pickle_path = os.path.join(output_split_dir, f"{video_name}.pkl")
        
        if os.path.exists(pickle_path):
            continue

        video_npy = video_to_npy(video_path)
        if video_npy.shape[0] < 2:  # Video is too short to process
            print(f"Warning: Video {video_name} has too few frames, skipped.")
            continue

        # --- Core modification track 1: ROI localization ---
        roi_bbox = find_doppler_roi_from_video(
            video_npy, 
            clip_size=CLIP_SIZE, 
            area_threshold=ROI_AREA_THRESHOLD,
            padding=ROI_PADDING
        )

        # --- Core modification track 2: Video frame cropping ---
        if roi_bbox:
            x1, y1, x2, y2 = roi_bbox
            # Efficiently crop all frames using list comprehension
            cropped_frames = [frame[y1:y2, x1:x2] for frame in video_npy]
            final_video_npy = np.array(cropped_frames)
            print(f"  -> Video {video_name} successfully cropped, ROI: {roi_bbox}")
        else:
            # If no ROI is found, skip this video to ensure dataset consistency
            print(f"  -> Warning: No valid ROI found for video {video_name}, skipped.")
            continue
            # # Alternatively, the original video can be used:
            # final_video_npy = video_npy

        # --- Package and save ---
        dataset_dict = {
            "data": final_video_npy,
            "label": label
        }
        with open(pickle_path, "wb") as f:
            pickle.dump(dataset_dict, f)

if __name__ == "__main__":
    # Read split data
    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        total_split = json.load(f)
    
    # Output directory (modifiable as needed)
    print(f"All pre-cropped pickle files will be saved to: {OUTPUT_DIR}")

    # Process each split
    process_split(total_split["train"], OUTPUT_DIR, "train")
    process_split(total_split["val"], OUTPUT_DIR, "val")
    process_split(total_split["test"], OUTPUT_DIR, "test")