import argparse
import os
import glob
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.transforms import v2
import json

# Import utilities and model architectures (ensure proper import paths)
import utils
from nets_set.model_multi_views import MultiViewDualTokensFusionSize
from nets_set.model_single_view import ResnetTransformerDualTokensTemporalSpatialDecouplesize

class CHDCareAPI:
    def __init__(self, config_path_diag, config_path_xai, weights_path_diag, weights_path_xai, baseline_path_xai, device='cuda'):
        # Clean device string and check availability
        device_str = str(device).replace(' ', '').lower()
        if 'cuda' in device_str and not torch.cuda.is_available():
            print(f"[WARNING] No GPU detected. Falling back to 'cpu'")
            device_str = 'cpu'
            
        self.device = torch.device(device_str)
        print(f"[INFO] Initializing CHD-CARE API on {self.device}...")

        import yaml
        self.config_diag = yaml.load(open(config_path_diag, 'r'), Loader=yaml.FullLoader)
        self.config_xai = yaml.load(open(config_path_xai, 'r'), Loader=yaml.FullLoader)
        
        # 1. Initialize diagnosis model (multi-view)
        self.model_diag = MultiViewDualTokensFusionSize(
            view_num_classes=18, d_model=self.config_diag.get('d_model', 512),
            view_nhead=self.config_diag.get('view_nhead'), view_encoder_layers=self.config_diag.get('view_encoder_layers'),
            case_num_classes=self.config_diag['num_classes'],
            fusion_layers=self.config_diag.get('fusion_layers'), fusion_nhead=self.config_diag.get('fusion_nhead'),
            max_views=self.config_diag.get('max_views', 5), dropout=self.config_diag.get('dropout', 0.1),
            spatial_size=self.config_diag.get('spatial_size', 5)
        )
        self._load_weights(self.model_diag, weights_path_diag)
        self.model_diag = self.model_diag.to(self.device).eval()
        
        # 2. Initialize XAI model (single-view)
        self.spatial_res_xai = self.config_xai.get('spatial_resolution', 5)
        self.model_xai = ResnetTransformerDualTokensTemporalSpatialDecouplesize(
            num_classes=self.config_xai['num_classes'],
            d_model_cnn=self.config_xai['d_model_cnn'],
            num_layers=self.config_xai['num_layers'],
            dropout=self.config_xai['dropout'],
            spatial_resolution=self.spatial_res_xai
        )
        self._load_weights(self.model_xai, weights_path_xai, strict=False)
        self.model_xai = self.model_xai.to(self.device).eval()
        
        if os.path.exists(baseline_path_xai):
            self.baseline_xai = torch.load(baseline_path_xai, map_location='cpu').to(self.device)
        else:
             print(f"[WARNING] XAI baseline file not found: {baseline_path_xai}")
             self.baseline_xai = None
             
        self.transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
        ])
        print("[INFO] Initialization complete.")

    def _load_weights(self, model, path, strict=True):
        if path and os.path.exists(path):
            state_dict = torch.load(path, map_location='cpu')
            filter_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
            model.load_state_dict(filter_dict, strict=strict)
            print(f"       Loaded weights from {path}")
        else:
             print(f"[WARNING] Weights not found at {path}, using random initialization.")

    def _get_video_files(self, case_dir):
        """Helper: Get all video files in the target directory"""
        extensions = ['*.mp4', '*.avi', '*.mkv']
        video_files = []
        for ext in extensions:
            video_files.extend(glob.glob(os.path.join(case_dir, ext)))
            video_files.extend(glob.glob(os.path.join(case_dir, ext.upper())))
        return sorted(video_files)

    def diagnose(self, case_dir, output_dir):
        """
        Multi-view diagnosis mode
        """
        print(f"\n--- Running multi-view diagnosis (Case: {os.path.basename(case_dir)}) ---")
        os.makedirs(output_dir, exist_ok=True)
        
        video_files = self._get_video_files(case_dir)
        if not video_files:
            print(f"[ERROR] No video files found in {case_dir}!")
            return

        print(f"[INFO] Found {len(video_files)} videos, preparing for multi-view input...")
        
        valid_view_tensors = []
        valid_view_masks = []
        clip_size = self.config_diag.get('clip_size', 16)
        spatial_res = self.config_diag.get('spatial_size', 5)
        model_input_size = (224, 224)
        
        for vid_path in video_files:
            original_frames = utils.video_loader(vid_path)
            if original_frames is None or len(original_frames) < 1: continue

            indices = np.arange(len(original_frames))
            if len(original_frames) < clip_size:
                indices = np.tile(indices, (clip_size + len(original_frames) - 1) // len(original_frames))
            processed_16_frames = original_frames[indices[:clip_size]]

            roi_bbox = utils.find_doppler_roi_from_video(processed_16_frames, area_threshold=150, padding=20)
            if roi_bbox is None: continue

            frames_for_model_np = np.array([
                cv2.resize(frame[roi_bbox[1]:roi_bbox[3], roi_bbox[0]:roi_bbox[2]], model_input_size)
                for frame in processed_16_frames
            ])

            video_tensor_uint8 = torch.from_numpy(frames_for_model_np.copy()).permute(0, 3, 1, 2)
            input_tensor = self.transform(video_tensor_uint8) 
            
            mask_np = utils.create_string_token_mask_size(
                frames_for_model_np, spatial_resolution=spatial_res, vote_threshold=1
            )
            mask_tensor = torch.from_numpy(mask_np) 
            
            valid_view_tensors.append(input_tensor)
            valid_view_masks.append(mask_tensor)

        if not valid_view_tensors:
            print(f"[ERROR] Could not extract valid views with blood flow from this case.")
            return

        max_views = self.config_diag.get('max_views', 5)
        actual_views = len(valid_view_tensors)
        
        if actual_views > max_views:
            valid_view_tensors = valid_view_tensors[:max_views]
            valid_view_masks = valid_view_masks[:max_views]
            actual_views = max_views
            
        num_padding = max_views - actual_views
        if num_padding > 0:
            last_tensor = valid_view_tensors[-1]
            last_mask = valid_view_masks[-1]
            valid_view_tensors.extend([last_tensor] * num_padding)
            valid_view_masks.extend([last_mask] * num_padding)
            
        final_video_batch = torch.stack(valid_view_tensors).unsqueeze(0).to(self.device, non_blocking=True)
        final_mask_batch = torch.stack(valid_view_masks).unsqueeze(0).to(self.device, non_blocking=True)
        num_views_tensor = torch.tensor([actual_views], dtype=torch.long).to(self.device, non_blocking=True)
        
        # 3. Model inference and probability conversion
        print(f"[INFO] Performing multi-view diagnosis (using {actual_views} valid views)...")
        with torch.no_grad():
            with autocast():
                logits_temporal = self.model_diag(final_video_batch, num_views_tensor, final_mask_batch) 
            
            preds_temporal = torch.argmax(logits_temporal, dim=1)
            probs_temporal = F.softmax(logits_temporal, dim=1)
            
        pred_label_idx = preds_temporal.item()
        probabilities = probs_temporal[0].cpu().numpy().tolist()
        
        # Format as confidence dictionary for four subtypes
        class_names = ["Normal", "VSD", "ASD", "PDA"]
        confidences = {}
        for i, class_name in enumerate(class_names):
            if i < len(probabilities):
                confidences[class_name] = round(probabilities[i], 4)
        
        case_id = os.path.basename(case_dir)
        diagnosis_str = class_names[pred_label_idx] if pred_label_idx < len(class_names) else f"Unknown_Index_{pred_label_idx}"
        
        case_info = {
            "Case_ID": case_id,
            "Diagnosis": diagnosis_str,
            "Class_Index": pred_label_idx,
            "Confidences": confidences,
            "Valid_Views_Used": actual_views
        }
        
        out_path = os.path.join(output_dir, f'Diagnosis_{case_id}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(case_info, f, indent=4, ensure_ascii=False)
            
        print(f"[SUCCESS] Diagnosis complete. Result: {diagnosis_str}. Report saved to {out_path}")
        return pred_label_idx


    def visualize(self, case_dir, output_dir):
         """
         Single-view XAI visualization mode: Saves heatmaps (and bboxes if anomalous), 
         and automatically generates a keyframe index summary JSON file.
         """
         import utils
         import cv2
         import numpy as np
         import json

         print(f"\n--- Running single-view XAI visualization (Case: {os.path.basename(case_dir)}) ---")
         os.makedirs(output_dir, exist_ok=True)
         
         video_files = self._get_video_files(case_dir)
         if not video_files: return
             
         clip_size = self.config_xai.get('clip_size', 16)
         spatial_res = self.spatial_res_xai
         model_input_size = (224, 224)
         final_box_size = 56 

         POSITIVE_TO_NEGATIVE_PAIR_MAP = {
             10: 0, 11: 1, 12: 2, 13: 3, 14: 4,
             15: 5, 16: 6, 8: 1, 9: 4,
             17: 7
         }

         # Dictionary to collect keyframe index summary for each video
         keyframes_summary = {}

         for i, video_path in enumerate(video_files):
             vid_name = os.path.basename(video_path)
             vid_prefix = vid_name.split('.')[0]
             print(f"  -> [{i+1}/{len(video_files)}] Processing: {vid_name}")
             
             original_frames = utils.video_loader(video_path)
             if original_frames is None or len(original_frames) < 1: continue

             indices = np.arange(len(original_frames))
             if len(original_frames) < clip_size:
                 indices = np.tile(indices, (clip_size + len(original_frames) - 1) // len(original_frames))
             processed_16_frames = original_frames[indices[:clip_size]]

             roi_bbox = utils.find_doppler_roi_from_video(processed_16_frames, area_threshold=150, padding=20)
             if roi_bbox is None: 
                 print("     [SKIP] No obvious blood flow ROI detected.")
                 keyframes_summary[vid_name] = "No obvious blood flow ROI detected, visualization skipped"
                 continue

             frames_for_model_np = np.array([
                 cv2.resize(frame[roi_bbox[1]:roi_bbox[3], roi_bbox[0]:roi_bbox[2]], model_input_size)
                 for frame in processed_16_frames
             ])

             video_tensor_uint8 = torch.from_numpy(frames_for_model_np.copy()).permute(0, 3, 1, 2)
             input_tensor = self.transform(video_tensor_uint8).unsqueeze(0).to(self.device)
             
             mask_np = utils.create_string_token_mask_size(
                 frames_for_model_np, spatial_resolution=spatial_res, vote_threshold=1
             )
             doppler_mask = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)

             # --- 1. Model Forward Pass ---
             with torch.no_grad():
                 outputs = self.model_xai(input_tensor, None, doppler_mask)
                 
             pred_label_idx = torch.argmax(outputs["predicted_scores"], dim=1).item()
             
             # Check if the predicted category belongs to a normal/negative class
             is_normal = (pred_label_idx <= 7)
             
             # --- 2. Extract Attention Score Distribution ---
             attention_vector = outputs["all_layer_attention_scores"][-1][0, pred_label_idx, :]
             
             # --- 3. Identify Key Frame & Log to Summary ---
             if is_normal:
                 print(f"     [INFO] Predicted as normal (Index: {pred_label_idx}), only heatmap will be saved.")
                 # For negative cases, default to the median frame as the representative view
                 key_frame_idx = clip_size // 2 
                 # Log negative statement as requested
                 keyframes_summary[vid_name] = "Predicted as negative view, no keyframe or defect box available"
             else:
                 if self.baseline_xai is None:
                     print("     [ERROR] Missing baseline file, cannot determine keyframe.")
                     keyframes_summary[vid_name] = "Missing baseline features, cannot locate keyframe"
                     continue
                     
                 key_patch_idx = torch.argmax(attention_vector).item()
                 key_patch_raw_string = outputs["raw_strings_for_loss"][0, key_patch_idx, :, :]
                 
                 normal_id = POSITIVE_TO_NEGATIVE_PAIR_MAP.get(pred_label_idx)
                 if normal_id is not None:
                     key_patch_baseline = self.baseline_xai[normal_id, key_patch_idx, :] 
                 else:
                     key_patch_baseline = torch.mean(self.baseline_xai, dim=0)[key_patch_idx, :] 
                 
                 residuals = torch.linalg.norm(key_patch_raw_string - key_patch_baseline.unsqueeze(0), dim=-1)
                 key_frame_idx = torch.argmax(residuals).item()
                 print(f"     [INFO] Key frame locked: Frame {key_frame_idx} (based on baseline residuals)")
                 # Log positive keyframe index
                 keyframes_summary[vid_name] = int(key_frame_idx)

             # --- 4. Heatmap Post-processing & Visualization ---
             heatmap_small = attention_vector.cpu().numpy().reshape(spatial_res, spatial_res)
             heatmap_224 = cv2.resize(heatmap_small, model_input_size, interpolation=cv2.INTER_LINEAR)
             
             if heatmap_224.max() > heatmap_224.min():
                 heatmap_norm = (heatmap_224 - heatmap_224.min()) / (heatmap_224.max() - heatmap_224.min())
             else:
                 heatmap_norm = np.zeros(model_input_size, dtype=np.float32)

             rx1, ry1, rx2, ry2 = roi_bbox

             # [Image A: Superimpose heatmap overlay on original frame for BOTH positive and negative cases]
             frame_overlay = original_frames[key_frame_idx].copy()
             heatmap_roi = cv2.resize(heatmap_norm, (rx2 - rx1, ry2 - ry1))
             heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_roi), cv2.COLORMAP_JET)
             
             roi_original = frame_overlay[ry1:ry2, rx1:rx2]
             blended_roi = cv2.addWeighted(roi_original, 0.6, heatmap_colored, 0.4, 0)
             frame_overlay[ry1:ry2, rx1:rx2] = blended_roi
             
             overlay_path = os.path.join(output_dir, f"{vid_prefix}_Heatmap.png")
             cv2.imwrite(overlay_path, frame_overlay)
             print(f"     [SUCCESS] Heatmap saved to: {overlay_path}")

             # [Image B: Generate defect bounding box ONLY for positive anomalous cases]
             if not is_normal:
                 peak_y, peak_x = np.unravel_index(np.argmax(heatmap_224), heatmap_224.shape)
                 cx, cy, half_size = peak_x, peak_y, final_box_size // 2
                 box_model = [max(0, cx-half_size), max(0, cy-half_size), min(224, cx+half_size), min(224, cy+half_size)]
                     
                 mx1, my1, mx2, my2 = box_model
                 scale_x_inv, scale_y_inv = (rx2 - rx1) / 224.0, (ry2 - ry1) / 224.0
                 box_orig = [
                     int(mx1 * scale_x_inv + rx1), int(my1 * scale_y_inv + ry1),
                     int(mx2 * scale_x_inv + rx1), int(my2 * scale_y_inv + ry1)
                 ]
                 
                 frame_bbox = original_frames[key_frame_idx].copy()
                 cv2.rectangle(frame_bbox, (box_orig[0], box_orig[1]), (box_orig[2], box_orig[3]), (0, 255, 0), 3)
                 cv2.putText(frame_bbox, "CHD-CARE AI", (box_orig[0], max(20, box_orig[1] - 10)), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                     
                 bbox_path = os.path.join(output_dir, f"{vid_prefix}_BBox.png")
                 cv2.imwrite(bbox_path, frame_bbox)
                 print(f"     [SUCCESS] Localization box saved to: {bbox_path}")

         # --- 5. Save Keyframe Summary Report as JSON File after the loop ---
         summary_json_path = os.path.join(output_dir, "XAI_KeyFrames_Summary.json")
         with open(summary_json_path, 'w', encoding='utf-8') as f:
             json.dump(keyframes_summary, f, indent=4, ensure_ascii=False)
             
         print(f"[SUCCESS] Keyframe summary saved to: {summary_json_path}")
         print(f"[SUCCESS] Explainable evidence generation complete for this case.")


def parse_args():
    parser = argparse.ArgumentParser(description="CHD-CARE: Diagnosis and Visualization API")
    
    parser.add_argument('--case_dir', type=str, required=True, help="Path to folder containing all ultrasound videos for a single case")
    parser.add_argument('--task', type=str, required=True, choices=['diagnose', 'visualize', 'both'], 
                        help="Select task: 'diagnose', 'visualize', or 'both'")
    
    # Default output path set to parent directory CHD_CARE_Results
    parser.add_argument('--output_dir', type=str, default='../CHD-CARE/Output_Results', help="Directory to save results (default: parent directory)")
    parser.add_argument('--device', type=str, default='cuda', help="Computation device (e.g., 'cuda:0')")
    
    parser.add_argument('--diag_config', type=str, default='./Config/train_config_mutil_views.yaml')
    parser.add_argument('--diag_weights', type=str)
    parser.add_argument('--xai_config', type=str, default='./Config/test_config_single_view_for_XAI_visual.yaml')
    parser.add_argument('--xai_weights', type=str)
    parser.add_argument('--xai_baseline', type=str)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    api = CHDCareAPI(
        config_path_diag=args.diag_config,
        config_path_xai=args.xai_config,
        weights_path_diag=args.diag_weights,
        weights_path_xai=args.xai_weights,
        baseline_path_xai=args.xai_baseline,
        device=args.device
    )
    
    # Extract case name (e.g., 'patient_001')
    case_name = os.path.basename(os.path.normpath(args.case_dir))
    
    # Create dedicated folder for current case: Results/patient_001
    patient_out_dir = os.path.join(args.output_dir, case_name)
    
    if args.task in ['diagnose', 'both']:
        # Diagnosis report saved directly in case directory
        api.diagnose(args.case_dir, patient_out_dir)
             
    if args.task in ['visualize', 'both']:
        # Visualizations saved in XAI_Visuals subfolder
        xai_out_dir = os.path.join(patient_out_dir, 'XAI_Visuals')
        api.visualize(args.case_dir, xai_out_dir)

if __name__ == "__main__":
    main()