import argparse
import yaml
config_path ='./Config/test_config_single_view_for_XAI_visual.yaml'
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
for c in config:
    print(c, config[c])
cuda_device = config['cuda_device']

import os
import sys
import re
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from nets.model_single_view import ResnetTransformerDualTokensTemporalSpatialDecouplesizeGradCam
from functions.functions_single_view import *
from torchvision.transforms import v2
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from utils import *
import matplotlib.patches as patches
import cv2

# Hyper-parameters
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

d_model_transformer = config['d_model_transformer']
d_model_cnn = config['d_model_cnn']
compressed_dim = config['compressed_dim']
nhead = config['nhead']
num_layers = config['num_layers']
dropout = config['dropout']

fix_randomness(seed)

# Paths and constants
DATASET_ROOT_DIR = '/mnt/data1/xxx/CHD_1001/organized_test_dataset'
OUTPUT_DIR = '/mnt/data1/xxx/CHD_1001/Evaluation_Results_VisualV2/DualTokensSize_TS_Heatmap_gradcampp_IOU'
DISEASE_SUBTYPES = ['VSD','ASD','PDA']
IOU_THRESHOLDS = [
    0.02, 0.04, 0.06, 0.08, 0.1,
    0.12, 0.14, 0.16, 0.18, 0.2,
    0.22, 0.24, 0.26, 0.28, 0.3,
    0.32, 0.34, 0.36, 0.38, 0.4,
    0.42, 0.44, 0.46, 0.48, 0.5,
]

ROI_AREA_THRESHOLD = 150
ROI_PADDING = 20
MODEL_INPUT_SIZE = (224, 224)
SPATIAL_RESOLUTION = 4
PATCH_SIZE = MODEL_INPUT_SIZE[0] // SPATIAL_RESOLUTION

POSITIVE_TO_NEGATIVE_PAIR_MAP = {
    10: 0, 11: 1, 12: 2, 13: 3, 14: 4,
    15: 5, 16: 6, 8: 1, 9: 4,
    17: 7
}

def get_fine_grained_label_from_filename(filename, label_map):
    fn_lower = filename.lower()
    match = re.search(r'_label_(\w+)_view(\d+)', fn_lower)
    if match:
        disease_lower = match.group(1)
        view_num = match.group(2)
        disease_upper = 'normal' if disease_lower in ['normal', 'noraml'] else disease_lower.upper()
        return label_map.get(f"{disease_upper}_view{view_num}", -1)

def parse_doctor_annotations(json_path, frame_limit=16):
    if not os.path.exists(json_path): return None
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
    annotations = {}
    bounding_box_model = data.get("Models", {}).get("BoundingBoxLabelModel", [])
    if not bounding_box_model: return annotations
    for item in bounding_box_model:
        frame_idx = item.get("FrameCount")
        if frame_idx is not None and frame_idx < frame_limit:
            p1, p2 = item.get("p1"), item.get("p2")
            if p1 and p2:
                annotations[frame_idx] = [min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1])]
    return annotations

def transform_gt_box_to_model_space(gt_box, roi_bbox, model_input_size=(224, 224)):
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bbox
    gt_x1_orig, gt_y1_orig, gt_x2_orig, gt_y2_orig = gt_box
    roi_w, roi_h = roi_x2 - roi_x1, roi_y2 - roi_y1
    if roi_w <= 0 or roi_h <= 0: return None
    gt_x1_rel, gt_y1_rel = gt_x1_orig - roi_x1, gt_y1_orig - roi_y1
    scale_x, scale_y = model_input_size[0] / roi_w, model_input_size[1] / roi_h
    x1_model = max(0, gt_x1_rel * scale_x)
    y1_model = max(0, gt_y1_rel * scale_y)
    x2_model = min(model_input_size[0], (gt_x2_orig - roi_x1) * scale_x)
    y2_model = min(model_input_size[1], (gt_y2_orig - roi_y1) * scale_y)
    if x1_model >= x2_model or y1_model >= y2_model: return None
    return [int(x1_model), int(y1_model), int(x2_model), int(y2_model)]

def calculate_iou(boxA, boxB):
    if boxA is None or boxB is None: return 0.0
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0: return 0.0
    return interArea / unionArea

def scale_gt_box_to_patch_size(gt_box, patch_size):
    if gt_box is None: return None
    x1, y1, x2, y2 = gt_box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    half_size = patch_size / 2
    return [
        int(center_x - half_size), int(center_y - half_size),
        int(center_x + half_size), int(center_y + half_size)
    ]

def get_center_and_draw_box(center_point, box_size, image_size=(224, 224)):
    cx, cy = center_point
    half_size = box_size // 2
    x1 = max(0, cx - half_size)
    y1 = max(0, cy - half_size)
    x2 = min(image_size[0], cx + half_size)
    y2 = min(image_size[1], cy + half_size)
    return [int(x1), int(y1), int(x2), int(y2)]

def transform_box_from_model_to_original_space(box_model, roi_bbox):
    if box_model is None or roi_bbox is None: return None
    mx1, my1, mx2, my2 = box_model
    rx1, ry1, rx2, ry2 = roi_bbox
    roi_w, roi_h = rx2 - rx1, ry2 - ry1
    if roi_w <= 0 or roi_h <= 0: return None
    model_w, model_h = (224, 224)
    scale_x_inv, scale_y_inv = roi_w / model_w, roi_h / model_h
    cx1, cy1 = mx1 * scale_x_inv, my1 * scale_y_inv
    cx2, cy2 = mx2 * scale_x_inv, my2 * scale_y_inv
    final_x1, final_y1, final_x2, final_y2 = cx1 + rx1, cy1 + ry1, cx2 + rx1, cy2 + ry1
    return [int(final_x1), int(final_y1), int(final_x2), int(final_y2)]

def create_ensemble_visualization(output_path, background_image, individual_heatmaps, fused_heatmap, pred_box, gt_box, iou, full_video_filename, pred_label, pred_key_frame):
    model_names = sorted(individual_heatmaps.keys())
    num_models = len(model_names)
    fig, axes = plt.subplots(2, num_models, figsize=(num_models * 6, 12), dpi=120)
    title = (f"Grad-CAM++ Evaluation purely based on 5x5 Model for: {full_video_filename}\n"
             f"Predicted Class: {pred_label} | Predicted Key Frame: {pred_key_frame} | Final IoU: {iou:.4f}")
    fig.suptitle(title, fontsize=16)

    for i, name in enumerate(model_names):
        ax = axes[0, i]
        heatmap = individual_heatmaps[name]
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(background_image, 0.6, heatmap_colored, 0.4, 0)
        ax.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Grad-CAM++ Heatmap: {name}', fontsize=14)
        ax.axis('off')

    ax_fused = axes[1, 0]
    im_fused = ax_fused.imshow(fused_heatmap, cmap='viridis')
    ax_fused.set_title('Used Heatmap (5x5 only)', fontsize=14)
    ax_fused.axis('off')
    fig.colorbar(im_fused, ax=ax_fused, fraction=0.046, pad=0.04)

    ax_compare = axes[1, 1]
    ax_compare.imshow(cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB))

    if pred_box:
        p_rect = patches.Rectangle((pred_box[0], pred_box[1]), pred_box[2]-pred_box[0], pred_box[3]-pred_box[1], linewidth=3, edgecolor='lime', facecolor='none')
        ax_compare.add_patch(p_rect)
    if gt_box:
        g_rect = patches.Rectangle((gt_box[0], gt_box[1]), gt_box[2]-gt_box[0], gt_box[3]-gt_box[1], linewidth=3, edgecolor='red', facecolor='none')
        ax_compare.add_patch(g_rect)
    ax_compare.set_title('BBox Comparison', fontsize=14)
    ax_compare.axis('off')
    for i in range(2, num_models): axes[1, i].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path)
    plt.close(fig)

def create_original_space_visualization(output_path, original_frame, pred_box_orig, gt_box_orig, iou, full_video_filename, pred_key_frame):
    frame_to_draw = original_frame.copy()
    if pred_box_orig:
        cv2.rectangle(frame_to_draw, (pred_box_orig[0], pred_box_orig[1]), (pred_box_orig[2], pred_box_orig[3]), (0, 255, 0), 2)
        cv2.putText(frame_to_draw, "Pred", (pred_box_orig[0], pred_box_orig[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    if gt_box_orig:
        cv2.rectangle(frame_to_draw, (gt_box_orig[0], gt_box_orig[1]), (gt_box_orig[2], gt_box_orig[3]), (0, 0, 255), 2)
        cv2.putText(frame_to_draw, "GT", (gt_box_orig[0], gt_box_orig[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    fig, ax = plt.subplots(figsize=(12, 12 * frame_to_draw.shape[0] / frame_to_draw.shape[1]))
    ax.imshow(cv2.cvtColor(frame_to_draw, cv2.COLOR_BGR2RGB))
    title = (f"Original Space Comparison (Grad-CAM++ 5x5): {full_video_filename}\n"
             f"Predicted Key Frame: {pred_key_frame} | IoU (model space): {iou:.3f}")
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# Grad-CAM++ implementation
class GradCAMpp:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output[0]

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_cam(self, model_output, class_idx):
        scores = model_output["predicted_scores"]
        positive_score = scores[0, class_idx]
        negative_scores = scores[0, :8]
        target_score = positive_score - negative_scores.mean()

        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to get gradients or activations.")

        grads = self.gradients
        activations = self.activations
        grads_pos = F.relu(grads)
        grad_power_2 = grads_pos.pow(2)
        grad_power_3 = grads_pos.pow(3)

        alpha_num = grad_power_2 * activations
        numerator = torch.sum(alpha_num, dim=1, keepdim=True)

        alpha_denom = grad_power_3 * activations
        denominator = torch.sum(alpha_denom, dim=1, keepdim=True)
        denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))

        alpha_k_c = numerator / denominator
        weights = torch.sum(F.relu(alpha_k_c * grads_pos), dim=1, keepdim=True)
        cam = torch.einsum('bsc,bcs->bs', activations, weights.transpose(1, 2))
        return F.relu(cam)


if __name__ == '__main__':
    test_transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                     std=[0.26862954, 0.26130258, 0.27577711])
    ])

    torch.cuda.set_device(cuda_device)

    MODEL_CONFIGS = {
        '5x5': {
            'resume_path': './best_valid_acc/epoch_92.95020331726518.pth',
            'baseline_path': './final_baseline_features_5x5_512dim_1101.pt',
            'spatial_resolution': 5
        },
    }

    FINAL_BOX_SIZE = 56

    print("--- Loading all models ---")
    models = {}
    baselines = {}
    grad_cams = {}

    for name, config in MODEL_CONFIGS.items():
        print(f"  -> Loading model: {name}")
        model = ResnetTransformerDualTokensTemporalSpatialDecouplesizeGradCam(
            num_classes=num_classes,
            d_model_cnn=d_model_cnn,
            num_layers=num_layers,
            dropout=dropout,
            spatial_resolution=config['spatial_resolution'])

        resume_path = config['resume_path']
        if resume_path != '':
            resume_model_dict = torch.load(resume_path, map_location='cpu')
            resume_model_filter_dict = {k: v for k, v in resume_model_dict.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
            model.load_state_dict(resume_model_filter_dict, strict=False)
            model.to(cuda_device).eval()
            models[name] = model

        if os.path.exists(config['baseline_path']):
            baselines[name] = torch.load(config['baseline_path'], map_location='cpu').to(cuda_device)
        else:
            raise FileNotFoundError(f"Baseline file not found for model '{name}': {config['baseline_path']}")

        grad_cams[name] = GradCAMpp(model=model, target_layer=model.target_spatial_transformer)

    print("--- All models loaded ---\n")

    # Evaluation statistics
    eval_stats = {
        subtype: {
            "temporal_total": 0, "temporal_hits": 0,
            "iou_conditional": {thresh: {"total": 0, "hits": 0} for thresh in IOU_THRESHOLDS}
        } for subtype in DISEASE_SUBTYPES
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for disease_type in DISEASE_SUBTYPES:
        subtype_dir = os.path.join(DATASET_ROOT_DIR, disease_type)
        if not os.path.isdir(subtype_dir): continue

        print(f"\n--- Processing subtype: {disease_type} ---")
        case_folders = [d for d in os.listdir(subtype_dir) if os.path.isdir(os.path.join(subtype_dir, d))]

        for case_name in tqdm(case_folders, desc=f"Evaluating {disease_type}", unit="case"):
            case_path = os.path.join(subtype_dir, case_name)
            video_files = [f for f in os.listdir(case_path) if f.lower().endswith(('.mp4', '.avi'))]
            video_path = os.path.join(case_path, video_files[0])
            json_path = os.path.join(case_path, "label.json")

            if not (os.path.exists(video_path) and os.path.exists(json_path)): continue

            try:
                true_label_idx = get_fine_grained_label_from_filename(os.path.basename(video_path), POSITIVE_TO_NEGATIVE_MAP)
                if true_label_idx == -1 or true_label_idx <= 7: continue

                original_frames = video_loader(video_path)
                if original_frames is None or len(original_frames) < 1: continue

                indices = np.arange(len(original_frames))
                if len(original_frames) < clip_size:
                    indices = np.tile(indices, (clip_size + len(original_frames) - 1) // len(original_frames))
                processed_16_frames = original_frames[indices[:clip_size]]

                roi_bbox = find_doppler_roi_from_video(processed_16_frames, ROI_AREA_THRESHOLD, ROI_PADDING)
                if roi_bbox is None: continue

                frames_for_model_np = np.array([
                    cv2.resize(frame[roi_bbox[1]:roi_bbox[3], roi_bbox[0]:roi_bbox[2]], MODEL_INPUT_SIZE)
                    for frame in processed_16_frames
                ])

                video_tensor_uint8 = torch.from_numpy(frames_for_model_np.copy()).permute(0, 3, 1, 2)
                input_tensor = test_transform(video_tensor_uint8).unsqueeze(0).to(cuda_device)

                doppler_masks_by_size = {}
                for name, model_config in MODEL_CONFIGS.items():
                    res = model_config['spatial_resolution']
                    mask_np = create_string_token_mask_size(
                        frames_for_model_np,
                        spatial_resolution=res,
                        vote_threshold=1
                    )
                    doppler_masks_by_size[name] = torch.from_numpy(mask_np).unsqueeze(0).to(cuda_device)

                # =====================================================================
                # Step 1: Get model predictions and temporal localization
                # =====================================================================
                all_logits_dict = {}
                key_frame_dict = {}

                for name, model in models.items():
                    with torch.no_grad():
                        outputs = model(input_tensor, None, doppler_masks_by_size[name])
                        all_logits_dict[name] = outputs["predicted_scores"]

                        pred_label_idx_single = torch.argmax(outputs["predicted_scores"], dim=1).item()
                        if pred_label_idx_single <= 7: continue

                        attention_vector = outputs["all_layer_attention_scores"][-1][0, pred_label_idx_single, :]
                        key_patch_idx = torch.argmax(attention_vector).item()

                        normal_id = POSITIVE_TO_NEGATIVE_PAIR_MAP.get(true_label_idx)
                        if normal_id is None: continue

                        key_patch_raw_string = outputs["raw_strings_for_loss"][0, key_patch_idx, :, :]
                        key_patch_baseline = baselines[name][normal_id, key_patch_idx, :]
                        residuals = torch.linalg.norm(key_patch_raw_string - key_patch_baseline.unsqueeze(0), dim=-1)
                        key_frame_dict[name] = torch.argmax(residuals).item()

                # Load ground truth annotations
                doctor_annotations = parse_doctor_annotations(json_path)
                if not doctor_annotations: continue

                eval_stats[disease_type]["temporal_total"] += 1

                gt_key_frames = list(doctor_annotations.keys())
                is_temporal_hit = False
                final_pred_key_frame = -1
                final_pred_label_idx = 0

                if "5x5" in key_frame_dict:
                    final_pred_key_frame = key_frame_dict["5x5"]
                    final_pred_label_idx = torch.argmax(all_logits_dict["5x5"], dim=1).item()

                    if final_pred_key_frame in gt_key_frames:
                        is_temporal_hit = True
                        eval_stats[disease_type]["temporal_hits"] += 1

                # =====================================================================
                # Step 2: Generate spatial heatmap and calculate IoU
                # =====================================================================
                iou = 0.0
                pred_box_ensemble = None
                gt_box_scaled = None

                if is_temporal_hit:
                    gt_box_orig = doctor_annotations[final_pred_key_frame]
                    gt_box_model_space = transform_gt_box_to_model_space(gt_box_orig, roi_bbox)
                    if gt_box_model_space is not None:
                        gt_box_scaled = scale_gt_box_to_patch_size(gt_box_model_space, patch_size=FINAL_BOX_SIZE)

                    if gt_box_scaled is not None and final_pred_label_idx > 7:
                        individual_heatmaps_for_vis = {}

                        for name, model in models.items():
                            grad_cam_instance = grad_cams[name]

                            current_input = input_tensor.detach().clone().requires_grad_(True)
                            outputs = model(current_input, None, doppler_masks_by_size[name])

                            cam_raw = grad_cam_instance.generate_cam(
                                model_output=outputs,
                                class_idx=final_pred_label_idx,
                            )

                            num_patch_tokens = model.num_patch_tokens
                            patch_cam = cam_raw[:, -num_patch_tokens:]
                            static_cam = patch_cam.mean(dim=0)

                            if static_cam.max() > 0:
                                static_cam = (static_cam - static_cam.min()) / (static_cam.max() - static_cam.min())

                            cam_heatmap = static_cam.detach().cpu().numpy().reshape(model.spatial_size, model.spatial_size)
                            heatmap_224 = cv2.resize(cam_heatmap, MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)

                            individual_heatmaps_for_vis[name] = heatmap_224

                        fused_heatmap = individual_heatmaps_for_vis.get("5x5", np.zeros(MODEL_INPUT_SIZE, dtype=np.float32))

                        if np.sum(fused_heatmap) > 0:
                            peak_y, peak_x = np.unravel_index(np.argmax(fused_heatmap), fused_heatmap.shape)
                            pred_box_ensemble = get_center_and_draw_box((peak_x, peak_y), FINAL_BOX_SIZE)
                            iou = calculate_iou(pred_box_ensemble, gt_box_scaled)

                        # Save visualization
                        full_video_filename = os.path.basename(video_path)
                        case_output_dir = os.path.join(OUTPUT_DIR, disease_type, case_name)
                        os.makedirs(case_output_dir, exist_ok=True)

                        create_ensemble_visualization(
                            output_path=os.path.join(case_output_dir, "ensemble_report.png"),
                            background_image=frames_for_model_np[final_pred_key_frame],
                            individual_heatmaps=individual_heatmaps_for_vis,
                            fused_heatmap=fused_heatmap,
                            pred_box=pred_box_ensemble,
                            gt_box=gt_box_scaled,
                            iou=iou,
                            full_video_filename=full_video_filename,
                            pred_label=final_pred_label_idx,
                            pred_key_frame=final_pred_key_frame
                        )

                        pred_box_in_original = transform_box_from_model_to_original_space(pred_box_ensemble, roi_bbox)
                        gt_box_in_original_scaled = transform_box_from_model_to_original_space(gt_box_scaled, roi_bbox)

                        create_original_space_visualization(
                            output_path=os.path.join(case_output_dir, "original_space_comparison.png"),
                            original_frame=original_frames[final_pred_key_frame],
                            pred_box_orig=pred_box_in_original,
                            gt_box_orig=gt_box_in_original_scaled,
                            iou=iou,
                            full_video_filename=full_video_filename,
                            pred_key_frame=final_pred_key_frame
                        )

                    # Update IoU statistics
                    for thresh in IOU_THRESHOLDS:
                        stats = eval_stats[disease_type]['iou_conditional'][thresh]
                        stats["total"] += 1
                        if iou > thresh:
                            stats["hits"] += 1

            except Exception as e:
                print(f"\nError processing case {case_name}: {e}")
                import traceback
                traceback.print_exc()

    # =====================================================================
    # Step 3: Generate and save JSON report
    # =====================================================================
    final_json_report = {}

    # Per-subtype results
    for disease_type in DISEASE_SUBTYPES:
        stats = eval_stats[disease_type]
        t_total = stats["temporal_total"]
        t_hits = stats["temporal_hits"]
        t_acc = (t_hits / t_total * 100) if t_total > 0 else 0.0

        subtype_dict = {
            "temporal_accuracy": {
                "total_samples": t_total,
                "hits": t_hits,
                "accuracy_percent": round(t_acc, 2)
            },
            "spatial_iou_conditional": {}
        }

        for thresh in IOU_THRESHOLDS:
            iou_total = stats['iou_conditional'][thresh]["total"]
            iou_hits = stats['iou_conditional'][thresh]["hits"]
            iou_acc = (iou_hits / iou_total * 100) if iou_total > 0 else 0.0

            subtype_dict["spatial_iou_conditional"][f"IoU_gt_{thresh:.2f}"] = {
                "total_eval_samples": iou_total,
                "hits": iou_hits,
                "accuracy_percent": round(iou_acc, 2)
            }

        final_json_report[disease_type] = subtype_dict

    # Overall average results
    overall_t_total = sum(s["temporal_total"] for s in eval_stats.values())
    overall_t_hits = sum(s["temporal_hits"] for s in eval_stats.values())
    overall_t_acc = (overall_t_hits / overall_t_total * 100) if overall_t_total > 0 else 0.0

    average_dict = {
        "temporal_accuracy": {
            "total_samples": overall_t_total,
            "hits": overall_t_hits,
            "accuracy_percent": round(overall_t_acc, 2)
        },
        "spatial_iou_conditional": {}
    }

    for thresh in IOU_THRESHOLDS:
        o_iou_total = sum(s['iou_conditional'][thresh]["total"] for s in eval_stats.values())
        o_iou_hits = sum(s['iou_conditional'][thresh]["hits"] for s in eval_stats.values())
        o_iou_acc = (o_iou_hits / o_iou_total * 100) if o_iou_total > 0 else 0.0

        average_dict["spatial_iou_conditional"][f"IoU_gt_{thresh:.2f}"] = {
            "total_eval_samples": o_iou_total,
            "hits": o_iou_hits,
            "accuracy_percent": round(o_iou_acc, 2)
        }

    final_json_report["Average"] = average_dict

    # Save final JSON
    report_json_path = os.path.join(OUTPUT_DIR, "final_evaluation_metrics_GradCAMpp.json")
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json_report, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 Evaluation completed! All metrics are properly calculated.")
    print(f"📁 Detailed results saved to JSON (ready for Excel):\n -> {report_json_path}")