# CHD-CARE: collaborative AI for reliable pediatric congenital heart disease diagnosis across tiered hospitals

This repository contains the official PyTorch implementation for the study: "[CHD-CARE: collaborative AI for reliable pediatric congenital heart disease diagnosis across tiered hospitals]". 

The provided framework, **CHD-CARE (Collaborative AI for Reliable Echocardiography in Congenital Heart Disease)**, encompasses a comprehensive computational pipeline for pediatric CHD analysis from color Doppler echocardiography videos. 

As a collaborative AI framework, it integrates robust data preprocessing, joint pretraining for temporal normal baseline construction, single-view feature extraction, and multi-view diagnostic classification. Beyond standardized classification, CHD-CARE is engineered to assist clinical workflows by providing **four-class predictions, class probabilities, temporal key-frame guidance, and defect-region localization** through the implementation of collaborative Explainable Artificial Intelligence (XAI) methodologies.

## 1. System Requirements

### Hardware Requirements
The training and inference scripts require a standard workstation equipped with a modern NVIDIA GPU. 
- **Recommended GPU:** NVIDIA RTX 3090 / A6000 or equivalent (Minimum 12GB VRAM for single-view; 24GB+ recommended for multi-view processing).
- **RAM:** Minimum 32 GB.

### Software Requirements
The codebase has been developed and rigorously tested on Linux (Ubuntu 20.04/22.04). For full reproducibility, we highly recommend matching our tested environment:
- Python >= 3.9 (Tested on 3.9.23)
- PyTorch >= 2.0.0 (Tested on 2.7.1+cu126)
- CUDA Toolkit >= 11.7 (Tested on 12.6)

### Installation
Clone this repository and configure the required environment:

```bash
git clone [https://github.com/strewberrr/CHD-CARE.git](https://github.com/strewberrr/CHD-CARE.git)
cd CHD-CARE

# 1. Install PyTorch according to your specific CUDA version (e.g., CUDA 12.x)
# Please refer to the official PyTorch website ([https://pytorch.org/](https://pytorch.org/)) for the exact command.
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 2. Install the remaining dependencies
pip install -r requirements.txt
```

## 2. Directory Structure and Module Description

The repository is modularized to separate data curation, architectural definition, training logic, and interpretability analysis, facilitating reproducibility and ablation studies.

```text
CHD-CARE/
├── Baseline_feature/           # Baseline feature extraction and pre-training modules
│   ├── extract_baseline_features.py  
│   └── train_baseline_feature.py     
├── Config/                     # Experimental parameters and configuration files
│   ├── test_config_multi_views.yaml
│   ├── test_config_single_view_for_XAI_visual.yaml
│   ├── train_config_baseline_feature.yaml
│   ├── train_config_mutil_views.yaml
│   └── train_config_single_view.yaml
├── Data_preprocess/            # Data preprocessing pipeline
│   └── make_pickle_cropped_video.py
├── Datasets/                   # Data loading modules
│   ├── dataset_baseline_feature.py
│   ├── dataset_mutil_views.py
│   └── dataset_single_view.py
├── functions/              # Core training/evaluation functions and loss calculations
│   ├── functions_baseline_feature.py
│   ├── functions_multi_views.py
│   └── functions_single_view.py
├── nets/                   # Network architecture definitions
│   ├── model_baseline_feature.py
│   ├── model_multi_views.py
│   └── model_single_view.py
├── Test/                       # Model inference and testing scripts
│   └── test_multi_views.py
├── Train/                      # Model training execution scripts
│   ├── train_multi_views.py
│   └── train_single_view.py
├── XAI_visual/                 # Explainability evaluation and key-frame/heatmap visualization
│   ├── hetmaps_and_location_gradcam.py
│   ├── hetmaps_and_location_gradcampp.py
│   └── hetmaps_and_location_ours.py
└── utils.py                    # Global utility functions
```

### Detailed Module Overview

### 📂 Directory Structure and Detailed Module Overview

* 🏗️ **`Baseline_feature/`** — Contains core scripts for joint pretraining and the construction of temporal normal baseline features.
  * `extract_baseline_features.py`: Executes feature extraction on negative (normal) samples and serializes them as pre-computed baseline tensors.
  * `train_baseline_feature.py`: Main script for model training incorporating the baseline features.

* ⚙️ **`Config/`** — Stores `.yaml` configuration files for various experimental settings, encompassing hyperparameters (e.g., learning rate, batch size), file paths, and network structure configurations.

* 🧹 **`Data_preprocess/`** — Automated data curation pipeline.
  * `make_pickle_cropped_video.py`: Data curation script for raw videos. Responsible for frame extraction, blood-flow Region of Interest (ROI) cropping, and data serialization into `.pkl` format to accelerate data loading.

* 🗄️ **`Datasets/`** — Contains custom dataset classes inheriting from `torch.utils.data.Dataset`, handling data loading, transformation, and batch sampling.
  * `dataset_baseline_feature.py`: Incorporates paired sampling logic (e.g., quadruplet sampling) for contrastive learning.
  * `dataset_mutil_views.py`: Processes multiple echocardiography views from the same case, supporting sequence padding and multi-view data aggregation.
  * `dataset_single_view.py`: Standard single-view video data loading.

* 🧮 **`functions/`** — Encapsulates specific training and validation loop logic, alongside custom loss calculations.
  * `functions_baseline_feature.py`: Implements the core computational logic for the joint pretraining phase, specifically handling structured quadruplet pairing and the mathematical formulation of the contrastive learning loss.
  * `functions_multi_views.py`: Implements processing logic for multi-view feature fusion and joint optimization.
  * `functions_single_view.py`: Executes single-view feature extraction and computes classification loss.

* 🧠 **`nets/`** — Stores PyTorch-based model architecture definitions(The "Brain" of CHD-CARE).
  * `model_baseline_feature.py`: Pre-trained CNN and Temporal Transformer architecture for contrastive learning networks.
  * `model_multi_views.py`: Model implementing multi-view feature aggregation.
  * `model_single_view.py`: Baseline CNN and interleaved Spatiotemporal Transformer network for single-view feature extraction.

* 🚀 **`Train/` & `Test/`** — Entry scripts for the experimental pipeline.
  * `Train/train_multi_views.py` / `train_single_view.py`: Instantiates models, optimizers, and data loaders to initiate the complete training pipeline.
  * `Test/test_multi_views.py`: Loads pre-trained weights to evaluate multi-view model performance on independent test sets, outputting comprehensive evaluation metrics (AUC, F1, Confusion Matrix).

* 🔍 **`XAI_visual/`** — Generates Class Activation Maps (CAM) and spatiotemporal attention heatmaps to interpret model decision bases.
  * `hetmaps_and_location_gradcam.py`: Heatmap generation based on the standard Grad-CAM algorithm.
  * `hetmaps_and_location_gradcampp.py`: Heatmap generation based on the Grad-CAM++ algorithm, demonstrating higher sensitivity to multi-target localization.
  * `hetmaps_and_location_ours.py`: Spatial localization heatmap generation based on the proprietary attention mechanism and interpretation algorithm proposed in this study.

* 🧰 **`utils.py`** — Contains fundamental utility functions called globally across the project, including random seed fixation, learning rate scheduling, checkpoint saving, evaluation metric calculation, denormalization, and confusion matrix updates.

## 3. Quick Start: Demo Inference

To facilitate easy reproduction and clinical evaluation, we provide an end-to-end inference API (`chd_care_tool.py`) and a ready-to-use demonstration script (`run_demo.py`).

### 3.1 Sample Data Preparation

Due to file size limitations on GitHub, the raw pediatric echocardiography videos for demonstration cannot be directly uploaded to this repository. Instead, the comprehensive sample dataset is securely hosted on Zenodo. 

Please download the dataset package from the direct link below, extract the compressed files, and place the patient folders directly into the `Demo_Cases/` directory located at the root of this project:

* [⬇️ Direct Download: Demo_Cases.zip](https://zenodo.org/records/20382397/files/Demo_Cases.zip?download=1)

To fully demonstrate and evaluate the multi-class diagnostic capabilities of the CHD-CARE framework, these sample cases encompass all four clinical categories analyzed in our study:

* **`patient_001`**: Ventricular Septal Defect (**VSD**)
* **`patient_002`**: Atrial Septal Defect (**ASD**)
* **`patient_003`**: Patent Ductus Arteriosus (**PDA**)
* **`patient_004`**: **Normal** (Negative Control Group)

Each patient folder contains the standard color Doppler echocardiography videos required for the analysis. After successful downloading and extraction, your local directory structure should follow this layout:

```text
CHD-CARE/
├── Demo_Cases/
│   ├── patient_001/                  <-- Contains raw VSD videos
│   ├── patient_002/                  <-- Contains raw ASD videos
│   ├── patient_003/                  <-- Contains raw PDA videos
│   └── patient_004/                  <-- Contains raw Normal videos
├── run_demo.py
├── chd_care_tool.py
├── inference.py
└── ...
```

### 3.2 Model Weights & Preparation
The CHD-CARE framework requires specific trained model weights and offline baseline features. To keep this repository lightweight, the `weights/` directory is **not** included by default. We host these large files publicly on [Zenodo](https://zenodo.org/records/20382397) to guarantee long-term availability.

**Option A: Automatic Download (Recommended)**
You do not need to download anything manually. Simply run the `run_demo.py` script. The system will automatically create the `weights/` directory, detect missing files, and securely download them from Zenodo with a progress bar.

**Option B: Manual Download (For Offline Servers)**
If you are deploying this on an offline GPU cluster, please manually download the following files:
* [best_multiview_model.pth](https://zenodo.org/records/20382397/files/best_multiview_model.pth?download=1)
* [best_singleview_model.pth](https://zenodo.org/records/20382397/files/best_singleview_model.pth?download=1)
* [baseline_features.pt](https://zenodo.org/records/20382397/files/baseline_features.pt?download=1)

Once downloaded, manually create a `weights/` folder in the root directory and place the three files inside.

### 3.3 Running the Pipeline
You can run the full diagnostic and XAI visualization pipeline on a specific patient case using the following command:

```bash
# Run both diagnosis and visualization for patient_001
python run_demo.py --case_dir ./Demo_Cases/patient_001 --task both

# Or run specific tasks separately
python run_demo.py --case_dir ./Demo_Cases/patient_001 --task diagnose
python run_demo.py --case_dir ./Demo_Cases/patient_001 --task visualize
```

### 3.4 Expected Output Structure
Upon successful execution, the framework will automatically generate a patient-centric archive in the designated output directory (default is `./Output_Results/` located in the project root). This clean, hierarchical structure ensures seamless integration with electronic medical records (EMR) systems:

```text
CHD-CARE/
└── Output_Results/
    └── patient_001/                              <-- Patient-specific archive
        ├── Diagnosis_patient_001.json            <-- Comprehensive diagnostic report with 4-class probabilities
        └── XAI_Visuals/                          <-- Collaborative Explainable AI evidence
            ├── apical_4_chamber_Heatmap.png      <-- Spatiotemporal attention score distribution overlay
            ├── apical_4_chamber_BBox.png         <-- Defect localization bounding box (generated only for anomalous views)
            └── XAI_KeyFrames_Summary.json        <-- Summary log detailing critical frame indices and negative view identifiers
          