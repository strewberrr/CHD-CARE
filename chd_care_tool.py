import os
import urllib.request
import sys
from inference import CHDCareAPI

class CHD_CARE:
    def __init__(self, device='cuda'):
        """
        Initialize the system and automatically load default configurations and weights
        """
        # Fixed scope issue by adding self. prefix
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        config_diag = os.path.join(self.base_dir, 'Config/train_config_mutil_views.yaml')
        config_xai  = os.path.join(self.base_dir, 'Config/test_config_single_view_for_XAI_visual.yaml')
        
        # Weight import options:
        # 1. Manually download and place into the corresponding folder
        # 2. Run the script directly to auto-download from links

        # Local weight save path
        weights_dir = os.path.join(self.base_dir, 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        
        weights_diag = os.path.join(weights_dir, 'best_multiview_model.pth')
        weights_xai  = os.path.join(weights_dir, 'best_singleview_model.pth')
        baseline_xai = os.path.join(weights_dir, 'baseline_features.pt')
        
        # Configure Zenodo direct download links
        self.download_urls = {
            weights_diag: "https://zenodo.org/records/20382397/files/best_multiview_model.pth?download=1",
            weights_xai:  "https://zenodo.org/records/20382397/files/best_singleview_model.pth?download=1",
            baseline_xai: "https://zenodo.org/records/20382397/files/baseline_features.pt?download=1"
        }

        # Critical step: trigger auto-download before initializing API
        self._check_and_download_weights()

        self.api = CHDCareAPI(
            config_path_diag=config_diag,
            config_path_xai=config_xai,
            weights_path_diag=weights_diag,
            weights_path_xai=weights_xai,
            baseline_path_xai=baseline_xai,
            device=device
        )
        
        self.class_map = {0: 'Normal', 1: 'VSD', 2: 'ASD', 3: 'PDA'}

    def _check_and_download_weights(self):
        """Check weight files; auto-download from cloud if missing"""
        for filepath, url in self.download_urls.items():
            if not os.path.exists(filepath):
                filename = os.path.basename(filepath)
                print(f"\n[INFO] Required file not found: {filename}")
                print(f"[INFO] Auto-downloading from Zenodo, please wait...")
                try:
                    self._download_with_progress(url, filepath)
                    print(f"\n[SUCCESS] {filename} downloaded successfully!")
                except Exception as e:
                    print(f"\n[ERROR] Failed to download {filename}. Please manually download and place into weights/ folder as per README.")
                    print(f"Error message: {e}")

    def _download_with_progress(self, url, dest_path):
        """Downloader with console progress bar"""
        def reporthook(blocknum, blocksize, totalsize):
            read_so_far = blocknum * blocksize
            if totalsize > 0:
                percent = read_so_far * 1e2 / totalsize
                s = f"\rDownload Progress: [{percent:5.1f}%] {read_so_far / (1024*1024):.1f}MB / {totalsize / (1024*1024):.1f}MB"
                sys.stdout.write(s)
                if read_so_far >= totalsize:
                    sys.stdout.write("\n")
            else:
                sys.stdout.write(f"\rDownloaded {read_so_far / (1024*1024):.1f}MB")
                
        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)

    def analyze_case(self, case_dir, output_dir='./CHD-CARE/Output_Results', task='both'):
        result_dict = {"Case_Directory": os.path.abspath(case_dir)}
        
        # Extract case name (e.g., 'patient_001')
        case_name = os.path.basename(os.path.normpath(case_dir))
        
        # New directory structure: use case name as dedicated folder
        patient_out_dir = os.path.join(output_dir, case_name)
        os.makedirs(patient_out_dir, exist_ok=True)

        if task in ['diagnose', 'both']:
            pred_idx = self.api.diagnose(case_dir, patient_out_dir)
            diagnosis_str = self.class_map.get(pred_idx, f"Unknown_Class_{pred_idx}")
            result_dict["Diagnosis"] = diagnosis_str
            result_dict["Class_Index"] = pred_idx
            result_dict["Diagnosis_Report_Path"] = os.path.abspath(patient_out_dir)

        if task in ['visualize', 'both']:
            # Save visualization results into XAI_Visuals subfolder
            xai_out_dir = os.path.join(patient_out_dir, 'XAI_Visuals')
            self.api.visualize(case_dir, xai_out_dir)
            result_dict["Visualizations_Saved_To"] = os.path.abspath(xai_out_dir)
            
        return result_dict