from chd_care_tool import CHD_CARE

# Initialize CHD-CARE system
chd_care = CHD_CARE(device='cuda:7')
patient_dir = "./CHD-CARE/Demo_Cases/patient_004"

# Full pipeline: Diagnosis + Visualization
result_all = chd_care.analyze_case(patient_dir, task='both')
print(result_all["Diagnosis"])

# # Diagnosis-only mode
# result_diag = chd_care.analyze_case(patient_dir, task='diagnose')
# print(f"Diagnosis Result: {result_diag['Diagnosis']}")

# # Visualization-only mode
# result_vis = chd_care.analyze_case(patient_dir, task='visualize')
# print(f"Visualization Output Path: {result_vis['Visualizations_Saved_To']}") 