import os
import pandas as pd
from pathlib import Path

# Directory containing patient folders
base_dir = r"D:\all_ARAT_videos"

# Output CSV file
output_csv = "D:\Github\Multi_view-automatic-assessment\camera_assignments.csv"

# Lists to store data
data = []

# Camera assignments based on impaired side
# cam4 (left), cam1 (right)
ipsilateral_map = {
    "left_Impaired": "cam4",
    "right_Impaired": "cam1"
}
contralateral_map = {
    "left_Impaired": "cam1",
    "right_Impaired": "cam4"
}

# Iterate through patient folders
for patient_folder in os.listdir(base_dir):
    patient_path = os.path.join(base_dir, patient_folder)
    
    # Check if it's a directory and starts with ARAT_
    if os.path.isdir(patient_path) and patient_folder.startswith("ARAT_"):
        patient_id = patient_folder
        
        # Get list of video files in the patient folder
        for video_file in os.listdir(patient_path):
            # Check if file is a video (based on common extensions)
            if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Check for impaired side in filename
                if "left_Impaired" in video_file:
                    impaired_side = "left_Impaired"
                elif "right_Impaired" in video_file:
                    impaired_side = "right_Impaired"
                else:
                    continue  # Skip if no impaired side is found
                
                # Assign cameras based on impaired side
                ipsilateral_camera = ipsilateral_map.get(impaired_side, "")
                contralateral_camera = contralateral_map.get(impaired_side, "")
                
                # Add to data list
                data.append({
                    "patient_id": patient_id.split('_')[1],
                    "ipsilateral_camera_id": ipsilateral_camera,
                    "contralateral_camera_id": contralateral_camera
                })
                
                # Process only one valid video per patient
                break

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv(output_csv, index=False)
print(f"CSV file '{output_csv}' has been created with {len(df)} entries.")