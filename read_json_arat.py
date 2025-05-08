import os
import json
import pickle
import glob
from tqdm import tqdm
import numpy as np
import datetime

# Configuration - modify these values directly
JSON_DIR = "D:/all_ARAT_openpose"   # Directory containing JSON files
OUTPUT_DIR = "D:/pickle_files"       # Directory to save pickle files
GROUP_SIZE = 10                      # Number of patients to include in each pickle file

# Error tracking
error_log = []  # Will store errors as (patient_id, activity_id, error_type, details)

def extract_patient_id(folder_name):
    """Extract numeric patient ID from folder name like 'ARAT_001'"""
    try:
        # Split by underscore and get the numeric part
        parts = folder_name.split('_')
        if len(parts) >= 2:
            # Convert to integer (remove leading zeros)
            return int(parts[1])
        return None
    except (ValueError, IndexError):
        return None

def determine_view_type(folder_path):
    """
    Determine if the folder is for ipsilateral or top view
    """
    # Check if 'cam3' is in the path
    if 'cam3' in folder_path:
        return 'top'
    # Check if it's ipsilateral (either right_Impaired_cam1 or left_Impaired_cam4)
    elif ('right_Impaired_cam1' in folder_path) or ('left_Impaired_cam4' in folder_path):
        return 'ipsi'
    # Not a valid view type
    return None

def extract_activity_id(folder_name):
    """Extract activity ID from folder name like 'ARAT_01_right_Impaired_cam1_activity2'"""
    try:
        # Look for "activity" followed by a number at the end
        if 'activity' in folder_name:
            # Extract the number after "activity"
            parts = folder_name.split('activity')
            if len(parts) > 1:
                # The activity ID might be followed by other text or underscores
                activity_id_part = parts[1].split('_')[0]
                return int(activity_id_part)
        return None
    except (ValueError, IndexError):
        return None

def extract_frame_number(filename):
    """Extract frame number from filename like 'ARAT_01_right_Impaired_cam1_activity8_000000000001_keypoints.json'"""
    try:
        # Split by underscore and find the part that starts with zeros
        parts = filename.split('_')
        return int(parts[6])
    except (ValueError, IndexError):
        # Log error but don't add to main error log as this is too frequent
        # print(f"Could not extract frame number from {filename}")
        return None

def process_json_files():
    """
    Process JSON files and organize them by patient ID into pickle files
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all patient directories (ARAT_XXX format)
    patient_dirs = [d for d in glob.glob(os.path.join(JSON_DIR, "ARAT_*")) if os.path.isdir(d)]
    print(f"Found {len(patient_dirs)} patient directories")
    
    # Dictionary to store all patient data
    all_patient_data = {}
    
    # Track patient and activity IDs with no people data
    no_people_data = set()  # Will store (patient_id, activity_id, view_type) tuples
    file_errors = set()     # Will store (patient_id, activity_id, view_type) tuples
    
    for patient_dir in tqdm(patient_dirs, desc="Processing patient directories"):
        # Extract patient ID from directory name
        patient_id = extract_patient_id(os.path.basename(patient_dir))
        if patient_id is None:
            print(f"Skipping invalid patient directory: {patient_dir}")
            error_log.append((None, None, "invalid_patient_dir", patient_dir))
            continue
        
        # Initialize patient data structure (simplified)
        if patient_id not in all_patient_data:
            all_patient_data[patient_id] = {
                'top': {},   # Will store activity_id -> keypoints arrays
                'ipsi': {}   # Will store activity_id -> keypoints arrays
            }
        
        # Recursively find all JSON files under the patient directory
        json_files = glob.glob(os.path.join(patient_dir, "**", "*.json"), recursive=True)
        
        # Group JSON files by activity ID and view type
        activity_view_files = {}
        
        for json_file in json_files:
            # Skip if 'cam2' in the filename
            if 'cam2' in json_file:
                continue
            if 'Unimpaired' in json_file:
                continue            
            # Determine view type
            view_type = determine_view_type(json_file)
            if view_type is None:
                continue
            
            # Extract activity ID from path
            activity_id = None
            path_parts = json_file.split(os.sep)
            for part in path_parts:
                if 'activity' in part:
                    activity_id = extract_activity_id(part)
                    break
            
            if activity_id is None:
                error_log.append((patient_id, None, "missing_activity_id", json_file))
                continue
            
            # Group by activity ID and view type
            key = (activity_id, view_type)
            if key not in activity_view_files:
                activity_view_files[key] = []
            
            activity_view_files[key].append(json_file)
        
        # Process each activity and view type
        for (activity_id, view_type), files in activity_view_files.items():
            # Dictionary to store frame data temporarily
            frames_data = {}
            
            # First pass to check if any frames have people
            total_frames = 0
            frames_with_people = 0
            empty_keypoints_frames = 0
            
            for json_file in files:
                # Extract frame number from filename
                frame_num = extract_frame_number(os.path.basename(json_file))
                if frame_num is None:
                    continue
                
                total_frames += 1
                
                # Load JSON data
                try:
                    with open(json_file, 'r') as f:
                        keypoint_data = json.load(f)
                    
                    # Check if people data exists
                    if 'people' in keypoint_data and len(keypoint_data['people']) > 0:
                        # Check if pose keypoints exist
                        pose_keypoints = keypoint_data['people'][0].get('pose_keypoints_2d', [])
                        if pose_keypoints:
                            frames_with_people += 1
                        else:
                            empty_keypoints_frames += 1
                except Exception as e:
                    file_errors.add((patient_id, activity_id, view_type, str(e)))
                    error_log.append((patient_id, activity_id, "file_error", f"{json_file}: {str(e)}"))
            
            # Only flag as "no people" if no frames have people
            if total_frames > 0 and frames_with_people == 0:
                no_people_data.add((patient_id, activity_id, view_type, f"no_people_in_all_{total_frames}_frames"))
                # Add to error log
                error_log.append((patient_id, activity_id, "no_people_data", 
                                  f"View: {view_type}, Total frames: {total_frames}, Empty keypoints: {empty_keypoints_frames}"))
            
            # Second pass to extract keypoints
            for json_file in files:
                # Extract frame number from filename
                frame_num = extract_frame_number(os.path.basename(json_file))
                if frame_num is None:
                    continue
                
                # Load JSON data
                try:
                    with open(json_file, 'r') as f:
                        keypoint_data = json.load(f)
                    
                    # Extract the 2D keypoints if available
                    if 'people' in keypoint_data and len(keypoint_data['people']) > 0:
                        # Get pose keypoints (x, y, confidence)
                        pose_keypoints = keypoint_data['people'][0].get('pose_keypoints_2d', [])
                        
                        # Convert to numpy array and reshape
                        if pose_keypoints:
                            pose_keypoints = np.array(pose_keypoints).reshape(-1, 3)
                            # Only store x, y coordinates
                            frames_data[frame_num] = pose_keypoints[:, :2]
                except Exception as e:
                    # Already logged in first pass
                    pass
            
            # If we have frame data, process it into the required format
            if frames_data:
                # Sort frame numbers
                sorted_frames = sorted(frames_data.keys())
                
                # Create a 3D array (25, 2, time_frames)
                # First get the number of keypoints from the first frame
                first_frame = frames_data[sorted_frames[0]]
                num_keypoints = first_frame.shape[0]
                
                # Initialize the array
                keypoints_array = np.zeros((num_keypoints, 2, len(sorted_frames)))
                
                # Fill the array
                for i, frame_num in enumerate(sorted_frames):
                    keypoints = frames_data[frame_num]
                    # Handle frames with different numbers of keypoints
                    if keypoints.shape[0] == num_keypoints:
                        keypoints_array[:, :, i] = keypoints
                    else:
                        error_log.append((patient_id, activity_id, "keypoint_count_mismatch", 
                                         f"Frame {frame_num} has {keypoints.shape[0]} keypoints, expected {num_keypoints}"))
                
                # Store the array in the patient data
                all_patient_data[patient_id][view_type][activity_id] = keypoints_array
    
    # Save error log
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    error_log_file = os.path.join(OUTPUT_DIR, f"error_log_{timestamp}.txt")
    
    with open(error_log_file, 'w') as f:
        f.write(f"Error Log - {timestamp}\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary of no people data
        f.write("Patients/Activities with No People Data:\n")
        f.write("-" * 50 + "\n")
        for patient_id, activity_id, view_type, reason in sorted(no_people_data):
            f.write(f"Patient {patient_id}, Activity {activity_id}, View {view_type}: {reason}\n")
        
        f.write("\n")
        
        # Summary of file errors
        f.write("Patients/Activities with File Errors:\n")
        f.write("-" * 50 + "\n")
        for patient_id, activity_id, view_type, error in sorted(file_errors):
            f.write(f"Patient {patient_id}, Activity {activity_id}, View {view_type}: {error}\n")
        
        f.write("\n")
        
        # Detailed error log
        f.write("Detailed Error Log:\n")
        f.write("-" * 50 + "\n")
        for patient_id, activity_id, error_type, details in error_log:
            f.write(f"Patient {patient_id}, Activity {activity_id}, Error: {error_type}\n")
            f.write(f"Details: {details}\n\n")
            
    print(f"Error log saved to {error_log_file}")
    
    # Group patients by GROUP_SIZE
    patient_ids = sorted(list(all_patient_data.keys()))
    
    # Special handling for patients 91-110
    special_group_patients = [pid for pid in patient_ids if 91 <= pid <= 110]
    normal_patients = [pid for pid in patient_ids if pid not in special_group_patients]
    
    # Group normal patients
    patient_groups = []
    for i in range(0, len(normal_patients), GROUP_SIZE):
        group = normal_patients[i:i+GROUP_SIZE]
        if group:  # Only add non-empty groups
            patient_groups.append(group)
    
    # Add special group
    if special_group_patients:
        patient_groups.append(special_group_patients)
    
    # Save each group to a pickle file
    print("Saving patient data to pickle files...")
    for group in patient_groups:
        if not group:
            continue
            
        if 91 <= group[0] <= 110:
            # This is the special group (91-110)
            output_file = os.path.join(OUTPUT_DIR, f"patients_91_to_110.pkl")
        else:
            # Regular group
            start_id = group[0]
            end_id = group[-1]
            output_file = os.path.join(OUTPUT_DIR, f"patients_{start_id}_to_{end_id}.pkl")
        
        # Collect data for this group
        group_data = {patient_id: all_patient_data[patient_id] for patient_id in group}
        
        # Save to pickle file
        with open(output_file, 'wb') as f:
            pickle.dump(group_data, f)
        
        print(f"Saved {len(group)} patients to {output_file}")
    
    # Print summary
    print(f"Successfully created {len(patient_groups)} pickle files in {OUTPUT_DIR}")
    print(f"Found {len(no_people_data)} patient/activity/view combinations with no people data")
    print(f"Found {len(file_errors)} patient/activity/view combinations with file errors")
    print(f"Total of {len(error_log)} detailed errors logged")

if __name__ == "__main__":
    process_json_files()