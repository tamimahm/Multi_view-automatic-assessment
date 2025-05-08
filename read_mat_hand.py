import os
import scipy.io
import pickle
import glob
from tqdm import tqdm
import numpy as np
import datetime

# Configuration - modify these values directly
HAND_DIR = "D:/ARAT_2D_joints/all_ARAT_hand"  # Directory containing hand keypoint MAT files
OUTPUT_DIR = "D:/pickle_files_hand"            # Directory to save pickle files
GROUP_SIZE = 10                                # Number of patients to include in each pickle file

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

def determine_view_type(folder_path, mat_file):
    """
    Determine if the mat file is for ipsilateral (ipsi) or top view
    """
    # Check if it's a top view
    if 'top' in mat_file.lower():
        return 'top'
    
    # Check if it's ipsilateral
    if 'right_Impaired' in folder_path and 'right' in mat_file.lower():
        return 'ipsi'
    elif 'left_Impaired' in folder_path and 'left' in mat_file.lower():
        return 'ipsi'
    
    # Not ipsilateral or top view
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

def process_hand_mat_file(mat_file):
    """
    Process a MAT file containing hand keypoints
    Returns keypoints in the shape (num_frames, num_keypoints, 2)
    """
    try:
        # Load MAT file
        mat_data = scipy.io.loadmat(mat_file)
        
        # Extract keypoints
        # The MAT file structure may vary, so let's try a few common structures
        if 'handData' in mat_data:
            # First try: handData structure
            hand_keypoints = mat_data['handData']
        elif 'landmarks' in mat_data:
            # Second try: landmarks structure
            hand_keypoints = mat_data['landmarks']
        else:
            # Get the first non-standard key that might contain the data
            # (excluding keys that start with '__')
            potential_keys = [key for key in mat_data.keys() if not key.startswith('__')]
            if potential_keys:
                hand_keypoints = mat_data[potential_keys[0]]
            else:
                # No suitable data found
                return None
        
        # Check if we have valid keypoints
        if hand_keypoints.size == 0:
            return None
        
        # Convert to the right format if needed
        # Assuming the format is either (frames, hands, keypoints, 2) or (frames, keypoints, 2)
        if len(hand_keypoints.shape) == 4:
            # Format is (frames, hands, keypoints, 2)
            # Extract only keypoints (x, y) coordinates
            num_frames = hand_keypoints.shape[0]
            num_hands = hand_keypoints.shape[1]
            num_keypoints = hand_keypoints.shape[2]
            
            # Reshape to (num_frames, num_keypoints*num_hands, 2)
            reshaped_keypoints = np.zeros((num_frames, num_keypoints*num_hands, 2))
            
            # For each frame, concatenate keypoints from both hands
            for frame in range(num_frames):
                hand_data = []
                for hand in range(num_hands):
                    if hand < hand_keypoints.shape[1]:
                        hand_data.append(hand_keypoints[frame, hand])
                
                if hand_data:
                    # Concatenate hand data
                    frame_data = np.concatenate(hand_data, axis=0)
                    if frame_data.shape[0] > 0:
                        # Make sure the data fits in the reshaped array
                        max_keypoints = min(frame_data.shape[0], reshaped_keypoints.shape[1])
                        reshaped_keypoints[frame, :max_keypoints, :] = frame_data[:max_keypoints]
            
            return reshaped_keypoints
        
        elif len(hand_keypoints.shape) == 3:
            # Format is (frames, keypoints, 2)
            return hand_keypoints
        
        else:
            # Unexpected format
            return None
        
    except Exception as e:
        # Log error and return None
        error_log.append((None, None, "mat_file_error", f"{mat_file}: {str(e)}"))
        return None

def process_hand_keypoints():
    """
    Process hand keypoint MAT files and organize them by patient ID into pickle files
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all patient directories
    patient_dirs = [d for d in glob.glob(os.path.join(HAND_DIR, "ARAT_*")) if os.path.isdir(d)]
    print(f"Found {len(patient_dirs)} patient directories")
    
    # Dictionary to store all patient data
    all_patient_data = {}
    
    # Track errors
    no_keypoints_data = set()  # Will store (patient_id, activity_id, view_type) tuples
    file_errors = set()        # Will store (patient_id, activity_id, view_type) tuples
    
    for patient_dir in tqdm(patient_dirs, desc="Processing patient directories"):
        # Extract patient ID from directory name
        patient_id = extract_patient_id(os.path.basename(patient_dir))
        if patient_id is None:
            print(f"Skipping invalid patient directory: {patient_dir}")
            error_log.append((None, None, "invalid_patient_dir", patient_dir))
            continue
        
        # Skip if "Unimpaired" in the directory name
        if "Unimpaired" in patient_dir:
            continue
        
        # Initialize patient data structure
        if patient_id not in all_patient_data:
            all_patient_data[patient_id] = {
                'top': {},   # Will store activity_id -> keypoints arrays
                'ipsi': {}   # Will store activity_id -> keypoints arrays
            }
        
        # Get all activity directories
        activity_dirs = [d for d in glob.glob(os.path.join(patient_dir, "*activity*")) if os.path.isdir(d)]
        
        for activity_dir in activity_dirs:
            # Extract activity ID from directory name
            activity_id = extract_activity_id(os.path.basename(activity_dir))
            if activity_id is None:
                error_log.append((patient_id, None, "missing_activity_id", activity_dir))
                continue
            
            # Get all MAT files in the activity directory
            keypoints_dir = os.path.join(activity_dir, "keypoints")
            if not os.path.isdir(keypoints_dir):
                keypoints_dir = activity_dir  # Try without "keypoints" subdirectory
            
            mat_files = glob.glob(os.path.join(keypoints_dir, "*.mat"))
            
            # Process each MAT file
            for mat_file in mat_files:
                # Determine view type
                view_type = determine_view_type(activity_dir, os.path.basename(mat_file))
                if view_type is None:
                    continue  # Skip if not ipsi or top view
                
                # Process MAT file to get keypoints
                hand_keypoints = process_hand_mat_file(mat_file)
                
                if hand_keypoints is None or hand_keypoints.size == 0:
                    # No valid keypoints found
                    no_keypoints_data.add((patient_id, activity_id, view_type))
                    error_log.append((patient_id, activity_id, "no_hand_keypoints", 
                                     f"View: {view_type}, File: {os.path.basename(mat_file)}"))
                    continue
                
                # Check if we have enough frames
                if hand_keypoints.shape[0] < 5:  # Arbitrary threshold, adjust as needed
                    error_log.append((patient_id, activity_id, "too_few_frames", 
                                     f"View: {view_type}, Frames: {hand_keypoints.shape[0]}"))
                
                # Convert to the desired format (num_keypoints, 2, num_frames)
                # Current shape is (num_frames, num_keypoints, 2)
                hand_keypoints = np.transpose(hand_keypoints, (1, 2, 0))
                
                # Store the keypoints in the patient data
                all_patient_data[patient_id][view_type][activity_id] = hand_keypoints
    
    # Save error log
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    error_log_file = os.path.join(OUTPUT_DIR, f"hand_error_log_{timestamp}.txt")
    
    with open(error_log_file, 'w') as f:
        f.write(f"Hand Keypoints Error Log - {timestamp}\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary of missing keypoints data
        f.write("Patients/Activities with No Hand Keypoints Data:\n")
        f.write("-" * 50 + "\n")
        for patient_id, activity_id, view_type in sorted(no_keypoints_data):
            f.write(f"Patient {patient_id}, Activity {activity_id}, View {view_type}\n")
        
        f.write("\n")
        
        # Summary of file errors
        f.write("Patients/Activities with File Errors:\n")
        f.write("-" * 50 + "\n")
        for patient_id, activity_id, view_type in sorted(file_errors):
            f.write(f"Patient {patient_id}, Activity {activity_id}, View {view_type}\n")
        
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
            output_file = os.path.join(OUTPUT_DIR, f"hand_patients_91_to_110.pkl")
        else:
            # Regular group
            start_id = group[0]
            end_id = group[-1]
            output_file = os.path.join(OUTPUT_DIR, f"hand_patients_{start_id}_to_{end_id}.pkl")
        
        # Collect data for this group
        group_data = {patient_id: all_patient_data[patient_id] for patient_id in group}
        
        # Save to pickle file
        with open(output_file, 'wb') as f:
            pickle.dump(group_data, f)
        
        print(f"Saved {len(group)} patients to {output_file}")
    
    # Print summary
    print(f"Successfully created {len(patient_groups)} pickle files in {OUTPUT_DIR}")
    print(f"Found {len(no_keypoints_data)} patient/activity/view combinations with no hand keypoints")
    print(f"Found {len(file_errors)} patient/activity/view combinations with file errors")
    print(f"Total of {len(error_log)} detailed errors logged")

if __name__ == "__main__":
    process_hand_keypoints()