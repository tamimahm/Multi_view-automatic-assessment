import os
import pandas as pd
import pickle
import glob
from tqdm import tqdm
import numpy as np
import datetime

# Configuration - modify these values directly
BASE_OBJECT_DIR = "D:/data_res_trident/alternative"                           # Base directory for object data
TOP_DIR = "D:/data_res_trident/alternative/top_0.85"                          # Directory for top view data
IPSI_DIR = "D:/data_res_trident/alternative/ipsilateral_0.85"                 # Directory for ipsilateral data
CORRECTED_IPSI_DIR = "D:/tamim_deep_learning/ARAT_impairment/Segmentation/missing object files/data_res_trident/alternative/ipsilateral_0.85"  # Directory with corrected ipsi data
OUTPUT_DIR = "D:/pickle_files_object"                                         # Directory to save pickle files
GROUP_SIZE = 10                                                               # Number of patients to include in each pickle file

# Error tracking
error_log = []  # Will store errors as (patient_id, activity_id, error_type, details)

def extract_patient_id(folder_name):
    """Extract numeric patient ID from folder name like 'ARAT_001'"""
    try:
        # For folder names like "ARAT_001"
        parts = folder_name.split('_')
        if len(parts) >= 2:
            # Convert to integer (remove leading zeros)
            return int(parts[1])
        return None
    except (ValueError, IndexError):
        return None

def extract_activity_id(filename):
    """Extract activity ID from filename like 'ARAT_01_right_Impaired_cam1_activity2.csv'"""
    try:
        # For filenames with "activity" followed by a number
        if 'activity' in filename:
            # Extract the part after "activity"
            activity_part = filename.split('activity')[1].split('.')[0].split('_')[0]
            return int(activity_part)
    except (ValueError, IndexError):
        pass
    
    return None

def process_csv_file(csv_path, patient_id=None, activity_id=None):
    """
    Process a CSV file containing object location data
    Returns data in shape (2, num_frames) - [x_coords, y_coords]
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Check if the file contains x and y columns
        if 'x' not in df.columns or 'y' not in df.columns:
            error_log.append((patient_id, activity_id, "missing_columns", f"File {csv_path} doesn't have x and y columns"))
            return None
        
        # Extract x and y coordinates
        x_coords = df['x'].values
        y_coords = df['y'].values
        
        # Check if we have enough data
        if len(x_coords) < 5:  # Arbitrary threshold, adjust as needed
            error_log.append((patient_id, activity_id, "too_few_frames", f"File {csv_path} has only {len(x_coords)} frames"))
        
        # Check for missing data (NaN values)
        if np.isnan(x_coords).any() or np.isnan(y_coords).any():
            # Count NaN values
            nan_count_x = np.isnan(x_coords).sum()
            nan_count_y = np.isnan(y_coords).sum()
            error_log.append((patient_id, activity_id, "missing_values", 
                            f"File {csv_path} has {nan_count_x} NaN x values and {nan_count_y} NaN y values"))
            
            # Replace NaN with zeros
            x_coords = np.nan_to_num(x_coords)
            y_coords = np.nan_to_num(y_coords)
        
        # Return data in shape (2, num_frames)
        return np.array([x_coords, y_coords])
        
    except Exception as e:
        error_log.append((patient_id, activity_id, "csv_file_error", f"{csv_path}: {str(e)}"))
        return None

def process_object_locations():
    """
    Process object location CSV files and organize them by patient ID into pickle files
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Dictionary to store all patient data
    all_patient_data = {}
    
    # Track errors
    no_object_data = set()  # Will store (patient_id, activity_id, view_type) tuples
    file_errors = set()     # Will store (patient_id, activity_id, view_type) tuples
    
    # Process top view data
    print(f"Processing top view object data from {TOP_DIR}...")
    
    if os.path.exists(TOP_DIR):
        # Get all patient directories
        patient_dirs = [d for d in glob.glob(os.path.join(TOP_DIR, "ARAT_*")) if os.path.isdir(d)]
        
        for patient_dir in tqdm(patient_dirs, desc="Processing top view patients"):
            # Extract patient ID
            patient_id = extract_patient_id(os.path.basename(patient_dir))
            if patient_id is None:
                error_log.append((None, None, "invalid_patient_dir", patient_dir))
                continue
            
            # Initialize patient data if not exists
            if patient_id not in all_patient_data:
                all_patient_data[patient_id] = {
                    'top': {},   # Will store activity_id -> object location arrays
                    'ipsi': {}   # Will store activity_id -> object location arrays
                }
            
            # Process all CSV files in this patient directory
            csv_files = glob.glob(os.path.join(patient_dir, "*.csv"))
            
            for csv_file in csv_files:
                # Skip if "Unimpaired" in the file name
                if "Unimpaired" in csv_file:
                    continue
                
                # Extract activity ID from filename
                activity_id = extract_activity_id(os.path.basename(csv_file))
                if activity_id is None:
                    error_log.append((patient_id, None, "missing_activity_id", csv_file))
                    continue
                
                # Process CSV file
                object_data = process_csv_file(csv_file, patient_id, activity_id)
                
                if object_data is None or object_data.size == 0:
                    # No valid object location data
                    no_object_data.add((patient_id, activity_id, 'top'))
                    error_log.append((patient_id, activity_id, "no_object_data", 
                                     f"View: top, File: {os.path.basename(csv_file)}"))
                    continue
                
                # Store object location data
                all_patient_data[patient_id]['top'][activity_id] = object_data
    else:
        print(f"Warning: Top view directory {TOP_DIR} not found")
    
    # Process ipsilateral view data
    # First, collect all patient IDs and activity IDs from both directories
    print("Collecting all patient and activity IDs from ipsilateral data...")
    
    # Dictionary to track all available ipsilateral data
    # Format: {(patient_id, activity_id): {'corrected': file_path, 'original': file_path}}
    ipsi_data_map = {}
    
    # Check for corrected ipsilateral data first
    if os.path.exists(CORRECTED_IPSI_DIR):
        print(f"Checking corrected ipsilateral directory: {CORRECTED_IPSI_DIR}")
        # Get all patient directories
        patient_dirs = [d for d in glob.glob(os.path.join(CORRECTED_IPSI_DIR, "ARAT_*")) if os.path.isdir(d)]
        
        for patient_dir in tqdm(patient_dirs, desc="Scanning corrected ipsi patient directories"):
            # Extract patient ID
            patient_id = int(os.path.basename(patient_dir).split('_')[1])
            if patient_id is None:
                error_log.append((None, None, "invalid_patient_dir", patient_dir))
                continue
            
            # Process all CSV files in this patient directory
            csv_files = glob.glob(os.path.join(patient_dir, "*.csv"))
            
            for csv_file in csv_files:
                # Skip if "Unimpaired" in the file name
                if "Unimpaired" in csv_file:
                    continue
                
                # Extract activity ID from filename
                activity_id = int(os.path.basename(csv_file).split("_")[5].split('.')[0].replace('activity',''))
                if activity_id is None:
                    error_log.append((patient_id, None, "missing_activity_id", csv_file))
                    continue
                
                # Add to data map
                key = (patient_id, activity_id)
                if key not in ipsi_data_map:
                    ipsi_data_map[key] = {'corrected': None, 'original': None}
                
                ipsi_data_map[key]['corrected'] = csv_file
    else:
        print(f"Warning: Corrected ipsilateral directory {CORRECTED_IPSI_DIR} not found")
    
    # Now check for original ipsilateral data
    if os.path.exists(IPSI_DIR):
        print(f"Checking original ipsilateral directory: {IPSI_DIR}")
        # Get all patient directories
        patient_dirs = [d for d in glob.glob(os.path.join(IPSI_DIR, "ARAT_*")) if os.path.isdir(d)]
        
        for patient_dir in tqdm(patient_dirs, desc="Scanning original ipsi patient directories"):
            # Extract patient ID
            patient_id = int(os.path.basename(patient_dir).split('_')[1])
            if patient_id is None:
                error_log.append((None, None, "invalid_patient_dir", patient_dir))
                continue
            
            # Process all CSV files in this patient directory
            csv_files = glob.glob(os.path.join(patient_dir, "*.csv"))
            
            for csv_file in csv_files:
                # Skip if "Unimpaired" in the file name
                if "Unimpaired" in csv_file:
                    continue
                
                # Extract activity ID from filename
                activity_id = int(os.path.basename(csv_file).split("_")[5].split('.')[0].replace('activity',''))
                if activity_id is None:
                    error_log.append((patient_id, None, "missing_activity_id", csv_file))
                    continue
                
                # Add to data map
                key = (patient_id, activity_id)
                if key not in ipsi_data_map:
                    ipsi_data_map[key] = {'corrected': None, 'original': None}
                
                ipsi_data_map[key]['original'] = csv_file
    else:
        print(f"Warning: Original ipsilateral directory {IPSI_DIR} not found")
    
    # Process all ipsilateral data, prioritizing corrected data
    print("Processing ipsilateral data (prioritizing corrected data)...")
    
    for (patient_id, activity_id), data_sources in tqdm(ipsi_data_map.items(), desc="Processing ipsilateral data"):
        # Initialize patient data if not exists
        if patient_id not in all_patient_data:
            all_patient_data[patient_id] = {
                'top': {},   # Will store activity_id -> object location arrays
                'ipsi': {}   # Will store activity_id -> object location arrays
            }
        
        # Determine which file to use (prioritize corrected)
        csv_file = data_sources['corrected'] if data_sources['corrected'] else data_sources['original']
        source_type = "corrected" if data_sources['corrected'] else "original"
        
        if csv_file:
            # Process CSV file
            object_data = process_csv_file(csv_file, patient_id, activity_id)
            
            if object_data is None or object_data.size == 0:
                # No valid object location data
                no_object_data.add((patient_id, activity_id, 'ipsi'))
                error_log.append((patient_id, activity_id, "no_object_data", 
                                f"View: ipsi ({source_type}), File: {os.path.basename(csv_file)}"))
                continue
            
            # Store object location data
            all_patient_data[patient_id]['ipsi'][activity_id] = object_data
            
            if source_type == "corrected" and data_sources['original']:
                print(f"Using corrected data for patient {patient_id}, activity {activity_id} (ignoring original)")
        else:
            print(f"No ipsilateral data found for patient {patient_id}, activity {activity_id}")
    
    # Save error log
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    error_log_file = os.path.join(OUTPUT_DIR, f"object_error_log_{timestamp}.txt")
    
    with open(error_log_file, 'w') as f:
        f.write(f"Object Location Error Log - {timestamp}\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary of missing object data
        f.write("Patients/Activities with No Object Location Data:\n")
        f.write("-" * 50 + "\n")
        for patient_id, activity_id, view_type in sorted(no_object_data):
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
            output_file = os.path.join(OUTPUT_DIR, f"object_patients_91_to_110.pkl")
        else:
            # Regular group
            start_id = group[0]
            end_id = group[-1]
            output_file = os.path.join(OUTPUT_DIR, f"object_patients_{start_id}_to_{end_id}.pkl")
        
        # Collect data for this group
        group_data = {patient_id: all_patient_data[patient_id] for patient_id in group}
        
        # Save to pickle file
        with open(output_file, 'wb') as f:
            pickle.dump(group_data, f)
        
        print(f"Saved {len(group)} patients to {output_file}")
    
    # Print summary
    print(f"Successfully created {len(patient_groups)} pickle files in {OUTPUT_DIR}")
    print(f"Found {len(no_object_data)} patient/activity/view combinations with no object data")
    print(f"Found {len(file_errors)} patient/activity/view combinations with file errors")
    print(f"Total of {len(error_log)} detailed errors logged")

if __name__ == "__main__":
    process_object_locations()