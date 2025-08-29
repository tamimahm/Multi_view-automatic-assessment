import os
import cv2
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

ALL_COMPOSITES = ['SEAFR', 'TS', 'ROME', 'FPS', 'WPAT', 'HA', 'DP', 
                  'DPO', 'SAT', 'DMR','THS','PP',  'FPO', 'SA']

SEGMENT_COMPOSITE_MAPPING = {
    0: ['SEAFR', 'TS', 'ROME', 'FPS'],      # IP segment
    1: ['WPAT', 'HA', 'DP','SA'],           # T segment  
    2: ['SAT', 'FPS', 'TS', 'DPO'],         # MTR segment
    3: ['FPO', 'DMR', 'TS', 'FPS']          # PR segment
}

def load_video_segments_info(csv_dir):
    pth_file = os.path.join(csv_dir, "pth_updated.csv")
    seg_file = os.path.join(csv_dir, "segmentation_updated.csv")
    
    # Read CSVs
    pth_df = pd.read_csv(pth_file)
    seg_df = pd.read_csv(seg_file)
    
    # Merge based on PatientTaskHandmappingId (note: column names differ in case)
    merged_df = pd.merge(pth_df, seg_df, left_on='PatientTaskHandmappingId', right_on='PatientTaskHandMappingId')
    
    # Group by FileName, PatientTaskHandmappingId, and CameraId to aggregate segments
    grouped = merged_df.groupby(['FileName', 'PatientTaskHandmappingId', 'CameraId'])
    
    records = []
    # Define activities to skip
    skip_activities = {"7", "17", "18", "19"}
    for (file_name, mapping_id, camera_id), group in grouped:
        # Aggregate segments as list of tuples (start, end)
        segments = list(zip(group['Start'], group['End']))
        # Split filename to extract patient id and activity id.
        # Expected filename format: ARAT_01_right_Impaired_cam1_activity11.mp4
        parts = file_name.split("_")
        if len(parts) < 5:
            print(f"Filename {file_name} does not match expected format. Skipping.")
            continue
        # Assume patient id is the numeric part from the second token, e.g., "01" from "ARAT_01"
        patient_id = int(parts[1].strip())
        # Activity part is assumed to be the last component (e.g., "activity11.mp4")
        activity_part = parts[-1]
        activity_id = activity_part.split('.')[0].replace("activity", "").strip()
        # Skip specified activities
        if activity_id in skip_activities:
            continue        
        record = {
            "FileName": file_name,
            "PatientTaskHandmappingId": mapping_id,
            "CameraId": int(camera_id),  # e.g., 1,2,3, etc.
            "patient_id": patient_id,
            "activity_id": int(activity_id),
            "segments": segments
        }
        records.append(record)
    
    return records

def load_rating_info(csv_dir):
    task_file = os.path.join(csv_dir, "task_final.csv")
    segment_file = os.path.join(csv_dir, "segment_final.csv")
    
    task_df = pd.read_csv(task_file)
    segment_df = pd.read_csv(segment_file)
    
    # Process task ratings: store first rating as 't1' and second (if available) as 't2'.
    task_ratings = {}
    for _, row in task_df.iterrows():
        mapping_id = row['PatientTaskHandMappingId']
        rating = row['Rating']
        if pd.notna(rating):
            if mapping_id not in task_ratings:
                task_ratings[mapping_id] = {'t1': rating}
            elif 't2' not in task_ratings[mapping_id]:
                task_ratings[mapping_id]['t2'] = rating
            # Ignore any additional ratings.
    
    # Process segment ratings:
    # For each mapping id and therapist, build a dictionary mapping each SegmentId to its rating.
    segment_ratings = {}
    grouped = segment_df.groupby(['PatientTaskHandMappingId', 'TherapistId'])
    for (mapping_id, therapist_id), group in grouped:
        seg_rating_dict = {}
        for _, row in group.iterrows():
            # Assuming segment_final.csv has a 'SegmentId' column.
            seg_id = row['SegmentId']
            rating = row['Rating']
            if pd.notna(rating):
                seg_rating_dict[seg_id] = rating
        if not seg_rating_dict:
            continue
        # For each mapping_id, store the first therapist's segment ratings as 't1'
        # and if a second therapist is available, store their ratings as 't2'.
        if mapping_id not in segment_ratings:
            segment_ratings[mapping_id] = {'t1': seg_rating_dict}
        elif 't1' in segment_ratings[mapping_id] and 't2' not in segment_ratings[mapping_id]:
            segment_ratings[mapping_id]['t2'] = seg_rating_dict
    
    # Process composite ratings
    composite_ratings = {} 
    for (mapping_id, therapist_id), group in grouped:
        # Extract composite values for this therapist
        composite_values = []
        
        for composite_name in ALL_COMPOSITES:
            if composite_name in group.columns:
                # Get the composite value (should be 0 or 1)
                value = group[composite_name].iloc[0] if len(group) > 0 else 0
                # Ensure it's 0 or 1
                composite_values.append(1 if value == 1 else 0)
            else:
                composite_values.append(0)
        
        if mapping_id not in composite_ratings:
            composite_ratings[mapping_id] = {'t1': composite_values}
        elif 't1' in composite_ratings[mapping_id] and 't2' not in composite_ratings[mapping_id]:
            composite_ratings[mapping_id]['t2'] = composite_values
    
    print(f"Loaded ratings for {len(task_ratings)} tasks, {len(segment_ratings)} segment groups, {len(composite_ratings)} composite groups")
    
    return task_ratings, segment_ratings, composite_ratings

def combine_composite_ratings(composite_rating):
    """Combine composite ratings from two therapists using union (if either has 1, result is 1)"""
    if not composite_rating:
        return [0] * len(ALL_COMPOSITES)
    
    t1_ratings = composite_rating.get('t1', [0] * len(ALL_COMPOSITES))
    t2_ratings = composite_rating.get('t2', [0] * len(ALL_COMPOSITES))
    
    # Ensure both arrays have the same length
    max_len = max(len(t1_ratings), len(t2_ratings), len(ALL_COMPOSITES))
    t1_ratings.extend([0] * (max_len - len(t1_ratings)))
    t2_ratings.extend([0] * (max_len - len(t2_ratings)))
    
    # Union: if either therapist marked 1, result is 1
    combined = []
    for i in range(len(ALL_COMPOSITES)):
        combined.append(1 if (t1_ratings[i] == 1 or t2_ratings[i] == 1) else 0)
    
    return combined

def get_segment_composites(combined_composites, segment_id):
    """Get checked and unchecked composites for a specific segment"""
    if segment_id not in SEGMENT_COMPOSITE_MAPPING:
        return "", ""
    
    valid_composites = SEGMENT_COMPOSITE_MAPPING[segment_id]
    checked_composites = []
    unchecked_composites = []
    
    for composite_name in valid_composites:
        if composite_name in ALL_COMPOSITES:
            comp_index = ALL_COMPOSITES.index(composite_name)
            if comp_index < len(combined_composites):
                if combined_composites[comp_index] == 1:
                    checked_composites.append(composite_name)
                else:
                    unchecked_composites.append(composite_name)
            else:
                unchecked_composites.append(composite_name)
    
    return ",".join(checked_composites), ",".join(unchecked_composites)

def process_videos_updated(csv_dir):
    records = load_video_segments_info(csv_dir)
    task_ratings_dict, segment_ratings_dict, composite_ratings_dict = load_rating_info(csv_dir)
    
    # Track processed task-activity combinations
    processed_keys = set()
    csv_rows = []
    
    no_rating_files = []
    zero_rating_vid = []
    
    for rec in records:
        mapping_id = rec["PatientTaskHandmappingId"]
        patient_id = rec["patient_id"]
        activity_id = rec["activity_id"]
        file_name = rec["FileName"]
        segments = rec["segments"]
        task_time = 1+((segments[3][1] - segments[0][0]) / 30)
        hand_id = file_name.split('_')[2]
        
        task_rating = task_ratings_dict.get(mapping_id, {})
        segment_rating = segment_ratings_dict.get(mapping_id, {})
        composite_rating = composite_ratings_dict.get(mapping_id, {})
        
        # Create task-activity key
        task_activity_key = (patient_id, activity_id)
        
        # Skip if already processed
        if task_activity_key in processed_keys:
            continue
        
        # Check task rating conditions - now including task_rating = 0
        valid_task = False
        final_task_score = None
        is_zero_rating = False
        
        if 't1' in task_rating and 't2' in task_rating:
            # Both therapists available
            if task_rating['t1'] == task_rating['t2'] and task_rating['t1'] <2:  # Changed to <= 2 to include 0
                valid_task = True
                final_task_score = task_rating['t1']
                if task_rating['t1'] == 0:
                    is_zero_rating = True
        elif 't1' in task_rating:
            # Only t1 available
            if task_rating['t1'] <2:  # Changed to <= 2 to include 0
                valid_task = True
                final_task_score = task_rating['t1']
                if task_rating['t1'] == 0:
                    is_zero_rating = True
        elif 't2' in task_rating:
            # Only t2 available
            if task_rating['t2'] <2:  # Changed to <= 2 to include 0
                valid_task = True
                final_task_score = task_rating['t2']
                if task_rating['t2'] == 0:
                    is_zero_rating = True
        
        if not valid_task:
            zero_rating_vid.append(file_name)
            continue
        
        # Mark as processed
        processed_keys.add(task_activity_key)
        
        # Create video filenames
        if hand_id == 'left':
            ipsi_cam = 'cam4'
            contra_cam = 'cam1'
        else:  # right or default
            ipsi_cam = 'cam1'
            contra_cam = 'cam4'
        
        if patient_id >= 100:
            top_filename = f"ARAT_{patient_id}_{hand_id}_Impaired_cam3_activity{activity_id}.mp4"
            contra_filename = f"ARAT_{patient_id}_{hand_id}_Impaired_{contra_cam}_activity{activity_id}.mp4"
            ipsi_filename = f"ARAT_{patient_id}_{hand_id}_Impaired_{ipsi_cam}_activity{activity_id}.mp4"
        else:
            top_filename = f"ARAT_0{patient_id:02d}_{hand_id}_Impaired_cam3_activity{activity_id}.mp4"
            contra_filename = f"ARAT_0{patient_id:02d}_{hand_id}_Impaired_{contra_cam}_activity{activity_id}.mp4"
            ipsi_filename = f"ARAT_0{patient_id:02d}_{hand_id}_Impaired_{ipsi_cam}_activity{activity_id}.mp4"
        
        # Create CSV row - basic info always included
        csv_row = {
            'Top': top_filename,
            'Contralateral': contra_filename,
            'Ipsilateral': ipsi_filename,
            'task_score': final_task_score,
            'task_time': round(task_time, 2),
            'hand_id': hand_id
        }
        
        if is_zero_rating:
            # If task_rating is 0, fill all segment fields with "N/A"
            for seg_id in range(4):
                seg_key = f'seg{seg_id + 1}'  # seg1, seg2, seg3, seg4
                csv_row[f'{seg_key}_score'] = "N/A"
                csv_row[f'{seg_key}_checked'] = "N/A"
                csv_row[f'{seg_key}_unchecked'] = "N/A"
        else:
            # Normal processing for non-zero ratings
            # Combine composite ratings from both therapists
            combined_composites = combine_composite_ratings(composite_rating)
            
            # Get segment scores (combine from both therapists if available)
            # Get segment scores (combine from both therapists if available)
            segment_scores = {}
            for seg_id in range(4):  # segments 0, 1, 2, 3
                seg_score = None
                if 't1' in segment_rating and seg_id + 1 in segment_rating['t1']:
                    seg_score = segment_rating['t1'][seg_id + 1]  # segment IDs are 1-based in data
                elif 't2' in segment_rating and seg_id + 1 in segment_rating['t2']:
                    seg_score = segment_rating['t2'][seg_id + 1]
                
                segment_scores[seg_id] = seg_score if seg_score is not None else 0

            # Define default composites for segments with >0 and <3 scores but no checked composites
            DEFAULT_COMPOSITES = {
                0: 'TS',    # seg0 (IP segment) -> TS
                1: 'SA',    # seg1 (T segment) -> SA  
                2: 'SAT',   # seg2 (MTR segment) -> SAT
                3: 'TS'     # seg3 (PR segment) -> TS
            }

            # First pass: process all segments normally and collect checked composites
            segment_checked_composites = {}
            segment_unchecked_composites = {}

            for seg_id in range(4):
                seg_score = segment_scores[seg_id]
                
                if seg_score == 0:
                    segment_checked_composites[seg_id] = []
                    segment_unchecked_composites[seg_id] = []
                else:
                    seg_checked, seg_unchecked = get_segment_composites(combined_composites, seg_id)
                    segment_checked_composites[seg_id] = seg_checked.split(',') if seg_checked else []
                    segment_unchecked_composites[seg_id] = seg_unchecked.split(',') if seg_unchecked else []

            # Apply special logic for seg3 (MTR) based on seg2 (T)
            seg2_checked_count = len([comp for comp in segment_checked_composites[1] if comp])  # seg1 = T segment
            seg3_score = segment_scores[2]  # seg2 = MTR segment

            if seg2_checked_count >= 2:  # T segment has >= 2 checked components
                print(f"T segment has {seg2_checked_count} checked components, applying MTR logic")
                
                if seg3_score == 1:
                    # Check all 4 components for MTR
                    segment_checked_composites[2] = ['SAT', 'FPS', 'TS', 'DPO']
                    segment_unchecked_composites[2] = []
                    print("MTR score=1: Checking all 4 components")
                    
                elif seg3_score == 2:
                    # Check TS, SAT for MTR
                    segment_checked_composites[2] = ['TS', 'SAT']
                    segment_unchecked_composites[2] = ['FPS', 'DPO']
                    print("MTR score=2: Checking TS, SAT")

            # Add segment information to CSV
            for seg_id in range(4):
                seg_key = f'seg{seg_id + 1}'  # seg1, seg2, seg3, seg4
                seg_score = segment_scores[seg_id]
                
                if seg_score == 0:
                    # If segment score is 0, set composites to N/A
                    csv_row[f'{seg_key}_score'] = seg_score
                    csv_row[f'{seg_key}_checked'] = "N/A"
                    csv_row[f'{seg_key}_unchecked'] = "N/A"
                else:
                    # Use processed composite lists
                    checked_list = segment_checked_composites[seg_id]
                    unchecked_list = segment_unchecked_composites[seg_id]
                    
                    # Apply default composite rule if no composites are checked and score is >0 and <3
                    if 0 < seg_score < 3 and not any(checked_list):
                        default_composite = DEFAULT_COMPOSITES[seg_id]
                        valid_composites = SEGMENT_COMPOSITE_MAPPING[seg_id]
                        
                        checked_list = [default_composite]
                        unchecked_list = [comp for comp in valid_composites if comp != default_composite]
                        
                        print(f"Applied default rule for segment {seg_id} (score={seg_score}): checked={default_composite}")
                    
                    # Convert lists back to comma-separated strings
                    seg_checked = ",".join([comp for comp in checked_list if comp])
                    seg_unchecked = ",".join([comp for comp in unchecked_list if comp])
                    
                    csv_row[f'{seg_key}_score'] = seg_score
                    csv_row[f'{seg_key}_checked'] = seg_checked
                    csv_row[f'{seg_key}_unchecked'] = seg_unchecked
        csv_rows.append(csv_row)
    
    # Create DataFrame and save CSV
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        csv_filename = os.path.join(csv_dir, f"final_data.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Created CSV with {len(csv_rows)} rows: {csv_filename}")
        
        # Print summary of zero vs non-zero ratings
        zero_count = sum(1 for row in csv_rows if row['task_score'] == 0)
        non_zero_count = len(csv_rows) - zero_count
        print(f"  - Zero ratings (N/A segments): {zero_count}")
        print(f"  - Non-zero ratings (full data): {non_zero_count}")
    else:
        print(f"No valid data found for vid")
    
    print(f"Files with invalid ratings (>2 or disagreement): {len(zero_rating_vid)}")
    print(f"Files with no ratings: {len(no_rating_files)}")
    
    return csv_rows

if __name__ == "__main__":
    # Directories (adjust as necessary)
    video_dir = r"D:\all_ARAT_videos"
    csv_dir = r"D:\files_database"
    

    # Process videos based on the updated CSVs, segmentation info, and ratings.
    datasets = process_videos_updated(csv_dir)