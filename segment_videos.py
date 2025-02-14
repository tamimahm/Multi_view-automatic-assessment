import os
import cv2
import pandas as pd
import numpy as np

def load_video_segments_info(csv_dir):
    """
    Loads and merges the updated CSV files.

    Expected files in csv_dir:
      - pth_updated.csv with columns: FileName, PatientTaskHandmappingId, CameraId
      - segmentation_updated.csv with columns: PatientTaskHandMappingId, SegmentId, Start, End

    Returns:
      A list of dictionaries. Each dictionary corresponds to one video file record with keys:
         'FileName', 'PatientTaskHandmappingId', 'CameraId', 'patient_id', 'activity_id', 'segments'
      where 'segments' is a list of (start, end) tuples.
    """
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
        patient_id = int(parts[1].strip())  # e.g., "ARAT_01"
        # Activity part is assumed to be the last component (e.g., "activity11.mp4")
        activity_part = parts[-1]
        activity_id = activity_part.split('.')[0].replace("activity", "").strip()
        # Skip specified activities
        if activity_id in skip_activities:
            continue        
        record = {
            "FileName": file_name,
            "PatientTaskHandmappingId": mapping_id,
            "CameraId": int(camera_id),  # e.g., "cam1", "cam2", etc.
            "patient_id": patient_id,
            "activity_id": int(activity_id),
            "segments": segments
        }
        records.append(record)
    
    return records

def extract_frames_from_segment(video_path, start_time, end_time, num_frames=10, target_size=(256,256)):
    """
    Extracts a fixed number of frames from a segment of a video.
    
    Args:
      video_path (str): Path to the video file.
      start_time (float): Start time (in seconds) of the segment.
      end_time (float): End time (in seconds) of the segment.
      num_frames (int): Number of frames to sample.
      target_size (tuple): Target (width, height) for resizing frames.
      
    Returns:
      List of preprocessed frames as NumPy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Warning: Unable to get FPS for video {video_path}. Skipping segment.")
        cap.release()
        return []
    
    start_frame = int(start_time)
    end_frame = int(end_time)
    total_frames = end_frame - start_frame
    if total_frames <= 0:
        print(f"Invalid segment times for video: {video_path} (start: {start_time}, end: {end_time})")
        cap.release()
        return []
    
    # Calculate evenly spaced frame indices
    frame_indices = np.linspace(start_frame, end_frame - 1, num=num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {idx} could not be read from {video_path}.")
            continue
        frame = cv2.resize(frame, target_size)
        frames.append(frame)
    
    cap.release()
    return frames

def process_videos_updated(video_dir, csv_dir, num_frames=10, target_size=(256,256)):
    """
    Processes video files using the updated CSV information:
      - Merges CSV files based on PatientTaskHandmappingId.
      - For each record, extracts patient and activity ids from the filename.
      - For each segment, extracts frames and organizes samples by camera view.
      
    Args:
      video_dir (str): Base directory containing patient folders.
      csv_dir (str): Directory containing the updated CSV files.
      num_frames (int): Number of frames to extract per segment.
      target_size (tuple): Size to which frames are resized.
      
    Returns:
      A dictionary with keys for each camera view (e.g., 'cam1', 'cam3', 'cam4') and values
      as lists of sample dictionaries. Each sample contains:
         - patient_id, activity_id, CameraId, segment (start, end), and frames.
    """
    records = load_video_segments_info(csv_dir)
    
    # We expect 4 camera ids per mapping but ignore cam2 if needed.
    valid_cameras = {"cam1", "cam3", "cam4"}
    datasets = {cam: [] for cam in valid_cameras}
    save_seg_video=[]
    for rec in records:
        camera_id = 'cam'+str(rec["CameraId"])
        # Skip if camera_id is not in our valid set (e.g., ignore cam2)
        if camera_id not in valid_cameras:
            continue
        
        patient_id = rec["patient_id"]
        activity_id = rec["activity_id"]
        file_name = rec["FileName"]
        segments = rec["segments"]
        
        # Construct the full path to the video file: video_dir/patient_id/file_name
        video_path = os.path.join(video_dir, 'ARAT_0'+str(patient_id), file_name)
        if not os.path.exists(video_path):
            print(f"Video file {video_path} does not exist. Skipping.")
            continue
        
        # Process each segment in the video
        for seg in segments:
            start_time, end_time = seg
            frames = extract_frames_from_segment(video_path, start_time, end_time, num_frames, target_size)
            if not frames:
                save_seg_video.append(video_path)
                continue
            sample = {
                "patient_id": patient_id,
                "activity_id": activity_id,
                "CameraId": camera_id,
                "segment": seg,
                "frames": frames
            }
            datasets[camera_id].append(sample)
    
    return datasets,save_seg_video

if __name__ == "__main__":
    # Directories (adjust as necessary)
    video_dir = r"D:\Chicago_study\all_ARAT_videos"
    csv_dir = r"D:\Chicago_study\files"
    
    # Process videos based on the updated CSVs and segmentation information
    datasets,save_seg_video = process_videos_updated(video_dir, csv_dir)
    
    # Report number of processed samples per camera view
    for cam, samples in datasets.items():
        print(f"Camera {cam}: {len(samples)} samples processed.")
