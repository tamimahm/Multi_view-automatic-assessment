import os
import cv2
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
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
    """
    Loads the task_final and segment_final CSV files containing ratings.
    
    Expected files in csv_dir:
      - task_final.csv with columns: PatientTaskHandMappingId, Completed, Initialized, Time, Impaired, Rating, TherapistId, CreatedAt, ModifiedAt, Finger
      - segment_final.csv with columns: PatientTaskHandMappingId, SegmentId, Completed, Initialized, Time, Impaired, Rating, TherapistId, CreatedAt, ModifiedAt, Finger
      
    Returns:
      Two dictionaries:
         task_ratings: mapping PatientTaskHandMappingId to a dictionary with keys 't1' and (optionally) 't2'
                       for task ratings.
         segment_ratings: mapping PatientTaskHandMappingId to a dictionary with keys 't1' and (optionally) 't2',
                          where each value is itself a dictionary mapping SegmentId to its rating.
    """   
    task_file = os.path.join(csv_dir, "task_final_updated.csv")
    segment_file = os.path.join(csv_dir, "segment_final_updated.csv")
    
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
        # If already both t1 and t2 exist, ignore extra groups.
    
    return task_ratings, segment_ratings


def extract_frames_from_segment(video_path, start_time, end_time,remaining_segments,incomplete_flag, num_frames=10, target_size=(256,256)):
    """
    Extracts a fixed number of frames from a segment of a video.
    
    Args:
      video_path (str): Path to the video file.
      start_time (float): Start time (in seconds or frame number) of the segment.
      end_time (float): End time (in seconds or frame number) of the segment.
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
    if incomplete_flag==1:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames=remaining_segments*num_frames
    else:
        end_frame = int(end_time)
    # Here start_time and end_time are assumed to be in frame numbers
    start_frame = int(start_time)
    
    # Get the total number of frames in the video
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # If end_frame exceeds total frames, adjust it
    if end_frame > total_video_frames:
        end_frame = total_video_frames    
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

def process_videos_updated(video_dir, csv_dir, vid, num_frames=10, target_size=(256,256)):
    """
    Processes video files using the updated CSV information:
      - Merges CSV files based on PatientTaskHandmappingId.
      - Loads task and segment ratings from their respective CSV files.
      - For each record, extracts patient and activity ids from the filename.
      - For each segment, extracts frames and organizes samples by camera view.
      - If no ratings are available in both task and segment files, the video filename is recorded.
      
    Note: If a segment has identical start and end times (indicating incomplete segments due to low task ratings
          like 0 or 1), that segment is skipped and the video filename is not saved in save_seg_video.
      
    Args:
      video_dir (str): Base directory containing patient folders.
      csv_dir (str): Directory containing the CSV files.
      num_frames (int): Number of frames to extract per segment.
      target_size (tuple): Size to which frames are resized.
      
    Returns:
      datasets: A dictionary with keys for each camera view (e.g., 'cam1', 'cam3', 'cam4') and values
                as lists of sample dictionaries. Each sample contains:
                   - patient_id, activity_id, CameraId, segment (start, end), frames, task_ratings, segment_ratings.
      save_seg_video: A list of video filenames (with full paths) that encountered errors during segment frame extraction.
      no_rating_files: A list of video filenames (with full paths) for which no ratings were found.
    """
    records = load_video_segments_info(csv_dir)
    task_ratings_dict, segment_ratings_dict = load_rating_info(csv_dir)
    
    # We expect 4 camera ids per mapping but ignore cam2 if needed.
    valid_cameras = {"cam1", "cam3", "cam4"}
    datasets = {cam: [] for cam in valid_cameras}
    save_seg_video = []
    no_rating_files = []
    zero_rating_vid=[]
    invalid_camera_unim=0
    no_rating=0
    no_video=0
    incomplete_segment=0
    no_frame=0
    seg_task_0=[]
    for rec in records:
        camera_id = 'cam' + str(rec["CameraId"])
        mapping_id = rec["PatientTaskHandmappingId"]
        patient_id = rec["patient_id"]
        if patient_id< (vid+1)*10 and patient_id>= (vid)*10:
            activity_id = rec["activity_id"]
            file_name = rec["FileName"]
            segments = rec["segments"]
            # Attach ratings (list may have 1 or 2 ratings)
            rec["task_ratings"] = task_ratings_dict.get(mapping_id, [])
            rec["segment_ratings"] = segment_ratings_dict.get(mapping_id, [])
            # Skip if camera_id is not in our valid set (e.g., ignore cam2)
            if camera_id not in valid_cameras or 'Unimpaired' in file_name:
                invalid_camera_unim+=1
                continue   
            if (rec["task_ratings"]['t1']==1 and rec["task_ratings"].get('t2', 0)==1):

                seg = rec.get("segment_ratings", {})
                
                # check t1
                all_t1_zero = True
                for v in seg.get('t1', {}).values():
                    if v != 0:
                        all_t1_zero = False
                        break

                # check t2
                all_t2_zero = True
                for v in seg.get('t2', {}).values():
                    if v != 0:
                        all_t2_zero = False
                        break

                if all_t1_zero and all_t2_zero:
                    seg_task_0.append(rec['FileName'])   

                if patient_id<100:
                    video_path = os.path.join(video_dir, 'ARAT_0'+str(patient_id), file_name)
                else:  
                    video_path = os.path.join(video_dir, 'ARAT_'+str(patient_id), file_name)  
                zero_rating_vid.append(video_path)
            # If no ratings are available in both task and segment files, record the filename.
            if (not rec["task_ratings"]) and (not rec["segment_ratings"]) :
                if patient_id<100:
                    video_path = os.path.join(video_dir, 'ARAT_0'+str(patient_id), file_name)
                else:  
                    video_path = os.path.join(video_dir, 'ARAT_'+str(patient_id), file_name)  
                no_rating_files.append(video_path)
                no_rating+=1
                continue
            
            # Construct the full path to the video file.
            # Here we assume patient folders are named like 'ARAT_0XX' where XX is the patient id.
            # Construct the initial video path using the given file_name
            if patient_id<100:
                video_path = os.path.join(video_dir, 'ARAT_0'+str(patient_id), file_name)
            else:  
                video_path = os.path.join(video_dir, 'ARAT_'+str(patient_id), file_name)  

            # Check if the file exists at that path
            if not os.path.exists(video_path):
                # If the file doesn't exist, check if the file name ends with ".mp4"
                if file_name.endswith(".mp4"):
                    # If it doesn't already have a space before ".mp4", create a variant with a space
                    if not file_name.endswith(" .mp4"):
                        file_name_with_space = file_name[:-4] + " .mp4"
                        if patient_id<100:
                            video_path_with_space = os.path.join(video_dir, 'ARAT_0' + str(patient_id), file_name_with_space)
                        else:
                            video_path_with_space = os.path.join(video_dir, 'ARAT_' + str(patient_id), file_name_with_space)                    
                        # If the variant with a space exists, use that path
                        if os.path.exists(video_path_with_space):
                            video_path = video_path_with_space
                        else:
                            print(f"Video file {video_path} (and variant {video_path_with_space}) does not exist. Skipping.")
                            no_video+=1
                            continue
                    else:
                        # The file name already ends with " .mp4", so no alternative exists
                        print(f"Video file {video_path} does not exist. Skipping.")
                        no_video+=1
                        continue
                else:
                    # If the file name does not end with ".mp4", there's no special case to try
                    print(f"Video file {video_path} does not exist. Skipping.")
                    no_video+=1
                    continue

            # At this point, video_path exists (either the original or the space-added variant)
            # Proceed with processing video_path...
            # Process each segment in the video
            incomplete_flag=0
            remaining_segments=0
            frames=[]
            for seg_index, seg in enumerate(segments):
                start_time, end_time = seg
                # If the segment is incomplete (start and end are the same), skip extraction without error logging.
                if start_time == end_time:
                    incomplete_segment+=1
                    continue
                if end_time==max(max(segments)) and seg_index!=3:
                    incomplete_segment+=1
                    remaining_segments=len(segments)-seg_index
                    incomplete_flag=1                    
                frames.extend(extract_frames_from_segment(video_path, start_time, end_time,remaining_segments,incomplete_flag, num_frames, target_size))
                if not frames:
                    save_seg_video.append(video_path)
                    no_frame+=1
                    continue
            sample = {
                "patient_id": patient_id,
                "activity_id": activity_id,
                "CameraId": camera_id,
                "frames": frames,
                "task_ratings": rec["task_ratings"]
            }
            datasets[camera_id].append(sample)    
    return datasets, save_seg_video, no_rating_files,zero_rating_vid

if __name__ == "__main__":
    # Directories (adjust as necessary)
    video_dir = r"D:\all_ARAT_videos"
    csv_dir = r"D:\nature_everything"
    for vid in range (11): 
        print(vid)   
        # Process videos based on the updated CSVs, segmentation info, and ratings.
        datasets, save_seg_video, no_rating_files,zero_rating_vid = process_videos_updated(video_dir, csv_dir,vid,num_frames=20, target_size=(256,256))
        # # Assuming datasets is a dictionary or list that needs to be saved as a .pkl file
        with open("D:/nature_everything/nature_dataset/task_dataset/datasets_"+str(vid)+".pkl", "wb") as f:
            pickle.dump(datasets, f)
        # # Saving save_seg_video as CSV
        # save_seg_video_df = pd.DataFrame(save_seg_video)
        # save_seg_video_df.to_csv("D:/files_database/save_seg_video_"+str(vid)+".csv", index=False)

        # # Saving no_rating_files as CSV
        # no_rating_files_df = pd.DataFrame(no_rating_files)
        # no_rating_files_df.to_csv("D:/files_database/no_rating_files_"+str(vid)+".csv", index=False)
        # # Saving no_rating_files as CSV
        # zero_rating_files_df = pd.DataFrame(zero_rating_vid)
        # zero_rating_files_df.to_csv("D:/files_database/zero_rating_files_"+str(vid)+".csv", index=False)





        # # Report number of processed samples per camera view
        # for cam, samples in datasets.items():
        #     print(f"Camera {cam}: {len(samples)} samples processed.")
        
        # if save_seg_video:
        #     print("\nVideos that encountered errors during segment extraction:")
        #     for video in save_seg_video:
        #         print(video)
        
        # if no_rating_files:
        #     print("\nVideos with no ratings available:")
        #     for video in no_rating_files:
        #         print(video)
