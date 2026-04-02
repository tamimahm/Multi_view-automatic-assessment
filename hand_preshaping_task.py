import os
import cv2
import pandas as pd
import numpy as np
import pickle


def load_video_segments_info(csv_dir):
    """
    Loads and merges the updated CSV files.
    Returns a list of record dicts filtered to activity 1 only.
    """
    pth_file = os.path.join(csv_dir, "pth_updated.csv")
    seg_file = os.path.join(csv_dir, "segmentation_updated.csv")

    pth_df = pd.read_csv(pth_file)
    seg_df = pd.read_csv(seg_file)

    merged_df = pd.merge(pth_df, seg_df, left_on='PatientTaskHandmappingId', right_on='PatientTaskHandMappingId')

    grouped = merged_df.groupby(['FileName', 'PatientTaskHandmappingId', 'CameraId'])

    records = []
    for (file_name, mapping_id, camera_id), group in grouped:
        segments = list(zip(group['Start'], group['End']))
        parts = file_name.split("_")
        if len(parts) < 5:
            print(f"Filename {file_name} does not match expected format. Skipping.")
            continue
        patient_id = int(parts[1].strip())
        activity_part = parts[-1]
        activity_id = activity_part.split('.')[0].replace("activity", "").strip()

        # Only keep exercise 1
        if activity_id != "1":
            continue

        record = {
            "FileName": file_name,
            "PatientTaskHandmappingId": mapping_id,
            "CameraId": int(camera_id),
            "patient_id": patient_id,
            "activity_id": int(activity_id),
            "segments": segments
        }
        records.append(record)

    return records


def load_rating_info(csv_dir):
    """
    Loads task and segment ratings from CSV files.
    """
    task_file = os.path.join(csv_dir, "task_final_updated.csv")
    segment_file = os.path.join(csv_dir, "segment_final_updated.csv")

    task_df = pd.read_csv(task_file)
    segment_df = pd.read_csv(segment_file)

    task_ratings = {}
    for _, row in task_df.iterrows():
        mapping_id = row['PatientTaskHandMappingId']
        rating = row['Rating']
        if pd.notna(rating):
            if mapping_id not in task_ratings:
                task_ratings[mapping_id] = {'t1': rating}
            elif 't2' not in task_ratings[mapping_id]:
                task_ratings[mapping_id]['t2'] = rating

    segment_ratings = {}
    grouped_seg = segment_df.groupby(['PatientTaskHandMappingId', 'TherapistId'])
    for (mapping_id, therapist_id), group in grouped_seg:
        seg_rating_dict = {}
        for _, row in group.iterrows():
            seg_id = row['SegmentId']
            rating = row['Rating']
            if pd.notna(rating):
                seg_rating_dict[seg_id] = rating
        if not seg_rating_dict:
            continue
        if mapping_id not in segment_ratings:
            segment_ratings[mapping_id] = {'t1': seg_rating_dict}
        elif 't1' in segment_ratings[mapping_id] and 't2' not in segment_ratings[mapping_id]:
            segment_ratings[mapping_id]['t2'] = seg_rating_dict

    return task_ratings, segment_ratings


def extract_frames(video_path, start_frame, end_frame, num_frames=20, target_size=(256, 256)):
    """
    Extracts evenly spaced frames from start_frame to end_frame.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Warning: Unable to get FPS for video {video_path}. Skipping.")
        cap.release()
        return []

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_frame)
    end_frame = int(end_frame)

    if end_frame > total_video_frames:
        end_frame = total_video_frames
    if end_frame - start_frame <= 0:
        print(f"Invalid frame range for video: {video_path} (start: {start_frame}, end: {end_frame})")
        cap.release()
        return []

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


def resolve_video_path(video_dir, patient_id, file_name):
    """
    Constructs the video path, handling the space-before-.mp4 naming quirk.
    Returns the path if found, else None.
    """
    if patient_id < 100:
        folder = os.path.join(video_dir, 'ARAT_0' + str(patient_id))
    else:
        folder = os.path.join(video_dir, 'ARAT_' + str(patient_id))

    video_path = os.path.join(folder, file_name)
    if os.path.exists(video_path):
        return video_path

    # Try variant with space before .mp4
    if file_name.endswith(".mp4") and not file_name.endswith(" .mp4"):
        alt_name = file_name[:-4] + " .mp4"
        alt_path = os.path.join(folder, alt_name)
        if os.path.exists(alt_path):
            return alt_path

    return None


def process_hand_preshaping(video_dir, csv_dir, num_frames=20, target_size=(256, 256)):
    """
    Processes exercise 1 videos for hand pre-shaping analysis.
    Extracts frames from start of IP (segment 0) to end of T (segment 1).
    Saves all patients in a single dataset.

    Returns:
      datasets: dict with keys 'cam1', 'cam3', 'cam4', each a list of samples.
                Each sample: {patient_id, activity_id, CameraId, frames,
                              ip_start, t_end, task_ratings, segment_ratings}
      skipped_videos: list of videos that could not be processed.
      no_rating_files: list of videos with no ratings.
    """
    records = load_video_segments_info(csv_dir)
    task_ratings_dict, segment_ratings_dict = load_rating_info(csv_dir)

    valid_cameras = {"cam1", "cam3", "cam4"}
    datasets = {cam: [] for cam in valid_cameras}
    skipped_videos = []
    no_rating_files = []

    for rec in records:
        camera_id = 'cam' + str(rec["CameraId"])
        mapping_id = rec["PatientTaskHandmappingId"]
        patient_id = rec["patient_id"]
        activity_id = rec["activity_id"]
        file_name = rec["FileName"]
        segments = rec["segments"]

        # Skip invalid cameras and unimpaired hand
        if camera_id not in valid_cameras or 'Unimpaired' in file_name:
            continue

        # Attach ratings
        task_ratings = task_ratings_dict.get(mapping_id, {})
        seg_ratings = segment_ratings_dict.get(mapping_id, {})

        # Skip if no ratings at all
        if not task_ratings and not seg_ratings:
            no_rating_files.append(file_name)
            continue

        # We need at least segments 0 (IP) and 1 (T)
        if len(segments) < 2:
            print(f"Not enough segments for {file_name}. Skipping.")
            skipped_videos.append(file_name)
            continue

        ip_start, ip_end = segments[0]
        t_start, t_end = segments[1]

        # Handle incomplete IP segment (start==end)
        if ip_start == ip_end:
            ip_start = 0

        # Handle incomplete T segment
        if t_start == t_end:
            # Use IP end as T start, and video end will be handled in extract
            t_start = ip_end
            t_end = t_start  # will be caught below

        # The range we want: start of IP to end of T
        frame_start = ip_start
        frame_end = t_end

        if frame_end <= frame_start:
            print(f"Invalid IP-to-T range for {file_name} (start: {frame_start}, end: {frame_end}). Skipping.")
            skipped_videos.append(file_name)
            continue

        # Resolve video path
        video_path = resolve_video_path(video_dir, patient_id, file_name)
        if video_path is None:
            print(f"Video file not found for {file_name}. Skipping.")
            skipped_videos.append(file_name)
            continue

        # Extract frames from IP start to T end
        frames = extract_frames(video_path, frame_start, frame_end, num_frames, target_size)
        if not frames:
            skipped_videos.append(file_name)
            continue

        # Extract segment ratings for IP (seg 1) and T (seg 2) only
        ip_t_segment_ratings = {}
        if seg_ratings:
            for therapist_key, ratings in seg_ratings.items():
                ip_t_segment_ratings[therapist_key] = {
                    'IP': ratings.get(1, None),   # SegmentId 1 = IP
                    'T': ratings.get(2, None)      # SegmentId 2 = T
                }

        sample = {
            "patient_id": patient_id,
            "activity_id": activity_id,
            "CameraId": camera_id,
            "ip_start": int(frame_start),
            "t_end": int(frame_end),
            "frames": frames,
            "task_ratings": task_ratings,
            "segment_ratings": ip_t_segment_ratings
        }
        datasets[camera_id].append(sample)

    return datasets, skipped_videos, no_rating_files


if __name__ == "__main__":
    video_dir = r"D:\all_ARAT_videos"
    csv_dir = r"D:\nature_everything"

    print("Processing hand pre-shaping data (exercise 1, IP to T)...")
    datasets, skipped_videos, no_rating_files = process_hand_preshaping(
        video_dir, csv_dir, num_frames=20, target_size=(256, 256)
    )

    # Report counts
    for cam, samples in datasets.items():
        print(f"Camera {cam}: {len(samples)} samples processed.")

    if skipped_videos:
        print(f"\n{len(skipped_videos)} videos skipped (errors/missing).")

    if no_rating_files:
        print(f"{len(no_rating_files)} videos had no ratings.")

    # Save as a single pickle file
    output_path = "D:/nature_everything/nature_dataset/hand_preshaping_exercise1.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(datasets, f)
    print(f"\nSaved to {output_path}")
