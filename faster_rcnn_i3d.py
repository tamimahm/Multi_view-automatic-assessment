# precompute_bboxes.py
import os
import pickle
import glob
import torch
import numpy as np
from PIL import Image
import cv2
import torchvision.models.detection as detection

def detect_person_bbox(frame, faster_rcnn, device='cuda'):
    if isinstance(frame, np.ndarray):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = np.array(frame)
    img_tensor = torch.from_numpy(frame_rgb / 255.0).permute(2, 0, 1).float().to(device)
    
    with torch.no_grad():
        prediction = faster_rcnn([img_tensor])[0]
    
    person_boxes = []
    scores = prediction['scores'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    boxes = prediction['boxes'].cpu().numpy()

    for i, label in enumerate(labels):
        if label == 1:  # Person class (COCO class ID 1)
            person_boxes.append((boxes[i], scores[i]))
    
    if len(person_boxes) == 0:
        return None
    
    person_boxes.sort(key=lambda x: x[1], reverse=True)
    return person_boxes[0][0]  # Return the bounding box with highest score

def precompute_bboxes(pickle_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load pre-trained Faster R-CNN model
    faster_rcnn = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    faster_rcnn.eval()
    faster_rcnn.to(device)

    pickle_files = glob.glob(os.path.join(pickle_dir, '*.pkl'))
    for pkl_file in pickle_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            for camera_id in data:
                if camera_id == 'cam3':
                    for segments_group in data[camera_id]:
                        for segment in segments_group:
                            frames = segment['frames']
                            video_id = (f"patient_{segment['patient_id']}_task_{segment['activity_id']}_"
                                        f"{segment['CameraId']}_seg_{segment['segment_id']}")
                            bboxes = []

                            for frame in frames:
                                bbox = detect_person_bbox(frame, faster_rcnn, device)
                                bboxes.append(bbox)

                            # Save bounding boxes to a pickle file
                            bbox_file = os.path.join(output_dir, f"{video_id}_bboxes.pkl")
                            with open(bbox_file, 'wb') as f:
                                pickle.dump(bboxes, f)
                            print(f"Saved bounding boxes for {video_id} to {bbox_file}")

if __name__ == '__main__':
    precompute_bboxes(
        pickle_dir='D:/pickle_dir',
        output_dir='D:/Github/Multi_view-automatic-assessment/bboxes'
    )