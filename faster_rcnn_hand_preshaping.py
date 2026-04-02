import os
import pickle
import torch
import numpy as np
import cv2
import pandas as pd
import torchvision.models.detection as detection

camera_box = 'ipsi'
# Path to the CSV file with camera assignments
ipsi_contra_csv = "D:\\nature_everything\\camera_assignments.csv"


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
    return person_boxes[0][0]


def precompute_bboxes(pickle_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Read camera assignments
    camera_df = pd.read_csv(ipsi_contra_csv)
    patient_to_ipsilateral = dict(zip(camera_df['patient_id'], camera_df['ipsilateral_camera_id']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    faster_rcnn = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    faster_rcnn.eval()
    faster_rcnn.to(device)

    # Load the single hand-preshaping pickle file
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    for camera_id in data:
        # Filter by camera based on camera_box setting
        if camera_box == 'top':
            if camera_id != 'cam3':
                continue
        else:
            if camera_id == 'cam3':
                continue

        for sample in data[camera_id]:
            patient_id = sample['patient_id']
            sample_camera_id = sample['CameraId']

            # Check if this camera is the target camera for the patient
            if camera_box == 'ipsi':
                target_camera = patient_to_ipsilateral.get(patient_id)
            else:
                target_camera = 'cam3'

            if target_camera == sample_camera_id:
                frames = sample['frames']
                video_id = (f"patient_{patient_id}_task_{sample['activity_id']}_"
                            f"{sample_camera_id}_preshaping")
                bboxes = []

                for frame in frames:
                    bbox = detect_person_bbox(frame, faster_rcnn, device)
                    bboxes.append(bbox)

                bbox_file = os.path.join(output_dir, f"{video_id}_bboxes.pkl")
                with open(bbox_file, 'wb') as f:
                    pickle.dump(bboxes, f)
                print(f"Saved bounding boxes for {video_id} to {bbox_file}")


if __name__ == '__main__':
    precompute_bboxes(
        pickle_path='D:/nature_everything/nature_dataset/hand_preshaping_exercise1.pkl',
        output_dir='D:/nature_everything/frcnn_boxes_preshaping/bboxes_' + camera_box
    )
