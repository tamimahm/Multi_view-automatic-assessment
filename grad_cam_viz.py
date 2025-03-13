import os
import pickle
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_i3d import InceptionI3d
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# ------------------
# 1) Custom Dataset for Pickle Files with Precomputed Bounding Boxes
# ------------------
class PickleVideoDataset(Dataset):
    def __init__(self, pickle_dir, transform=None, bbox_dir=None):
        self.pickle_dir = pickle_dir
        self.transform = transform
        self.bbox_dir = bbox_dir
        self.pickle_files = glob.glob(os.path.join(pickle_dir, '*.pkl'))
        self.samples = []

        for pkl_file in self.pickle_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                for camera_id in data:
                    if camera_id == 'cam3':
                        for segments_group in data[camera_id]:
                            for segment in segments_group:
                                # Filter segments based on labels (only 2 and 3)
                                if 'segment_ratings' in segment:
                                    rating = segment['segment_ratings'].get('t1', None)
                                    try:
                                        rating = int(rating)
                                        if rating in [1,2, 3]:  # Only include labels 2 and 3
                                            self.samples.append(segment)
                                    except (ValueError, TypeError):
                                        print(f"Skipping segment in {pkl_file} with invalid rating: {rating}")
                                else:
                                    print(f"Skipping segment in {pkl_file} with missing 'segment_ratings'")

    def __len__(self):
        return len(self.samples)

    def load_bboxes(self, video_id):
        bbox_file = os.path.join(self.bbox_dir, f"{video_id}_bboxes.pkl")
        if os.path.exists(bbox_file):
            with open(bbox_file, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"Warning: Bounding box file not found for {video_id}, using full frames")
            return None

    def crop_to_person(self, frame, bbox, padding=20):
        if isinstance(frame, torch.Tensor):
            frame = frame.permute(1, 2, 0).numpy()  # Convert to (height, width, channels)
        if isinstance(frame, np.ndarray):
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame_rgb = frame  # Assume BGR or RGB, no conversion needed
            else:
                frame_rgb = np.array(frame)
        else:
            frame_rgb = np.array(frame)

        if frame_rgb.dtype != np.uint8:
            frame_rgb = (frame_rgb * 255).astype(np.uint8) if frame_rgb.max() <= 1.0 else frame_rgb.astype(np.uint8)

        if bbox is None:
            return frame_rgb

        x1, y1, x2, y2 = map(int, bbox)
        # Ensure coordinates are valid
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame_rgb.shape[1], x2 + padding)
        y2 = min(frame_rgb.shape[0], y2 + padding)

        # Check if the crop is valid
        if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0:
            print(f"Warning: Invalid bounding box {bbox} for frame shape {frame_rgb.shape}, using full frame")
            return frame_rgb

        cropped = frame_rgb[y1:y2, x1:x2]
        return cropped

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = sample['frames']
        processed_frames = []
        raw_frames_for_viz = []

        video_id = (f"patient_{sample['patient_id']}_task_{sample['activity_id']}_"
                    f"{sample['CameraId']}_seg_{sample['segment_id']}")
        bboxes = self.load_bboxes(video_id) if self.bbox_dir else [None] * len(frames)

        for frame, bbox in zip(frames, bboxes):
            # Crop frame using bounding box
            cropped_frame = self.crop_to_person(frame, bbox)
            # Check frame dimensions
            if cropped_frame.shape[0] == 0 or cropped_frame.shape[1] == 0:
                print(f"Warning: Invalid cropped frame dimensions {cropped_frame.shape} for sample {idx}, skipping frame")
                continue

            raw_frames_for_viz.append(cropped_frame.copy())  # Store cropped frame for visualization
            pil_img = Image.fromarray(cropped_frame)
            if self.transform:
                pil_img = self.transform(pil_img)
            processed_frames.append(pil_img)

        min_frames = 16
        if len(processed_frames) == 0:
            video_tensor = torch.zeros((3, min_frames, 224, 224))
            print(f"Warning: No valid frames for sample {idx}, using dummy tensor")
            return video_tensor, video_id, [np.zeros((224, 224, 3), dtype=np.uint8)] * min_frames
        elif len(processed_frames) < min_frames:
            last_frame = processed_frames[-1]
            last_raw = raw_frames_for_viz[-1]
            processed_frames.extend([last_frame] * (min_frames - len(processed_frames)))
            raw_frames_for_viz.extend([last_raw] * (min_frames - len(raw_frames_for_viz)))
            video_tensor = torch.stack(processed_frames, dim=1)
        else:
            video_tensor = torch.stack(processed_frames, dim=1)

        print(f"Sample {idx} - video_tensor shape: {video_tensor.shape}, raw_frames_for_viz[0] shape: {raw_frames_for_viz[0].shape if raw_frames_for_viz else 'None'}")
        return video_tensor, video_id, raw_frames_for_viz

# -------------------------
# 2) Feature Extraction with Grad-CAM Visualization
# -------------------------
def extract_features_from_pickle(
    pickle_dir='path/to/pickle/files',
    save_dir='D:/i3d_features',
    checkpoint_path='D:/i3d_checkpoints/rgb_imagenet.pt',
    bbox_dir=None,
    layer_name='Mixed_4d',
    threshold=0.1
):
    os.makedirs(save_dir, exist_ok=True)

    spatial_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = PickleVideoDataset(pickle_dir=pickle_dir, transform=spatial_transform, bbox_dir=bbox_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    i3d = InceptionI3d(num_classes=3, in_channels=3)  # Using 3 classes as per your original code
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    i3d.load_state_dict(state_dict)
    i3d.to(device)
    i3d.eval()

    def extract_features_at_layer(model, x, target_layer):
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input, got {x.shape}")
        if x.shape[2] == 0:
            raise ValueError(f"Temporal dimension is 0 in input {x.shape}")

        print(f"Input shape to model: {x.shape}")
        features = None
        def hook_fn(module, input, output):
            nonlocal features
            features = output

        handle = model._modules[target_layer].register_forward_hook(hook_fn)
        try:
            logits = model(x)
            if logits.ndim == 3:
                logits = logits.mean(dim=2)
            handle.remove()
            return features, logits
        except Exception as e:
            handle.remove()
            raise e

    def compute_gradcam(features, logits, target_class=None):
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        features.retain_grad()
        
        i3d.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        gradients = features.grad
        weights = gradients.mean(dim=[2, 3, 4], keepdim=True)
        gradcam = (weights * features).sum(dim=1, keepdim=True)
        gradcam = F.relu(gradcam)
        
        return gradcam.squeeze(1)

    def visualize_heatmap(frame, heatmap, title, save_path, threshold):
        # Normalize heatmap
        heatmap = heatmap - heatmap.min()
        heatmap_max = heatmap.max()
        if heatmap_max > 0:
            heatmap = heatmap / (heatmap_max + 1e-8)        
        heatmap = np.where(heatmap > threshold, heatmap, 0)
        heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        # Create subplot: frame on the left, heatmap on the right
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        # Plot frame (no bounding box)
        ax1.imshow(frame_rgb)
        ax1.set_title("Raw Frame (Cropped)")
        ax1.axis('off')

        # Plot heatmap
        ax2.imshow(heatmap_resized, cmap='viridis')
        ax2.set_title("Grad-CAM Heatmap")
        ax2.axis('off')

        # Add a main title for the subplot
        plt.suptitle(title, fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    for video_tensor, video_id, raw_frames in dataloader:
        print(f"Dataloader output - video_tensor shape: {video_tensor.shape}, raw_frames length: {len(raw_frames[0])}")
        if video_tensor.shape[2] == 0:
            print(f"Skipping {video_id[0]} due to empty temporal dimension")
            continue

        video_tensor = video_tensor.to(device)
        video_tensor.requires_grad_(True)
        try:
            features_5d, logits = extract_features_at_layer(i3d, video_tensor, layer_name)
        except RuntimeError as e:
            print(f"Error processing {video_id[0]}: {e}")
            continue

        # Compute Grad-CAM heatmap
        gradcam_heatmap = compute_gradcam(features_5d, logits)

        # Visualization for all frames
        raw_frames_list = [np.array(f) for f in raw_frames]  # raw_frames is batched
        num_frames = min(gradcam_heatmap.shape[1], len(raw_frames_list))

        print(f"Num frames to visualize: {num_frames}")
        for frame_idx in range(num_frames):
            heatmap = gradcam_heatmap[0, frame_idx].cpu().detach().numpy()
            raw_frame = raw_frames_list[frame_idx]
            viz_title = f"{video_id[0]} - {layer_name} Grad-CAM - Frame {frame_idx}"
            viz_save_path = os.path.join(save_dir, f"{video_id[0]}_{layer_name}_gradcam_frame_{frame_idx}.png")
            visualize_heatmap(np.squeeze(raw_frame), heatmap, viz_title, viz_save_path, threshold)

        # Save features
        features_1d = features_5d.mean(dim=[2, 3, 4]).squeeze(0).cpu().detach().numpy()
        out_path = os.path.join(save_dir, f"{video_id[0]}_{layer_name}.npy")
        np.save(out_path, features_1d)
        print(f"Saved features for {video_id[0]} from {layer_name} to {out_path}")
        print(f"Visualizations saved for {num_frames} frames in {save_dir}")

# -------------------------
# 3) Execution
# -------------------------
if __name__ == '__main__':
    extract_features_from_pickle(
        pickle_dir='D:/pickle_dir',
        save_dir='D:/i3d_features',
        checkpoint_path='D:/Github/Multi_view-automatic-assessment/i3d_finetuned_arat_best.pt',
        bbox_dir='D:/Github/Multi_view-automatic-assessment/bboxes',
        layer_name='Mixed_5c',
        threshold=0.1
    )