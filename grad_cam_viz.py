import os
import pickle
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pytorch_i3d import InceptionI3d
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

# ------------------
# 1) Custom Dataset for Pickle Files
# ------------------
class PickleVideoDataset(Dataset):
    def __init__(self, pickle_dir, transform=None):
        self.pickle_dir = pickle_dir
        self.transform = transform
        self.pickle_files = glob.glob(os.path.join(pickle_dir, '*.pkl'))
        self.samples = []

        for pkl_file in self.pickle_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                for camera_id in data:
                    if camera_id == 'cam3':
                        for segments_group in data[camera_id]:
                            for segment in segments_group:
                                self.samples.append(segment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = sample['frames']
        processed_frames = []

        for frame in frames:
            if isinstance(frame, np.ndarray):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
            else:
                pil_img = frame
            if self.transform:
                pil_img = self.transform(pil_img)
            processed_frames.append(pil_img)

        min_frames = 16
        if len(processed_frames) == 0:
            video_tensor = torch.zeros((3, min_frames, 224, 224))
            print(f"Warning: No frames for sample {idx}, using dummy tensor")
        elif len(processed_frames) < min_frames:
            processed_frames.extend([processed_frames[-1]] * (min_frames - len(processed_frames)))
            video_tensor = torch.stack(processed_frames, dim=1)
        else:
            video_tensor = torch.stack(processed_frames, dim=1)

        video_id = (f"patient_{sample['patient_id']}_task_{sample['activity_id']}_"
                    f"{sample['CameraId']}_seg_{sample['segment_id']}")

        print(f"Sample {idx} - video_tensor shape: {video_tensor.shape}")
        return video_tensor, video_id, frames

# -------------------------
# 2) Feature Extraction with Grad-CAM Visualization
# -------------------------
def extract_features_from_pickle(
    pickle_dir='path/to/pickle/files',
    save_dir='D:/i3d_features',
    checkpoint_path='D:/i3d_checkpoints/rgb_imagenet.pt',
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

    dataset = PickleVideoDataset(pickle_dir=pickle_dir, transform=spatial_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    i3d = InceptionI3d(num_classes=3, in_channels=3)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    i3d.load_state_dict(state_dict)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
            if logits.ndim == 3:  # [B, C, T]
                logits = logits.mean(dim=2)  # [B, C]
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

        gradients = features.grad  # [1, C, T, H, W]
        weights = gradients.mean(dim=[2, 3, 4], keepdim=True)  # [1, C, 1, 1, 1]
        gradcam = (weights * features).sum(dim=1, keepdim=True)  # [1, 1, T, H, W]
        gradcam = F.relu(gradcam)
        
        return gradcam.squeeze(1)  # [1, T, H, W]

    def visualize_heatmap(frame, heatmap, title, save_path, threshold):
        heatmap = heatmap - heatmap.min()
        heatmap_max = heatmap.max()
        if heatmap_max > 0:
            heatmap = heatmap / (heatmap_max + 1e-8)
        
        heatmap = np.where(heatmap > threshold, heatmap, 0)
        
        print(f"{title} - Heatmap Min: {heatmap.min()}, Max: {heatmap.max()}")
        
        heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[-1] == 3 else frame
        plt.figure(figsize=(6, 4))
        plt.imshow(frame_rgb)
        plt.imshow(heatmap_resized, cmap='viridis', alpha=0.5)
        plt.title(title)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    for video_tensor, video_id, raw_frames in dataloader:
        print(f"Dataloader output - video_tensor shape: {video_tensor.shape}")
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
        gradcam_heatmap = compute_gradcam(features_5d, logits)  # [1, T_out, H_out, W_out]

        # Visualization for all frames
        raw_frames_list = [np.array(f) for f in raw_frames]
        num_frames = min(gradcam_heatmap.shape[1], len(raw_frames_list))

        for frame_idx in range(num_frames):
            heatmap = gradcam_heatmap[0, frame_idx].cpu().detach().numpy()  # Detach before converting to numpy
            raw_frame = raw_frames_list[frame_idx]
            viz_title = f"{video_id[0]} - {layer_name} Grad-CAM - Frame {frame_idx}"
            viz_save_path = os.path.join(save_dir, f"{video_id[0]}_{layer_name}_gradcam_frame_{frame_idx}.png")
            visualize_heatmap(raw_frame, heatmap, viz_title, viz_save_path, threshold)

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
        pickle_dir='D:/Github/Multi_view-automatic-assessment',
        save_dir='D:/i3d_features',
        checkpoint_path='D:/Github/Multi_view-automatic-assessment/i3d_finetuned_arat.pt',
        layer_name='Mixed_4d',
        threshold=0.1
    )