import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from feature_map_viz import show_spatiotemporal_heatmap
# This assumes you have pytorch_i3d.py in the same directory
from pytorch_i3d import InceptionI3d

# ------------------
# 1) Custom dataset
# ------------------
class MyVideoDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        """
        video_dir: directory containing MP4 files
        transform: spatial transforms (e.g., CenterCrop, Normalize)
        """
        self.video_dir = video_dir
        self.transform = transform
        
        # Gather a list of all .mp4 files in video_dir
        self.video_paths = glob.glob(os.path.join(video_dir, '*.mp4'))
        self.video_paths.sort()
        
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Read all frames with OpenCV
        frames = self._load_video_frames(video_path)
        
        # Apply transform frame by frame
        processed_frames = []
        for frame in frames:
            # Convert BGR to RGB if needed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL
            pil_img = Image.fromarray(frame_rgb)
            # Apply transform pipeline
            if self.transform:
                pil_img = self.transform(pil_img)
            processed_frames.append(pil_img)
        
        # Stack into a tensor of shape (3, T, H, W)
        # Each pil_img is a torch tensor of shape (3, H, W)
        if len(processed_frames) == 0:
            # If somehow no frames, create a dummy
            video_tensor = torch.zeros((3, 1, 224, 224))
        else:
            video_tensor = torch.stack(processed_frames, dim=1)  # shape: (3, T, H, W)
        
        return video_tensor, video_name

    def _load_video_frames(self, path):
        """
        Loads all frames from the given video path using OpenCV.
        Returns a list of numpy arrays (BGR), each shape (H,W,3).
        """
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

# -------------------------
# 2) Feature Extraction
# -------------------------
def extract_features(
    video_dir='D:\\try_videos_i3d',
    save_dir='D:\\i3d_features',
    checkpoint_path='D:\\i3d_checkpoints\\rgb_imagenet.pt'
):
    """
    Main function to:
    1. Load a pretrained I3D (RGB)
    2. Create a DataLoader for your MP4 videos
    3. Extract features per video and save as .npy
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 2A) Define spatial transforms (similar to I3D's training setup)
    # We'll do: Resize -> CenterCrop -> ToTensor -> Normalize
    # You can adjust as needed
    spatial_transform = transforms.Compose([
        transforms.Resize((256, 256)),     # keep aspect ratio or not
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    # 2B) Create dataset and dataloader
    dataset = MyVideoDataset(video_dir=video_dir, transform=spatial_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 2C) Load the I3D model
    #    We'll use a 400-class model with in_channels=3 for RGB
    i3d = InceptionI3d(num_classes=400, in_channels=3)
    # Replace final layer with 400 again, or whichever number, 
    # because we must match the checkpoint's architecture.
    i3d.replace_logits(400)
    
    # Load the checkpoint
    state_dict = torch.load(checkpoint_path)
    i3d.load_state_dict(state_dict)
    
    # Move to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    i3d.to(device)
    i3d.eval()  # inference mode
    
    with torch.no_grad():
        for video_tensor, video_name in dataloader:
            # video_tensor: shape (1, 3, T, H, W)
            video_tensor = video_tensor.to(device)
            
            b, c, t, h, w = video_tensor.shape
            
            # 2D) Extract spatiotemporal feature map
            #     This returns shape [batch_size, 1024, T_out, H_out, W_out]
            features_5d = i3d.extract_features(video_tensor)
            show_spatiotemporal_heatmap(features_5d)
            # (Optional) If you want a single 1024-d embedding, do global avg pool:
            # shape => [b, 1024, T_out, H_out, W_out] -> [b, 1024]
            features_5d = features_5d.mean(dim=[2, 3, 4])
            # shape => (1, 1024)
            features_1d = features_5d.squeeze(0).cpu().numpy()  # shape (1024,)
            
            # 2E) Save to .npy
            out_path = os.path.join(save_dir, f"{video_name}.npy")
            np.save(out_path, features_1d)
            print(f"Saved features for {video_name} -> {out_path}")

if __name__ == '__main__':
    # Example usage - adjust paths as needed
    extract_features(
        video_dir='D:\\try_videos_i3d',
        save_dir='D:\\i3d_features',
        checkpoint_path='D:\\Github\\Multi_view-automatic-assessment\\rgb_imagenet.pt'
    )
