import os
import pickle
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pytorch_i3d import InceptionI3d

# ------------------
# 1) Custom Dataset for Pickle Files
# ------------------
class PickleVideoDataset(Dataset):
    def __init__(self, pickle_dir, transform=None):
        """
        pickle_dir: directory containing pickle files with preprocessed data
        transform: spatial transforms (same as before)
        """
        self.pickle_dir = pickle_dir
        self.transform = transform
        
        # Load all pickle files
        self.pickle_files = glob.glob(os.path.join(pickle_dir, '*.pkl'))
        self.samples = []
        
        # Load and flatten data from all pickle files
        for pkl_file in self.pickle_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                # Flatten camera views and segments
                for camera_id in data:
                    if camera_id=='cam3':
                        for segment_sample in data[camera_id]:
                            self.samples.append(segment_sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get frames from the sample (already pre-extracted)
        frames = sample['frames']
        
        # Process frames (assuming frames are stored as PIL Images or numpy arrays)
        processed_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                # Convert numpy array to PIL Image
                pil_img = Image.fromarray(frame)
            else:
                # Assume it's already a PIL Image
                pil_img = frame
                
            # Apply transform pipeline
            if self.transform:
                pil_img = self.transform(pil_img)
                
            processed_frames.append(pil_img)
        
        # Stack into tensor (3, T, H, W)
        if len(processed_frames) == 0:
            video_tensor = torch.zeros((3, 1, 224, 224))
        else:
            video_tensor = torch.stack(processed_frames, dim=1)
            
        # Create unique identifier
        video_id = (f"patient_{sample['patient_id']}_task_{sample['activity_id']}_"
                    f"cam_{sample['CameraId']}_seg_{sample['segment_id']}")
        
        return video_tensor, video_id

# -------------------------
# 2) Modified Feature Extraction
# -------------------------
def extract_features_from_pickle(
    pickle_dir='path/to/pickle/files',
    save_dir='D:/i3d_features',
    checkpoint_path='D:/i3d_checkpoints/rgb_imagenet.pt'
):
    os.makedirs(save_dir, exist_ok=True)
    
    # Keep the same spatial transforms
    spatial_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset from pickle files
    dataset = PickleVideoDataset(pickle_dir=pickle_dir, transform=spatial_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Load I3D model (same as before)
    i3d = InceptionI3d(num_classes=400, in_channels=3)
    i3d.replace_logits(400)
    i3d.load_state_dict(torch.load(checkpoint_path))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    i3d.to(device)
    i3d.eval()
    
    with torch.no_grad():
        for video_tensor, video_id in dataloader:
            video_tensor = video_tensor.to(device)
            
            # Extract features
            features_5d = i3d.extract_features(video_tensor)
            
            # Pool features (optional: keep spatiotemporal dimensions if needed)
            features_5d = features_5d.mean(dim=[2, 3, 4])
            features_1d = features_5d.squeeze(0).cpu().numpy()
            
            # Save with structured name
            out_path = os.path.join(save_dir, f"{video_id}.npy")
            np.save(out_path, features_1d)
            print(f"Saved features for {video_id}")

# -------------------------
# 3) Execution
# -------------------------
if __name__ == '__main__':
    extract_features_from_pickle(
        pickle_dir='D:/Github/Multi_view-automatic-assessment',
        save_dir='D:/i3d_features',
        checkpoint_path='D:/Github/pytorch-i3d/models/rgb_imagenet.pt'
    )