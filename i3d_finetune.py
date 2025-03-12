import os
import pickle
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn, optim
from pytorch_i3d import InceptionI3d
from PIL import Image
from sklearn.model_selection import train_test_split

# ------------------------
# 1. Dataset with Segment Scores
# ------------------------
class ARATSegmentDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        segment = self.samples[idx]
        frames = segment['frames']
        label = int(segment['segment_ratings']['t1']) - 1  # Adjust score from [1,2,3] to [0,1,2]

        processed_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            processed_frames.append(frame)

        if len(processed_frames) == 0:
            video_tensor = torch.zeros((3, 1, 224, 224))
        else:
            video_tensor = torch.stack(processed_frames, dim=1)  # [3, T, H, W]

        return video_tensor, label

# ------------------------
# 2. Fine-tuning Function with Validation and Anti-Overfitting Measures
# ------------------------
def train_i3d(
    pickle_dir,
    checkpoint_path,
    num_epochs=15,
    batch_size=2,
    lr=1e-4,
    num_classes=3,
    val_split=0.2,
    weight_decay=1e-4,  # Added for L2 regularization
    patience=3          # For early stopping
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop and resize
        transforms.RandomHorizontalFlip(),                    # Random flip
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load all samples from pickle files
    pickle_files = glob.glob(os.path.join(pickle_dir, '*.pkl'))
    all_samples = []
    for pkl_file in pickle_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            for camera_id in data:
                if camera_id == 'cam3':
                    for segment_group in data[camera_id]:
                        for segment in segment_group:
                            if 'segment_ratings' in segment:
                                all_samples.append(segment)

    # Split into training and validation sets
    train_samples, val_samples = train_test_split(all_samples, test_size=val_split, random_state=42)

    # Create datasets
    train_dataset = ARATSegmentDataset(train_samples, transform=train_transform)
    val_dataset = ARATSegmentDataset(val_samples, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load and configure the I3D model
    i3d = InceptionI3d(num_classes=400, in_channels=3)
    i3d.load_state_dict(torch.load(checkpoint_path), strict=False)
    i3d.replace_logits(num_classes)  # Adjust for 3 classes
    # Add dropout before the final layer (requires custom modification in pytorch_i3d.py)
    i3d.to(device)

    for param in i3d.parameters():
        param.requires_grad = True

    # Define loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(i3d.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Training and validation loop with early stopping
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Training phase
        i3d.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for video_tensor, label in train_loader:
            video_tensor = video_tensor.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            out = i3d(video_tensor)
            if out.ndim == 3:
                out = out.mean(2)  # [B, C, T] -> [B, C]
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(out, 1)
            train_total += label.size(0)
            train_correct += (predicted == label).sum().item()

        train_acc = train_correct / train_total
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")

        # Validation phase
        i3d.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for video_tensor, label in val_loader:
                video_tensor = video_tensor.to(device)
                label = label.to(device)

                out = i3d(video_tensor)
                if out.ndim == 3:
                    out = out.mean(2)  # [B, C, T] -> [B, C]
                loss = criterion(out, label)

                val_loss += loss.item()
                _, predicted = torch.max(out, 1)
                val_total += label.size(0)
                val_correct += (predicted == label).sum().item()

        val_acc = val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)
        print(f"Validation Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss_avg)

        # Save best model based on validation loss (could use accuracy instead)
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_val_acc = val_acc
            torch.save(i3d.state_dict(), os.path.join(pickle_dir, 'i3d_finetuned_arat_best.pt'))
            print(f"Saved best model with Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Save final model
    torch.save(i3d.state_dict(), os.path.join(pickle_dir, 'i3d_finetuned_arat_final.pt'))
    print("Training complete. Final model saved as 'i3d_finetuned_arat_final.pt'")

# ------------------------
# 3. Run the Training
# ------------------------
if __name__ == '__main__':
    train_i3d(
        pickle_dir='D:/Github/Multi_view-automatic-assessment',
        checkpoint_path='D:/Github/pytorch-i3d/models/rgb_imagenet.pt',
        num_epochs=15,
        batch_size=2,
        lr=1e-4,
        num_classes=3,
        val_split=0.2,
        weight_decay=1e-4,  # L2 regularization strength
        patience=3          # Early stopping patience
    )