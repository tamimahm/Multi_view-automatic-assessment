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
from collections import Counter
import cv2

# ------------------------
# 1. Dataset with Segment Scores and Precomputed Bounding Boxes
# ------------------------
class ARATSegmentDataset(Dataset):
    def __init__(self, samples, transform=None, bbox_dir=None):
        self.samples = samples
        self.transform = transform
        self.bbox_dir = bbox_dir

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
        if isinstance(frame, np.ndarray):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = np.array(frame)

        if bbox is None:
            return Image.fromarray(frame_rgb)

        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame_rgb.shape[1], x2 + padding)
        y2 = min(frame_rgb.shape[0], y2 + padding)

        cropped = frame_rgb[y1:y2, x1:x2]
        return Image.fromarray(cropped)

    def __getitem__(self, idx):
        segment = self.samples[idx]
        frames = segment['frames']
        label = int(segment['segment_ratings']['t1']) - 1
        video_id = (f"patient_{segment['patient_id']}_task_{segment['activity_id']}_"
                    f"{segment['CameraId']}_seg_{segment['segment_id']}")

        bboxes = self.load_bboxes(video_id) if self.bbox_dir else [None] * len(frames)

        processed_frames = []
        for frame, bbox in zip(frames, bboxes):
            cropped_frame = self.crop_to_person(frame, bbox)
            if self.transform:
                frame = self.transform(cropped_frame)
            if torch.isnan(frame).any() or torch.isinf(frame).any():
                raise ValueError(f"NaN or Inf detected in frame for segment {idx}")
            processed_frames.append(frame)

        if len(processed_frames) == 0:
            video_tensor = torch.zeros((3, 1, 224, 224))
        else:
            video_tensor = torch.stack(processed_frames, dim=1)

        return video_tensor, label

# ------------------------
# 2. Fine-tuning Function
# ------------------------
def train_i3d(
    pickle_dir,
    checkpoint_path,
    bbox_dir,
    num_epochs=30,
    batch_size=2,
    lr=5e-5,
    num_classes=3,
    val_split=0.2,
    weight_decay=1e-3,
    patience=8
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load samples and count all labels
    pickle_files = glob.glob(os.path.join(pickle_dir, '*.pkl'))
    all_samples = []
    all_labels = []  # To count all labels, including unexpected ones

    for pkl_file in pickle_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            for camera_id in data:
                if camera_id == 'cam3':
                    for segment_group in data[camera_id]:
                        for segment in segment_group:
                            if 'segment_ratings' in segment:
                                rating = segment['segment_ratings'].get('t1', None)
                                try:
                                    rating = int(rating)
                                    all_labels.append(rating)  # Count all labels
                                    # Only include samples with labels 0, 1, 2 for training
                                    if rating in [1, 2, 3]:  # Original ratings 1, 2, 3
                                        all_samples.append(segment)
                                    else:
                                        print(f"Skipping segment in {pkl_file} with unexpected label: {rating}")
                                except (ValueError, TypeError):
                                    print(f"Skipping segment in {pkl_file} with invalid rating: {rating}")
                            else:
                                print(f"Skipping segment in {pkl_file} with missing 'segment_ratings'")

    if len(all_samples) == 0:
        raise ValueError("No valid samples found. Check your dataset.")

    # Compute distribution of all labels
    label_counts = Counter(all_labels)
    print("Label distribution in the dataset:")
    for label, count in sorted(label_counts.items()):
        print(f"Label {label}: {count} occurrences")

    # Compute class weights only for valid labels (0, 1, 2 after -1 adjustment)
    valid_labels = [int(sample['segment_ratings']['t1']) - 1 for sample in all_samples]  # Adjust to 0, 1, 2
    class_counts = Counter(valid_labels)
    print("Valid label distribution (0, 1, 2) used for training:")
    for label in [0, 1, 2]:
        count = class_counts.get(label, 0)
        print(f"Valid label {label}: {count} occurrences")
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor([total_samples / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)], dtype=torch.float).to(device)
    print("Class weights for training (for labels 0, 1, 2):", class_weights)

    train_samples, val_samples = train_test_split(all_samples, test_size=val_split, random_state=42)

    train_dataset = ARATSegmentDataset(train_samples, transform=train_transform, bbox_dir=bbox_dir)
    val_dataset = ARATSegmentDataset(val_samples, transform=val_transform, bbox_dir=bbox_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize I3D model
    i3d = InceptionI3d(num_classes=400, in_channels=3)
    i3d.load_state_dict(torch.load(checkpoint_path), strict=False)
    
    # Replace logits to match the number of classes
    i3d.replace_logits(num_classes)

    # Add dropout to the logits layer
    original_conv3d = i3d.logits.conv3d
    i3d.logits.conv3d = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Conv3d(
            in_channels=original_conv3d.in_channels,
            out_channels=original_conv3d.out_channels,
            kernel_size=original_conv3d.kernel_size,
            stride=original_conv3d.stride,
            padding=original_conv3d.padding,
            bias=original_conv3d.bias is not None
        )
    )
    i3d.logits.conv3d[1].weight.data.copy_(original_conv3d.weight.data)
    if original_conv3d.bias is not None:
        i3d.logits.conv3d[1].bias.data.copy_(original_conv3d.bias.data)

    # Freeze layers up to 'Mixed_4f' and update layers after
    # Since 'features' is not an attribute, iterate over named parameters
    freeze_until = 'Mixed_4f'
    freeze = True
    for name, param in i3d.named_parameters():
        if freeze_until in name:
            freeze = False
        if freeze:
            param.requires_grad = False
            print(f"Froze parameter: {name}")
        else:
            param.requires_grad = True
            print(f"Unfroze parameter: {name}")

    # Explicitly ensure logits is unfrozen
    for param in i3d.logits.parameters():
        param.requires_grad = True
        print(f"Unfroze parameter: logits.{param}")

    i3d.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, i3d.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)

    # Warmup phase
    warmup_epochs = 5
    warmup_lr_schedule = np.linspace(1e-5, lr, warmup_epochs)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    completed_epochs = -1

    for epoch in range(num_epochs):
        if completed_epochs >= epoch:
            print(f"Skipping duplicate epoch {epoch+1}")
            continue

        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr_schedule[epoch]
            print(f"Epoch {epoch+1} - Warmup LR: {warmup_lr_schedule[epoch]}")

        i3d.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_idx, (video_tensor, label) in enumerate(train_loader):
            video_tensor = video_tensor.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            out = i3d(video_tensor)
            if out.ndim == 3:
                out = out.mean(2)
            loss = criterion(out, label)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(i3d.parameters(), max_norm=1.0)
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
                    out = out.mean(2)
                loss = criterion(out, label)

                val_loss += loss.item()
                _, predicted = torch.max(out, 1)
                val_total += label.size(0)
                val_correct += (predicted == label).sum().item()

        val_acc = val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)
        print(f"Validation Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_val_acc = val_acc
            torch.save(i3d.state_dict(), os.path.join(pickle_dir, 'i3d_finetuned_arat_best.pt'))
            print(f"Saved best model with Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch >= warmup_epochs:
            scheduler.step(val_loss_avg)

        completed_epochs += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    torch.save(i3d.state_dict(), os.path.join(pickle_dir, 'i3d_finetuned_arat_final.pt'))
    print("Training complete. Final model saved as 'i3d_finetuned_arat_final.pt'")

# ------------------------
# 3. Run the Training
# ------------------------
if __name__ == '__main__':
    train_i3d(
        pickle_dir='D:/pickle_dir',
        checkpoint_path='D:/Github/pytorch-i3d/models/rgb_imagenet.pt',
        bbox_dir='D:/Github/Multi_view-automatic-assessment/bboxes',
        num_epochs=30,
        batch_size=2,
        lr=5e-5,
        num_classes=3,
        val_split=0.2,
        weight_decay=1e-3,
        patience=8
    )