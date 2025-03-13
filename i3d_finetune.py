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
        label = int(segment['segment_ratings']['t1']) - 1

        processed_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            if torch.isnan(frame).any() or torch.isinf(frame).any():
                raise ValueError(f"NaN or Inf detected in frame for segment {idx}")
            processed_frames.append(frame)

        if len(processed_frames) == 0:
            video_tensor = torch.zeros((3, 1, 224, 224))
        else:
            video_tensor = torch.stack(processed_frames, dim=1)

        return video_tensor, label

# ------------------------
# 2. Fine-tuning Function with Adjustments
# ------------------------
def train_i3d(
    pickle_dir,
    checkpoint_path,
    num_epochs=30,  # Increased epochs
    batch_size=2,
    lr=5e-5,  # Reduced peak learning rate
    num_classes=3,
    val_split=0.2,
    weight_decay=1e-3,  # Reduced weight decay
    patience=8  # Increased patience
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

    # Load samples, filtering out invalid labels
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
                                rating = segment['segment_ratings'].get('t1', None)
                                try:
                                    rating = int(rating)
                                    if rating not in [1, 2, 3]:
                                        print(f"Skipping segment in {pkl_file} with invalid rating: {rating}")
                                        continue
                                except (ValueError, TypeError):
                                    print(f"Skipping segment in {pkl_file} with invalid rating: {rating}")
                                    continue
                                all_samples.append(segment)
                            else:
                                print(f"Skipping segment in {pkl_file} with missing 'segment_ratings'")

    if len(all_samples) == 0:
        raise ValueError("No valid samples found after filtering. Check your dataset for valid ratings.")

    # Compute class distribution
    labels = [int(sample['segment_ratings']['t1']) - 1 for sample in all_samples]
    class_counts = Counter(labels)
    print("Class distribution (0, 1, 2):", class_counts)
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor([total_samples / (num_classes * class_counts[i]) for i in range(num_classes)], dtype=torch.float).to(device)
    print("Class weights:", class_weights)

    train_samples, val_samples = train_test_split(all_samples, test_size=val_split, random_state=42)

    train_dataset = ARATSegmentDataset(train_samples, transform=train_transform)
    val_dataset = ARATSegmentDataset(val_samples, transform=val_transform)

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
        nn.Dropout(p=0.2),  # Reduced dropout to 20%
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

    # Freeze early layers (up to Mixed_4f)
    freeze_until = 'Mixed_4f'
    freeze = True
    for name, param in i3d.named_parameters():
        if freeze_until in name:
            freeze = False
        if freeze:
            param.requires_grad = False
            print(f"Froze layer: {name}")

    i3d.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, i3d.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)

    # Warmup phase: linearly increase LR from 1e-5 to lr over 5 epochs
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

        # Warmup learning rate
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

        # Step scheduler after warmup
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
        num_epochs=30,
        batch_size=2,
        lr=5e-5,
        num_classes=3,
        val_split=0.2,
        weight_decay=1e-3,
        patience=8
    )
