import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torchvision import transforms
from pytorch_i3d import InceptionI3d  # Make sure you have pytorch_i3d.py in the same folder

########################################
# 1) Helper to load a video (frames)
########################################
def load_video_frames(video_path):
    """
    Loads all frames from the given video path using OpenCV.
    Returns a list of np arrays in BGR format (H,W,3).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

########################################
# 2) Transform & pack frames for I3D
########################################
def prepare_i3d_input(frames_bgr, transform):
    """
    frames_bgr: list of BGR frames, each shape (H,W,3)
    transform:  a torchvision transform pipeline that expects a PIL image
    Returns a torch.Tensor [1, 3, T, H', W'], ready for i3d.extract_features
    """
    # Convert each frame BGR->RGB -> PIL -> apply transform -> stack
    processed = []
    for f in frames_bgr:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        # Convert to PIL
        # (We assume the transform does Resize, CenterCrop(224), ToTensor, etc.)
        pil_img = transforms.functional.to_pil_image(rgb)
        tensor_img = transform(pil_img)  # shape [3, H', W']
        processed.append(tensor_img.unsqueeze(1))  # shape [3, 1, H', W']

    # Concatenate along time dimension => shape [3, T, H', W']
    if len(processed) == 0:
        # fallback if no frames
        video_tensor = torch.zeros((3,1,224,224))
    else:
        video_tensor = torch.cat(processed, dim=1)

    # Add batch dimension => [1, 3, T, H', W']
    return video_tensor.unsqueeze(0)

########################################
# 3) Upsample & Overlay Feature Maps
########################################
def overlay_features_on_frames(frames_bgr, feat_5d, alpha=0.5):
    """
    frames_bgr: list of original frames (BGR), each shape (H, W, 3)
    feat_5d: torch.Tensor of shape [1, 1024, T_out, H_out, W_out] from i3d.extract_features
             (no global pooling).
    alpha: blending factor for overlay.

    1) Averages channels => shape (1,1,T_out,H_out,W_out).
    2) Upsamples to (1,1,T,H,W) => match #frames and resolution.
    3) For each frame, overlay heatmap onto the original frame.

    Returns: a list of BGR frames with heatmap overlay.
    """
    device = feat_5d.device
    # Step 1: average across the 1024 channels => shape [1, 1, T_out, H_out, W_out]
    feat_5d_avg = feat_5d.mean(dim=1, keepdim=True)  # (batch=1, ch=1, T_out, H_out, W_out)

    T = len(frames_bgr)
    H, W, _ = frames_bgr[0].shape

    # Step 2: Upsample
    # We want to go from (1,1,T_out,H_out,W_out) -> (1,1,T,H,W).
    # We'll treat T_out as the 'depth' dimension, so use F.interpolate with mode='trilinear'.
    # scale_factor = (T / T_out, H / H_out, W / W_out).
    _, _, T_out, H_out, W_out = feat_5d_avg.shape
    scale_factor = (float(T)/T_out, float(H)/H_out, float(W)/W_out)

    feat_upsampled = F.interpolate(
        feat_5d_avg, 
        scale_factor=scale_factor,
        mode='trilinear', 
        align_corners=False
    )  # => [1,1,T,H,W]
    
    # Remove batch & channel dims => shape [T, H, W]
    feat_upsampled = feat_upsampled[0,0]  # => shape (T, H, W)

    # Move to CPU, convert to numpy
    feat_upsampled = feat_upsampled.cpu().numpy()  # shape (T, H, W)

    # Optional: normalize each frame's heatmap for better contrast
    # We'll do min-max per frame
    overlayed_frames = []
    for t in range(T):
        heatmap = feat_upsampled[t]
        # min-max scale => [0,1]
        hm_min, hm_max = heatmap.min(), heatmap.max()
        if hm_max > hm_min:  # avoid divide by zero
            heatmap_norm = (heatmap - hm_min) / (hm_max - hm_min)
        else:
            heatmap_norm = heatmap - hm_min  # all zeros

        # Convert heatmap to a 3-channel colored image using a colormap
        heatmap_color = cv2.applyColorMap(
            (heatmap_norm*255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)  
        # ^ if you prefer BGR for final, keep as is. 
        # We'll go to RGB so we can overlay with an RGB representation of the frame.

        # Convert original frame BGR->RGB
        frame_rgb = cv2.cvtColor(frames_bgr[t], cv2.COLOR_BGR2RGB)

        # Blend: out = alpha*heatmap + (1-alpha)*original
        blended = cv2.addWeighted(
            heatmap_color, alpha,
            frame_rgb, 1-alpha, 
            gamma=0
        )

        # Convert back to BGR if you want standard OpenCV format
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        overlayed_frames.append(blended_bgr)
    
    return overlayed_frames

########################################
# 4) Animate & Display in Matplotlib
########################################
def animate_frames(frames_bgr, fps=10):
    """
    Takes a list of BGR frames and displays an animation via matplotlib.
    """
    fig, ax = plt.subplots()
    ims = []
    for frame in frames_bgr:
        # Convert BGR->RGB for display in matplotlib
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = ax.imshow(rgb, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True, repeat_delay=1000)
    plt.axis('off')
    plt.show()

########################################
# 5) Putting it all together (Example)
########################################
def main():
    # Paths
    video_path = r"D:\try_videos_i3d\ARAT_01_right_Impaired_cam1_activity1.mp4"
    checkpoint_path = r"D:\Github\Multi_view-automatic-assessment\rgb_imagenet.pt"

    # 5A) Load frames
    frames_bgr = load_video_frames(video_path)
    if not frames_bgr:
        print("No frames found in video!")
        return
    
    # 5B) Define transforms & prepare input for I3D
    spatial_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    video_tensor = prepare_i3d_input(frames_bgr, spatial_transform)  # => [1,3,T,224,224]

    # 5C) Load I3D model
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(400)  # must match checkpoint shape
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    i3d.load_state_dict(state_dict)
    i3d.eval()

    # If you have CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    i3d.to(device)
    video_tensor = video_tensor.to(device)

    # 5D) Extract spatiotemporal features
    with torch.no_grad():
        features_5d = i3d.extract_features(video_tensor)  # shape [1, 1024, T_out, H_out, W_out]
    
    # 5E) Overlay and visualize
    overlayed_frames = overlay_features_on_frames(frames_bgr, features_5d, alpha=0.5)
    animate_frames(overlayed_frames, fps=10)

if __name__ == "__main__":
    main()
