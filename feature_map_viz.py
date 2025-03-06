import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def show_spatiotemporal_heatmap(features_5d):
    """
    Displays an animated heatmap of the I3D feature map over time.
    
    Args:
        features_5d (torch.Tensor): 
            A feature tensor of shape (1, 1024, T_out, H_out, W_out)
            as returned by i3d.extract_features(...). 
            If you've already removed the batch dimension, you can pass 
            shape (1024, T_out, H_out, W_out) instead—just adjust accordingly.
    
    Note:
        1) We average across the 1024 channels to get shape [T_out, H_out, W_out].
        2) Then we iterate through T_out frames, creating a heatmap per time step.
        3) This function uses plt.show() to display an interactive animation 
           (best run in a Jupyter notebook or an interactive Python session).
    """
    # If shape is (1, 1024, T_out, H_out, W_out), remove the batch dimension.
    if features_5d.dim() == 5 and features_5d.shape[0] == 1:
        features_5d = features_5d.squeeze(0)  # => (1024, T_out, H_out, W_out)
    
    # Average across channels => shape (T_out, H_out, W_out)
    # (If channels-first is [C, T, H, W], ensure we know which dim is time vs channels.)
    # Here it's typically (1024, T_out, H_out, W_out), so dim=0 is channels.
    feat_avg_channels = features_5d.mean(dim=0)  # => (T_out, H_out, W_out)

    # Move to CPU and convert to numpy if on GPU
    if feat_avg_channels.is_cuda:
        feat_avg_channels = feat_avg_channels.cpu()
    feat_avg_channels = feat_avg_channels.numpy()  # shape (T_out, H_out, W_out)

    # Prepare figure
    fig, ax = plt.subplots()
    # We'll store each frame's image in this list for animation
    ims = []

    # Iterate over time dimension
    T_out = feat_avg_channels.shape[0]
    for t in range(T_out):
        slice_2d = feat_avg_channels[t]
        # Create a heatmap image. Note 'extent' or 'origin' can be adjusted if needed
        im = ax.imshow(slice_2d, animated=True, cmap='viridis')
        # For animation, each frame is a list of artists
        ims.append([im])

    # Create the animation
    ani = animation.ArtistAnimation(fig, ims, interval=300, blit=True,
                                    repeat_delay=1000)
    ax.set_title("I3D Feature Map Activation Over Time")
    plt.colorbar(ims[0][0], ax=ax)
    plt.show()
