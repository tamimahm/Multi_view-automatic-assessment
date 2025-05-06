import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
import pickle
import glob
import scipy.io as sio
import pandas as pd
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split


# Global flags
USE_PRETRAINED = 0  # Set to 1 to load a pretrained model
USE_TOP_VIEW = 1  # Set to 1 for top view, 0 for ipsilateral view
SAVE_FEATURES = 0  # Set to 1 to save intermediate features


# Define paths based on your directory structure
OPENPOSE_DIR = "D:/all_ARAT_openpose"
HAND_DIR = "D:/ARAT_2D_joints/all_ARAT_hand"
OBJECT_DIR = "D:/data_res_trident/alternative"
PICKLE_DIR = "D:/pickle_dir"
OUTPUT_DIR = "./output/gnn_transformer"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


class KeypointDataProcessor:
    """Class to load and process keypoints from different sources"""
    
    def __init__(self, top_view=True):
        self.top_view = top_view
        # Define camera mapping based on the view
        self.camera_mapping = {
            True: "cam3",  # Top view uses cam3
            False: None    # Ipsilateral view will be determined from metadata
        }
        
        # Read ipsilateral camera assignments
        self.ipsi_camera_map = self._load_camera_assignments()
        
    def _load_camera_assignments(self):
        """Load camera assignments from CSV file"""
        camera_csv = "D:/Github/Multi_view-automatic-assessment/camera_assignments.csv"
        try:
            camera_df = pd.read_csv(camera_csv)
            # Create a mapping from patient_id to ipsilateral_camera_id
            return dict(zip(camera_df['patient_id'], camera_df['ipsilateral_camera_id']))
        except Exception as e:
            print(f"Error loading camera assignments: {e}")
            return {}
    
    def load_openpose_keypoints(self, patient_id, activity_id, segment_frames):
        """
        Load OpenPose keypoints for the specified patient, activity, and frames
        Returns keypoints for shoulders, elbows, wrists, neck, mid hip
        """
        keypoints_data = []
        
        # Map keypoint indices to names
        keypoint_indices = {
            'neck': 1,
            'right_shoulder': 2,
            'right_elbow': 3,
            'right_wrist': 4,
            'left_shoulder': 5,
            'left_elbow': 6,
            'left_wrist': 7,
            'mid_hip': 8
        }
        
        # Get the camera ID based on the view
        camera_id = self.camera_mapping[self.top_view]
        if not camera_id:
            camera_id = self.ipsi_camera_map.get(patient_id)
            if not camera_id:
                print(f"No ipsilateral camera found for patient {patient_id}")
                return None
        
        # Define path to keypoints for this patient/activity
        keypoints_path = os.path.join(
            OPENPOSE_DIR, 
            f"patient_{patient_id}", 
            f"activity_{activity_id}", 
            "keypoints"
        )
        
        if not os.path.exists(keypoints_path):
            print(f"OpenPose keypoints not found at {keypoints_path}")
            return None
        
        # Process each frame in the segment
        for frame_idx in segment_frames:
            # OpenPose keypoints are stored as JSON files
            json_file = os.path.join(keypoints_path, f"{frame_idx:012d}_{camera_id}_keypoints.json")
            
            if not os.path.exists(json_file):
                # Use a placeholder if the frame is missing
                keypoints_frame = np.zeros((len(keypoint_indices), 3))  # x, y, confidence
                keypoints_data.append(keypoints_frame)
                continue
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract keypoints for the specified indices
                if 'people' in data and len(data['people']) > 0:
                    pose_keypoints = data['people'][0]['pose_keypoints_2d']
                    
                    # Reshape into (x, y, confidence) triples
                    pose_keypoints = np.array(pose_keypoints).reshape(-1, 3)
                    
                    # Extract the required keypoints
                    selected_keypoints = []
                    for name, idx in keypoint_indices.items():
                        if idx < len(pose_keypoints):
                            selected_keypoints.append(pose_keypoints[idx])
                        else:
                            # Use zeros if keypoint index is out of range
                            selected_keypoints.append(np.zeros(3))
                    
                    keypoints_frame = np.array(selected_keypoints)
                else:
                    # No people detected, use zeros
                    keypoints_frame = np.zeros((len(keypoint_indices), 3))
                
                keypoints_data.append(keypoints_frame)
                
            except Exception as e:
                print(f"Error loading OpenPose keypoints for frame {frame_idx}: {e}")
                # Use zeros for this frame
                keypoints_frame = np.zeros((len(keypoint_indices), 3))
                keypoints_data.append(keypoints_frame)
        
        return np.array(keypoints_data)
    
    def load_hand_keypoints(self, patient_id, activity_id, segment_frames):
        """
        Load hand keypoints from MediaPipe for fingertips and wrist
        """
        # Define keypoint indices for fingertips and wrist
        keypoint_indices = {
            'wrist': 0,
            'thumb_tip': 4,
            'index_tip': 8,
            'middle_tip': 12,
            'ring_tip': 16,
            'pinky_tip': 20
        }
        
        # Get the camera ID based on the view
        camera_id = self.camera_mapping[self.top_view]
        if not camera_id:
            camera_id = self.ipsi_camera_map.get(patient_id)
            if not camera_id:
                print(f"No ipsilateral camera found for patient {patient_id}")
                return None
        
        # Convert camera_id to index for MAT files
        camera_indices = {'cam1': 0, 'cam2': 1, 'cam3': 2, 'cam4': 3}
        camera_idx = camera_indices.get(camera_id)
        if camera_idx is None:
            print(f"Invalid camera ID: {camera_id}")
            return None
        
        # Path to hand keypoints
        hand_path = os.path.join(
            HAND_DIR,
            f"patient_{patient_id}",
            f"activity_{activity_id}"
        )
        
        if not os.path.exists(hand_path):
            print(f"Hand keypoints not found at {hand_path}")
            return None
        
        # Look for MAT files in the directory
        mat_files = glob.glob(os.path.join(hand_path, "*.mat"))
        if not mat_files:
            print(f"No MAT files found in {hand_path}")
            return None
        
        # Load the MAT file - there should be one per view
        try:
            # Find the right MAT file for our camera
            correct_mat = None
            for mat_file in mat_files:
                if f"view{camera_idx+1}" in mat_file:
                    correct_mat = mat_file
                    break
            
            if not correct_mat:
                print(f"No MAT file found for camera {camera_id} in {hand_path}")
                # Return empty data
                return np.zeros((len(segment_frames), 12, 3))  # 12 = 6 keypoints × 2 hands
            
            # Load the MAT file
            mat_data = sio.loadmat(correct_mat)
            
            # MAT file structure might vary - adjust this based on the actual structure
            # This assumes the MAT file contains hand landmarks for each frame
            if 'landmarks' in mat_data:
                landmarks = mat_data['landmarks']
                
                # Process each frame
                keypoints_data = []
                for frame_idx in segment_frames:
                    if frame_idx < len(landmarks):
                        frame_landmarks = landmarks[frame_idx]
                        
                        # Extract left and right hand keypoints
                        left_hand = []
                        right_hand = []
                        
                        # Process left hand
                        if frame_landmarks.shape[0] > 0 and frame_landmarks[0] is not None:
                            for name, idx in keypoint_indices.items():
                                if idx < frame_landmarks[0].shape[0]:
                                    left_hand.append(np.append(frame_landmarks[0][idx], 1.0))  # Add confidence
                                else:
                                    left_hand.append(np.zeros(3))
                        else:
                            left_hand = [np.zeros(3) for _ in range(len(keypoint_indices))]
                        
                        # Process right hand
                        if frame_landmarks.shape[0] > 1 and frame_landmarks[1] is not None:
                            for name, idx in keypoint_indices.items():
                                if idx < frame_landmarks[1].shape[0]:
                                    right_hand.append(np.append(frame_landmarks[1][idx], 1.0))  # Add confidence
                                else:
                                    right_hand.append(np.zeros(3))
                        else:
                            right_hand = [np.zeros(3) for _ in range(len(keypoint_indices))]
                        
                        # Combine left and right hand keypoints
                        keypoints_frame = np.array(left_hand + right_hand)
                        keypoints_data.append(keypoints_frame)
                    else:
                        # Frame index out of range, use zeros
                        keypoints_frame = np.zeros((12, 3))  # 12 = 6 keypoints × 2 hands
                        keypoints_data.append(keypoints_frame)
                
                return np.array(keypoints_data)
            else:
                print(f"Unexpected MAT file structure: {list(mat_data.keys())}")
                # Return empty data
                return np.zeros((len(segment_frames), 12, 3))  # 12 = 6 keypoints × 2 hands
                
        except Exception as e:
            print(f"Error loading hand keypoints: {e}")
            # Return empty data
            return np.zeros((len(segment_frames), 12, 3))  # 12 = 6 keypoints × 2 hands
    
    def load_object_locations(self, patient_id, activity_id, segment_frames):
        """
        Load object locations from TridentNet
        """
        # Determine which subfolder to use based on the view
        if self.top_view:
            object_subdir = "top"
        else:
            # For ipsilateral, find the camera first
            camera_id = self.ipsi_camera_map.get(patient_id)
            if not camera_id:
                print(f"No ipsilateral camera found for patient {patient_id}")
                return None
            
            # Convert camera_id to folder name
            if camera_id == 'cam1':
                object_subdir = "ipsi_left"
            elif camera_id == 'cam2':
                object_subdir = "ipsi_right"
            else:
                print(f"Unsupported camera ID for object locations: {camera_id}")
                return None
        
        # Path to object location CSV
        object_path = os.path.join(
            OBJECT_DIR,
            object_subdir,
            f"patient_{patient_id}",
            f"activity_{activity_id}.csv"
        )
        
        if not os.path.exists(object_path):
            print(f"Object location data not found at {object_path}")
            return np.zeros((len(segment_frames), 1, 3))  # x, y, confidence
        
        try:
            # Read the CSV file
            object_df = pd.read_csv(object_path)
            
            # Process each frame
            object_data = []
            for frame_idx in segment_frames:
                if frame_idx < len(object_df):
                    # Extract x, y coordinates and add a confidence of 1.0
                    x, y = object_df.iloc[frame_idx]['x'], object_df.iloc[frame_idx]['y']
                    if np.isnan(x) or np.isnan(y):
                        object_data.append(np.array([[0.0, 0.0, 0.0]]))
                    else:
                        object_data.append(np.array([[x, y, 1.0]]))
                else:
                    # Frame index out of range, use zeros
                    object_data.append(np.array([[0.0, 0.0, 0.0]]))
            
            return np.array(object_data)
            
        except Exception as e:
            print(f"Error loading object locations: {e}")
            return np.zeros((len(segment_frames), 1, 3))  # x, y, confidence


class KeypointGraphDataset(Dataset):
    """
    Dataset for keypoint graph data
    """
    def __init__(self, segments, keypoint_processor, sequence_length=32, is_train=True):
        self.segments = segments
        self.keypoint_processor = keypoint_processor
        self.sequence_length = sequence_length
        self.is_train = is_train
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        video_id = segment['video_id']
        label = segment['label']
        
        # Extract patient_id and activity_id from video_id
        # Format: patient_{patient_id}_task_{activity_id}_{camera_id}_seg_{segment_id}
        parts = video_id.split('_')
        patient_id = parts[1]
        activity_id = parts[3]
        
        # Get segment frames
        frames = segment.get('segment_frames', list(range(len(segment['frames']))))
        
        # Sample frames if needed
        if len(frames) > self.sequence_length:
            # For training, randomly sample frames
            if self.is_train:
                step = max(1, len(frames) // self.sequence_length)
                starting_idx = np.random.randint(0, step)
                sampled_indices = [starting_idx + i * step for i in range(self.sequence_length)]
                sampled_indices = [min(i, len(frames) - 1) for i in sampled_indices]
            else:
                # For validation/testing, evenly sample frames
                step = len(frames) / self.sequence_length
                sampled_indices = [int(i * step) for i in range(self.sequence_length)]
        else:
            # If we have fewer frames than needed, pad by repeating the last frame
            sampled_indices = list(range(len(frames)))
            sampled_indices += [len(frames) - 1] * (self.sequence_length - len(frames))
        
        # Load keypoints for these frames
        body_keypoints = self.keypoint_processor.load_openpose_keypoints(
            patient_id, activity_id, [frames[i] for i in sampled_indices]
        )
        
        hand_keypoints = self.keypoint_processor.load_hand_keypoints(
            patient_id, activity_id, [frames[i] for i in sampled_indices]
        )
        
        object_locations = self.keypoint_processor.load_object_locations(
            patient_id, activity_id, [frames[i] for i in sampled_indices]
        )
        
        # Handle cases where keypoints couldn't be loaded
        if body_keypoints is None:
            body_keypoints = np.zeros((self.sequence_length, 8, 3))  # 8 body keypoints
        
        if hand_keypoints is None:
            hand_keypoints = np.zeros((self.sequence_length, 12, 3))  # 12 hand keypoints (6 per hand)
        
        if object_locations is None:
            object_locations = np.zeros((self.sequence_length, 1, 3))  # 1 object location
        
        # Create graph data for each frame
        graph_sequence = []
        for t in range(self.sequence_length):
            # Concatenate keypoints
            keypoints = np.concatenate([
                body_keypoints[t],  # 8 keypoints
                hand_keypoints[t],  # 12 keypoints
                object_locations[t]  # 1 keypoint
            ], axis=0)
            
            # Filter out keypoints with low confidence
            # valid_mask = keypoints[:, 2] > 0.2
            # if valid_mask.sum() > 0:
            #     valid_keypoints = keypoints[valid_mask, :2]
            # else:
            #     valid_keypoints = keypoints[:, :2]
            
            # Use all keypoints, filtering can be applied if needed
            valid_keypoints = keypoints[:, :2]
            
            # Create node features
            node_features = torch.tensor(valid_keypoints, dtype=torch.float)
            
            # Create edges - fully connected graph
            num_nodes = len(valid_keypoints)
            edge_index = []
            
            # Create edges for natural connections in body
            body_edges = [
                (0, 1),  # neck to right_shoulder
                (0, 4),  # neck to left_shoulder
                (1, 2),  # right_shoulder to right_elbow
                (2, 3),  # right_elbow to right_wrist
                (4, 5),  # left_shoulder to left_elbow
                (5, 6),  # left_elbow to left_wrist
                (0, 7),  # neck to mid_hip
            ]
            
            # Add body edges
            for src, dst in body_edges:
                edge_index.append([src, dst])
                edge_index.append([dst, src])  # Add reverse edge for undirected graph
            
            # Create edges for hand keypoints
            hand_offset = 8  # Start after body keypoints
            for hand_idx in range(2):  # Left and right hands
                hand_start = hand_offset + hand_idx * 6
                
                # Connect wrist to all fingertips
                for finger_idx in range(1, 6):  # 5 fingertips
                    src = hand_start  # wrist
                    dst = hand_start + finger_idx
                    edge_index.append([src, dst])
                    edge_index.append([dst, src])  # Add reverse edge
            
            # Connect object to right and left wrist
            object_idx = 8 + 12  # After body and hand keypoints
            edge_index.append([3, object_idx])    # Right wrist to object
            edge_index.append([object_idx, 3])    # Object to right wrist
            edge_index.append([6, object_idx])    # Left wrist to object
            edge_index.append([object_idx, 6])    # Object to left wrist
            
            # Create PyTorch Geometric Data object
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            graph_data = Data(x=node_features, edge_index=edge_index)
            
            graph_sequence.append(graph_data)
        
        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return graph_sequence, label_tensor, video_id


def collate_graph_sequences(batch):
    """
    Custom collate function for batching graph sequences
    """
    graph_sequences = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    video_ids = [item[2] for item in batch]
    
    # Reorganize as timestep batches for the transformer
    timestep_batches = []
    seq_length = len(graph_sequences[0])
    
    for t in range(seq_length):
        graphs_t = [seq[t] for seq in graph_sequences]
        batch_t = Batch.from_data_list(graphs_t)
        timestep_batches.append(batch_t)
    
    return timestep_batches, labels, video_ids


class SpatialTemporalGNN(nn.Module):
    """
    Graph Neural Network for processing spatial relationships in keypoints
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SpatialTemporalGNN, self).__init__()
        
        # Spatial GNN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        
        # Batch normalization for better stability
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, batch):
        """
        Process a batch of graphs
        batch: PyG Batch object
        """
        # First GCN layer
        x, edge_index = batch.x, batch.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global pooling to get graph-level representation
        batch_size = batch.num_graphs
        
        # Use scatter_mean to pool node features per graph
        graph_embeddings = pyg.nn.global_mean_pool(x, batch.batch)
        
        return graph_embeddings


class TemporalTransformerEncoder(nn.Module):
    """
    Transformer encoder for processing temporal relationships
    """
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super(TemporalTransformerEncoder, self).__init__()
        
        # Create encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Use batch_first=True for modern PyTorch
        )
        
        # Create transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, d_model))  # Max 100 timesteps
        nn.init.normal_(self.positional_encoding, mean=0, std=0.02)
    
    def forward(self, x):
        """
        Process a sequence of graph embeddings
        x: Tensor of shape [batch_size, seq_len, features]
        """
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Pass through transformer
        output = self.transformer_encoder(x)
        
        return output


class GNNTransformerModel(nn.Module):
    """
    Full model: GNN + Transformer for keypoint sequence classification
    """
    def __init__(
        self, 
        in_channels=2,          # 2D coordinates
        gnn_hidden_channels=64,
        gnn_out_channels=128,
        transformer_dim=128,
        transformer_heads=4,
        transformer_ff_dim=256,
        transformer_layers=4,
        num_classes=2,
        dropout=0.2
    ):
        super(GNNTransformerModel, self).__init__()
        
        # GNN for spatial processing
        self.gnn = SpatialTemporalGNN(
            in_channels=in_channels,
            hidden_channels=gnn_hidden_channels,
            out_channels=gnn_out_channels
        )
        
        # Transformer for temporal processing
        self.transformer = TemporalTransformerEncoder(
            d_model=gnn_out_channels,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            num_layers=transformer_layers,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(gnn_out_channels, gnn_out_channels//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_out_channels//2, num_classes)
        )
        
        # Final pooling strategy
        self.pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, graph_sequence):
        """
        Process a sequence of graphs
        graph_sequence: List of PyG Batch objects, one for each timestep
        """
        batch_size = graph_sequence[0].num_graphs
        seq_length = len(graph_sequence)
        
        # Process each graph in the sequence with the GNN
        gnn_outputs = []
        for t in range(seq_length):
            graph_embedding = self.gnn(graph_sequence[t])
            gnn_outputs.append(graph_embedding)
        
        # Stack GNN outputs to create a sequence
        # Shape: [batch_size, seq_length, gnn_out_channels]
        sequence_tensor = torch.stack(gnn_outputs, dim=1)
        
        # Process with transformer
        transformer_output = self.transformer(sequence_tensor)
        
        # Global temporal pooling
        # Option 1: Use the representation of the last timestep
        # final_repr = transformer_output[:, -1, :]
        
        # Option 2: Pool across the temporal dimension
        pooled_output = self.pooling(transformer_output.transpose(1, 2)).squeeze(-1)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits


def get_valid_rating(segment):
    """
    Get a valid rating only if t1 and t2 agree, or use the one that's available.
    Returns the rating if valid, or None if invalid.
    """
    t1_rating = segment['segment_ratings'].get('t1', None)
    t2_rating = segment['segment_ratings'].get('t2', None)
    
    # Try to convert ratings to integers
    try:
        t1_rating = int(t1_rating) if t1_rating is not None else None
    except (ValueError, TypeError):
        t1_rating = None
        
    try:
        t2_rating = int(t2_rating) if t2_rating is not None else None
    except (ValueError, TypeError):
        t2_rating = None
    
    # Decision logic
    if t1_rating is not None and t2_rating is not None:
        # Both ratings available - check if they match
        if t1_rating == t2_rating:
            return t1_rating  # They match, use either one
        else:
            return 'no_match'  # They don't match, invalid rating
    elif t1_rating is not None:
        # Only t1 is available
        return t1_rating
    elif t2_rating is not None:
        # Only t2 is available
        return t2_rating
    else:
        # No valid ratings
        return None


def process_dataset():
    """
    Process the pickle files to create segments for training/validation
    """
    print("Processing pickle files to create dataset segments...")
    pickle_files = glob.glob(os.path.join(PICKLE_DIR, '*.pkl'))
    
    # Create keypoint processor
    keypoint_processor = KeypointDataProcessor(top_view=bool(USE_TOP_VIEW))
    
    # Read camera assignments CSV
    camera_csv = "D:/Github/Multi_view-automatic-assessment/camera_assignments.csv"
    try:
        camera_df = pd.read_csv(camera_csv)
        patient_to_ipsilateral = dict(zip(camera_df['patient_id'], camera_df['ipsilateral_camera_id']))
    except Exception as e:
        print(f"Error reading camera assignments: {e}")
        patient_to_ipsilateral = {}
    
    # Process pickle files
    all_segments = []
    r1, r3, no_match = 0, 0, 0
    
    for pkl_file in tqdm(pickle_files, desc="Collecting segments"):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle file {pkl_file}: {e}")
            continue
        
        for camera_id in data:
            for segments_group in data[camera_id]:
                for segment in segments_group:
                    # Extract patient_id and camera_id from segment
                    patient_id = segment['patient_id']
                    segment_camera_id = segment['CameraId']
                    
                    # Determine which camera to use based on the view setting
                    if USE_TOP_VIEW:
                        # For top view, only use camera 3
                        if camera_id != 'cam3':
                            continue
                        ipsilateral_camera = camera_id
                    else:
                        # For ipsilateral view, check against the mapping
                        ipsilateral_camera = patient_to_ipsil