import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
import torch
from HGCN.hypergraph import Hypergraph

# Define body part indices (MediaPipe 33 joints)
index_nose = 0
index_left_eye_inner = 1
index_left_eye = 2
index_left_eye_outer = 3
index_right_eye = 4
index_right_eye_outer = 5
index_left_ear = 6
index_right_ear = 7
index_mouth_left = 8
index_mouth_right = 9
index_left_shoulder = 10
index_right_shoulder = 11
index_left_elbow = 12
index_right_elbow = 13
index_left_wrist = 14
index_right_wrist = 15
index_left_pinky = 16
index_right_pinky = 17
index_left_index = 18
index_right_index = 19
index_right_thumb = 20
index_left_hip = 21
index_left_knee = 22
index_right_knee = 23
index_left_ankle = 24
index_right_ankle = 25
index_left_heel = 26
index_right_heel = 27
index_left_foot_index = 28
index_right_foot_index = 29
index_right_eye_inner = 30
index_left_thumb = 31
index_right_hip = 32


class FallDataLoader:
    """
    Data loader for skeleton-based fall detection.
    
    Implements preprocessing with single-matrix hypergraph as per paper Section 3.2.
    """
    def __init__(self, dir):
        super().__init__()
        self.num_channel = 3  # x, y, z coordinates
        self.dir = dir
        self.body_part = self.body_parts()
        self.num_timestep = 100
        self.num_node = len(self.body_part)  # 33 joints
        
        # Load dataset
        self.train_x, self.train_y = self.import_dataset()
        self.batch_size = self.train_y.shape[0] // self.num_timestep
        
        # Initialize hypergraph (single-matrix approach)
        self.hypergraph = Hypergraph(num_node=self.num_node)
        
        # Scaler for normalization
        self.sc1 = StandardScaler()
        
        # Preprocess data
        self.scaled_x, self.scaled_y = self.preprocessing()

    def body_parts(self):
        """Define the 33 body parts as indices (MediaPipe format)."""
        return [
            index_nose, index_left_eye_inner, index_left_eye,
            index_left_eye_outer, index_right_eye, index_right_eye_outer,
            index_left_ear, index_right_ear, index_mouth_left,
            index_mouth_right, index_left_shoulder, index_right_shoulder,
            index_left_elbow, index_right_elbow, index_left_wrist,
            index_right_wrist, index_left_pinky, index_right_pinky,
            index_left_index, index_right_index, index_right_thumb,
            index_left_hip, index_left_knee, index_right_knee,
            index_left_ankle, index_right_ankle, index_left_heel,
            index_right_heel, index_left_foot_index, index_right_foot_index,
            index_right_eye_inner, index_left_thumb, index_right_hip
        ]

    def import_dataset(self):
        """Load the training data (X and Y)."""
        try:
            train_x = pd.read_csv(f"./{self.dir}/Train_X.csv", header=None).iloc[:, :].values
            print("Train_X shape:", train_x.shape)
            train_y = pd.read_csv(f"./{self.dir}/Train_Y.csv", header=None).iloc[:, :].values
            print("Train_Y shape:", train_y.shape)
            return train_x, train_y
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def augment_motion_sequence_fixed_length(self, sequence, max_variation=0.25):
        """Time-warping augmentation - stretch or compress sequence."""
        seq_len = sequence.shape[0]
        max_frames_to_add_or_remove = int(seq_len * max_variation)
        l = np.random.randint(0, max(1, max_frames_to_add_or_remove))
        
        if np.random.random() > 0.5:
            # Stretch
            indices = np.linspace(0, seq_len - 1, seq_len + l, dtype=int)
            extended_sequence = sequence[indices]
            fixed_sequence = np.array([
                extended_sequence[int(i)] 
                for i in np.linspace(0, len(extended_sequence) - 1, seq_len)
            ])
        else:
            # Compress
            indices = np.sort(np.random.choice(seq_len, max(1, seq_len - l), replace=False))
            reduced_sequence = sequence[indices]
            fixed_sequence = np.array([
                reduced_sequence[int(i)] 
                for i in np.linspace(0, len(reduced_sequence) - 1, seq_len)
            ])
        return fixed_sequence

    def rotate_skeleton(self, sequence, angle_range=(-10, 10)):
        """Rotate skeleton in 3D space around z-axis."""
        angle = np.radians(np.random.uniform(*angle_range))
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        return np.einsum('ij,tkj->tki', rotation_matrix, sequence)

    def scale_skeleton(self, sequence, scale_range=(0.9, 1.1)):
        """Scale skeleton dimensions uniformly."""
        scale_factor = np.random.uniform(*scale_range)
        return sequence * scale_factor

    def preprocessing(self):
        """
        Preprocess training data using single-matrix hypergraph.
        
        Paper Section 3.2: Uses only H_norm for hypergraph convolution,
        not dual matrices (H and H2).
        """
        # Extract features for selected body parts
        X_train = np.zeros((self.train_x.shape[0], self.num_node * self.num_channel), dtype=np.float32)
        
        for row in range(self.train_x.shape[0]):
            counter = 0
            for parts in self.body_part:
                X_train[row, counter:counter+self.num_channel] = self.train_x[row, parts:parts+self.num_channel]
                counter += self.num_channel
        
        # Normalize features
        X_train = self.sc1.fit_transform(X_train)
        
        # Get normalized hypergraph Laplacian (single-matrix approach)
        H_norm = self.hypergraph.H_norm.numpy() if isinstance(self.hypergraph.H_norm, torch.Tensor) else self.hypergraph.H_norm
        
        # Reshape into sequences
        num_batches = X_train.shape[0] // self.num_timestep
        X_train_ = np.zeros((num_batches, self.num_timestep, self.num_node, self.num_channel), dtype=np.float32)
        Y_train_ = np.zeros((num_batches, self.num_timestep), dtype=np.float32)
        
        # Process each frame with hypergraph convolution
        for batch in range(num_batches):
            for timestep in range(self.num_timestep):
                frame_idx = (batch * self.num_timestep) + timestep
                frame_features = X_train[frame_idx].reshape(self.num_node, self.num_channel)
                
                # Single-matrix hypergraph operation (Equation 1 in paper):
                # H_norm already encodes: D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}
                weighted_features = H_norm @ frame_features
                
                X_train_[batch, timestep] = weighted_features
                Y_train_[batch, timestep] = self.train_y[frame_idx]
        
        print("Original shapes:", X_train_.shape, Y_train_.shape)
        
        # Apply data augmentation
        augmented_data = []
        augmented_labels = []
        
        for i, sequence in enumerate(X_train_):
            augmented_sequences = [
                self.augment_motion_sequence_fixed_length(sequence),
                self.rotate_skeleton(sequence),
                self.scale_skeleton(sequence)
            ]
            augmented_data.extend(augmented_sequences)
            augmented_labels.extend([Y_train_[i]] * len(augmented_sequences))
        
        augmented_data = np.array(augmented_data, dtype=np.float32)
        augmented_labels = np.array(augmented_labels, dtype=np.float32)
        
        # Combine original and augmented data
        X_final = np.concatenate([X_train_, augmented_data], axis=0)
        Y_final = np.concatenate([Y_train_, augmented_labels], axis=0)
        
        print("Final shapes after augmentation:", X_final.shape, Y_final.shape)
        
        return X_final, Y_final

    def prepare_hypergraph_features(self):
        """Returns the hypergraph features needed for training (single-matrix)."""
        return {
            'H': self.hypergraph.H,
            'H_norm': self.hypergraph.H_norm,
        }

    def save_scalers(self, path):
        """Save the scalers for future use."""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.sc1, os.path.join(path, 'sc1.pkl'))

    def load_scalers(self, path):
        """Load the scalers from disk."""
        self.sc1 = joblib.load(os.path.join(path, 'sc1.pkl'))