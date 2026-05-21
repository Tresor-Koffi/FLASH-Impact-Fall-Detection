import numpy as np
import torch

class Hypergraph:
    def __init__(self, num_node=33):
        self.num_node = num_node
        self.hyperedges = self.define_hyperedges()
        self.H, self.H_norm = self.construct_hypergraph()

    def define_hyperedges(self):
        """
        Define 26 biomechanically-grounded hyperedges based on:
        - Force transmission chains
        - Protective reflexes (arm extension)
        - Primary impact points
        - Functional joint groups
        """
        hyperedges = [
            # Head/face region
            [0, 1, 2, 3, 4, 5, 30],           # Face group (nose, eyes)
            [6, 7],                            # Ears
            [8, 9],                            # Mouth
            
            # Upper body - shoulders and torso
            [10, 11, 21, 32],                  # Shoulder-hip chain (force transmission)
            [10, 11],                          # Shoulder pair
            [21, 32],                          # Hip pair
            
            # Left arm chain (protective reflex)
            [10, 12, 14],                      # Left shoulder-elbow-wrist
            [12, 14],                          # Left elbow-wrist
            [14, 16, 18, 31],                  # Left hand (wrist, pinky, index, thumb)
            
            # Right arm chain (protective reflex)
            [11, 13, 15],                      # Right shoulder-elbow-wrist
            [13, 15],                          # Right elbow-wrist
            [15, 17, 19, 20],                  # Right hand (wrist, pinky, index, thumb)
            
            # Left leg chain (impact points)
            [21, 22, 24],                      # Left hip-knee-ankle
            [22, 24],                          # Left knee-ankle
            [24, 26, 28],                      # Left foot (ankle, heel, foot index)
            
            # Right leg chain (impact points)
            [32, 23, 25],                      # Right hip-knee-ankle
            [23, 25],                          # Right knee-ankle
            [25, 27, 29],                      # Right foot (ankle, heel, foot index)
            
            # Cross-body coordination
            [10, 32],                          # Left shoulder - right hip diagonal
            [11, 21],                          # Right shoulder - left hip diagonal
            
            # Primary impact points (forward fall)
            [0, 14, 15, 22, 23],               # Nose, wrists, knees
            
            # Primary impact points (backward fall)
            [21, 32, 26, 27],                  # Hips and heels
            
            # Primary impact points (lateral fall)
            [10, 21, 22, 24],                  # Left side
            [11, 32, 23, 25],                  # Right side
            
            # Full body vertical chain
            [0, 10, 11, 21, 32, 22, 23],       # Head to knees
            
            # Lower extremity group
            [24, 25, 26, 27, 28, 29],          # All foot joints
        ]
        return hyperedges

    def construct_hypergraph(self):
        """
        Construct single-matrix hypergraph representation.
        Returns H (incidence matrix) and H_norm (normalized Laplacian).
        
        Paper Equation 1:
        H_norm = D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}
        """
        num_hyperedges = len(self.hyperedges)
        
        # Create incidence matrix H: [num_nodes, num_hyperedges]
        H = np.zeros((self.num_node, num_hyperedges), dtype=np.float32)
        for e_idx, edge in enumerate(self.hyperedges):
            for v in edge:
                if v < self.num_node:
                    H[v, e_idx] = 1.0

        # Compute degree matrices
        # D_v: vertex degree (sum over hyperedges for each node)
        d_v = np.sum(H, axis=1)
        # D_e: hyperedge degree (sum over nodes for each hyperedge)
        d_e = np.sum(H, axis=0)
        
        # Add epsilon for numerical stability
        eps = 1e-8
        D_v_inv_sqrt = np.diag(1.0 / np.sqrt(d_v + eps))
        D_e_inv = np.diag(1.0 / (d_e + eps))
        
        # Normalized hypergraph Laplacian (Equation 1 in paper):
        # H_norm = D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}
        H_norm = D_v_inv_sqrt @ H @ D_e_inv @ H.T @ D_v_inv_sqrt

        # Convert to PyTorch tensors
        H = torch.tensor(H, dtype=torch.float32)
        H_norm = torch.tensor(H_norm, dtype=torch.float32)

        return H, H_norm

    def get_num_hyperedges(self):
        """Return number of hyperedges (E=26 as per paper)."""
        return len(self.hyperedges)