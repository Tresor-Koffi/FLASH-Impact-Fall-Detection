"""
Hypergraph with ablation support for Reviewer #2 Question 2
Tests sensitivity to hyperedge construction using 6 biomechanical hyperedges
"""

import numpy as np
import torch

class HypergraphAblation:
    def __init__(self, num_node=33, ablation_mode="full"):
        """
        Args:
            num_node: number of joints (33)
            ablation_mode: 
                - "full": all 6 hyperedges (baseline)
                - "no_torso": remove torso hyperedge
                - "no_arms": remove both arm hyperedges
                - "no_legs": remove both leg hyperedges
                - "no_head": remove head hyperedge
                - "all_in_one": all joints in one hyperedge
                - "random": random hyperedge assignment (6 groups)
        """
        self.num_node = num_node
        self.ablation_mode = ablation_mode
        self.hyperedges = self.define_hyperedges()
        self.H, self.H_norm = self.construct_hypergraph()

    def define_hyperedges(self):
        """
        Define 6 biomechanical hyperedges as per paper Section 3.2
        Based on: force transmission chains, protective reflexes, primary impact points
        """
        
        # The 6 hyperedges from your paper
        full_hyperedges = [
            # Torso (force transmission hub, core stabilization)
            [10, 11, 21, 32],           # Left/right shoulders + left/right hips
            
            # Left Arm (protective reflex)
            [10, 12, 14],                # Left shoulder, elbow, wrist
            
            # Right Arm (protective reflex)
            [11, 13, 15],                # Right shoulder, elbow, wrist
            
            # Left Leg (primary impact point, force transmission)
            [21, 22, 24],                # Left hip, knee, ankle
            
            # Right Leg (primary impact point, force transmission)
            [32, 23, 25],                # Right hip, knee, ankle
            
            # Head (orientation tracking, balance)
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Nose, eyes, ears, mouth
        ]
        
        if self.ablation_mode == "no_torso":
            # Remove torso hyperedge (index 0)
            hyperedges = full_hyperedges[1:]
            print(f"   Removed torso hyperedge")
            
        elif self.ablation_mode == "no_arms":
            # Remove both arm hyperedges (indices 1 and 2)
            hyperedges = [full_hyperedges[0]] + full_hyperedges[3:]
            print(f"   Removed left and right arm hyperedges")
            
        elif self.ablation_mode == "no_legs":
            # Remove both leg hyperedges (indices 3 and 4)
            hyperedges = full_hyperedges[:3] + [full_hyperedges[5]]
            print(f"   Removed left and right leg hyperedges")
            
        elif self.ablation_mode == "no_head":
            # Remove head hyperedge (index 5)
            hyperedges = full_hyperedges[:5]
            print(f"   Removed head hyperedge")
            
        elif self.ablation_mode == "all_in_one":
            # Single hyperedge with all joints
            hyperedges = [list(range(self.num_node))]
            print(f"   All joints in one hyperedge")
            
        elif self.ablation_mode == "random":
            # Random assignment: 6 random hyperedges (non-biomechanical)
            num_hyperedges = 6
            hyperedges = []
            for i in range(num_hyperedges):
                size = np.random.randint(3, 10)  # Random size between 3-9 joints
                joints = np.random.choice(self.num_node, size, replace=False)
                hyperedges.append(joints.tolist())
            print(f"   Random hyperedge assignment (non-biomechanical)")
            
        else:  # "full"
            hyperedges = full_hyperedges
            print(f"   Full model with 6 biomechanical hyperedges")
        
        print(f"   Number of hyperedges: {len(hyperedges)}")
        return hyperedges

    def construct_hypergraph(self):
        """Construct incidence matrix H and normalized H_norm"""
        num_hyperedges = len(self.hyperedges)
        
        # Create incidence matrix H: [num_nodes, num_hyperedges]
        H = np.zeros((self.num_node, num_hyperedges), dtype=np.float32)
        for e_idx, edge in enumerate(self.hyperedges):
            for v in edge:
                if v < self.num_node:
                    H[v, e_idx] = 1.0
        
        # Compute degree matrices
        d_v = np.sum(H, axis=1)
        d_e = np.sum(H, axis=0)
        
        eps = 1e-8
        D_v_inv_sqrt = np.diag(1.0 / np.sqrt(d_v + eps))
        D_e_inv = np.diag(1.0 / (d_e + eps))
        
        # Normalized hypergraph Laplacian (Equation 1 in paper)
        H_norm = D_v_inv_sqrt @ H @ D_e_inv @ H.T @ D_v_inv_sqrt
        
        H = torch.tensor(H, dtype=torch.float32)
        H_norm = torch.tensor(H_norm, dtype=torch.float32)
        
        return H, H_norm

    def get_num_hyperedges(self):
        return len(self.hyperedges)