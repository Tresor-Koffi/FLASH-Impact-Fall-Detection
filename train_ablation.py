"""
Hyperedge Ablation Study for Reviewer #2 Question 2
Run: python train_ablation.py

Tests:
1. Full model (6 hyperedges) - baseline
2. Remove torso hyperedge
3. Remove arms hyperedges  
4. Remove legs hyperedges
5. Remove head hyperedge
6. All joints in one hyperedge
7. Random hyperedge assignment (6 random groups)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from HGCN.data_processing import FallDataLoader
from HGCN.hypergraph_ablation import HypergraphAblation


# =================== HYPERGRAPH CONVOLUTION ===================

class HypergraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
    
    def forward(self, x, H_norm):
        x = torch.einsum('ij,btjf->btif', H_norm, x)
        x = self.linear(x)
        return nn.functional.relu(x)


# =================== SELECTIVE SSM ===================

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=256, dropout_rate=0.3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.A_proj = nn.Linear(d_model, d_state)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))
        self.delta_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        x_proj = self.in_proj(x)
        x_main, x_gate = x_proj.chunk(2, dim=-1)
        
        A = -nn.functional.softplus(self.A_proj(x_main))
        B = self.B_proj(x_main)
        C = self.C_proj(x_main)
        delta = nn.functional.softplus(self.delta_proj(x_main))
        
        deltaA = delta.unsqueeze(-1) * A.unsqueeze(-2)
        
        h = torch.zeros(batch_size, self.d_model, self.d_state, device=device)
        outputs = []
        
        for t in range(seq_len):
            h = h * torch.exp(deltaA[:, t]) + B[:, t].unsqueeze(1) * x_main[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)
        y = y + x_main * self.D
        y = y * nn.functional.silu(x_gate)
        y = self.out_proj(y)
        
        if self.training:
            y = self.dropout(y)
        
        return y


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=256, dropout_rate=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, dropout_rate)
    
    def forward(self, x):
        return x + self.ssm(self.norm(x))


# =================== HYPERMAMBA WITH CUSTOM HYPERGRAPH ===================

class HyperMambaAblation(nn.Module):
    def __init__(self, device='cuda', ablation_mode="full", num_nodes=33):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        
        # Load custom hypergraph based on ablation mode
        hypergraph = HypergraphAblation(num_node=num_nodes, ablation_mode=ablation_mode)
        self.register_buffer('H_norm', hypergraph.H_norm)
        
        # Hypergraph convolution layers
        self.hgcn1 = HypergraphConv(3, 128)
        self.hgcn2 = HypergraphConv(128, 128)
        
        # Mamba block
        mamba_dim = num_nodes * 128
        self.mamba = MambaBlock(mamba_dim, d_state=256, dropout_rate=0.3)
        
        # Multi-scale temporal convolutions
        self.conv_z1 = nn.Conv2d(128, 64, kernel_size=(9, 1), padding=(4, 0))
        self.conv_z2 = nn.Conv2d(64, 64, kernel_size=(15, 1), padding=(7, 0))
        self.pad_z3 = nn.ZeroPad2d((0, 0, 10, 9))
        self.conv_z3 = nn.Conv2d(64, 64, kernel_size=(20, 1))
        self.dropout = nn.Dropout(0.3)
        
        # Classification head
        self.conv_final_1 = nn.Conv2d(192, 64, kernel_size=(1, 1))
        self.conv_final_2 = nn.Conv2d(64, 32, kernel_size=(1, num_nodes))
        self.dense1 = nn.Linear(32, 32)
        self.dense2 = nn.Linear(32, 1)
        self.dropout_final = nn.Dropout(0.5)
    
    def forward(self, x):
        B, T, N, C = x.shape
        
        # Hypergraph convolution
        x = self.hgcn1(x, self.H_norm)
        x = self.hgcn2(x, self.H_norm)
        
        # Mamba temporal modeling
        x_seq = x.reshape(B, T, -1)
        x_seq = self.mamba(x_seq)
        x = x_seq.reshape(B, T, N, -1)
        
        # Multi-scale temporal convolutions
        x = x.permute(0, 3, 1, 2).contiguous()
        
        z1 = self.dropout(nn.functional.relu(self.conv_z1(x)))
        z2 = self.dropout(nn.functional.relu(self.conv_z2(z1)))
        z3_padded = self.pad_z3(z2)
        z3 = self.dropout(nn.functional.relu(self.conv_z3(z3_padded)))
        
        z = torch.cat([z1, z2, z3], dim=1)
        
        # Classification
        x = nn.functional.relu(self.conv_final_1(z))
        x = self.conv_final_2(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        
        x = nn.functional.relu(self.dense1(x))
        if self.training:
            x = self.dropout_final(x)
        x = self.dense2(x).squeeze(-1)
        
        return x


# =================== TRAINING FUNCTION ===================

def train_model(ablation_mode, train_loader, val_loader, epochs=50):
    """Train model with given ablation mode and return best validation accuracy"""
    
    print(f"\n{'='*60}")
    print(f"Training: {ablation_mode}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HyperMambaAblation(device=device, ablation_mode=ablation_mode).to(device)
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            preds = (outputs > 0).float()
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.numel()
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                preds = (outputs > 0).float()
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.numel()
        
        val_acc = val_correct / val_total
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    print(f"  Best validation accuracy: {best_val_acc*100:.2f}%")
    return best_val_acc


# =================== MAIN ===================

def main():
    print("="*60)
    print("Hyperedge Ablation Study for Reviewer #2 Question 2")
    print("Testing sensitivity to hyperedge construction")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    data_loader = FallDataLoader("Data/")
    
    X = data_loader.scaled_x
    y = data_loader.scaled_y
    
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    
    # FIX: Flatten y to 1D for stratification
    # y shape is (376, 100), we need to use the last frame for stratification
    y_flat = y[:, -1]  # Use last frame label for stratification
    
    # Reshape X to 2D for splitting: (376, 100, 33, 3) -> (376, 100*33*3)
    X_flat = X.reshape(X.shape[0], -1)
    
    print(f"   Flattened: X_flat={X_flat.shape}, y_flat={y_flat.shape}")
    
    # Split into train/val (80/20)
    train_x_flat, val_x_flat, train_y_flat, val_y_flat = train_test_split(
        X_flat, y_flat, test_size=0.2, random_state=42, stratify=y_flat
    )
    
    # Reshape back to original dimensions
    train_x = train_x_flat.reshape(-1, 100, 33, 3)
    val_x = val_x_flat.reshape(-1, 100, 33, 3)
    
    # For labels, we need full sequence (100 frames)
    # Get indices of train and val samples
    train_indices = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42, stratify=y_flat
    )[0]
    val_indices = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42, stratify=y_flat
    )[1]
    
    train_y = y[train_indices]
    val_y = y[val_indices]
    
    train_dataset = TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y))
    val_dataset = TensorDataset(torch.FloatTensor(val_x), torch.FloatTensor(val_y))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Run ablations
    ablation_modes = [
        "full",
        "no_torso",
        "no_arms",
        "no_legs",
        "no_head",
        "all_in_one",
        "random"
    ]
    
    results = {}
    
    for mode in ablation_modes:
        acc = train_model(mode, train_loader, val_loader, epochs=50)
        results[mode] = acc
    
    # Print summary table
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    
    baseline = results["full"]
    
    print("\n| Configuration | Accuracy (%) | Drop from Full (pp) |")
    print("|---------------|--------------|---------------------|")
    
    for mode, acc in results.items():
        drop = (baseline - acc) * 100
        print(f"| {mode:<13} | {acc*100:.2f} | {drop:.2f} |")
    
    # Save results
    import json
    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to ablation_results.json")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()