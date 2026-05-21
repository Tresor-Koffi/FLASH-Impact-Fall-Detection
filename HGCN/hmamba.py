import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from HGCN.hypergraph_ablation import HypergraphAblation

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


# =================== HYPERGRAPH CONVOLUTION (Section 3.3) ===================
class HypergraphConv(nn.Module):
    """
    Hypergraph Convolution Layer using single normalized matrix.
    
    Paper Equation 2:
    X^{(l+1)} = σ(H_norm X^{(l)} W^{(l)})
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
    
    def forward(self, x, H_norm):
        """
        Args:
            x: [batch_size, seq_len, num_nodes, features]
            H_norm: [num_nodes, num_nodes] normalized hypergraph Laplacian
        Returns:
            [batch_size, seq_len, num_nodes, out_features]
        """
        # Apply hypergraph convolution: H_norm @ X
        x = torch.einsum('ij,btjf->btif', H_norm, x)
        # Apply linear transformation
        x = self.linear(x)
        return F.relu(x)


# =================== SELECTIVE SSM (Section 3.4) ===================
class SelectiveSSM(nn.Module):
    """
    Selective State-Space Model based on Mamba architecture.
    
    Paper Equations 3-6:
    Continuous: dh/dt = A(t)h(t) + B(t)x(t), y(t) = C(t)h(t) + Dx(t)
    Discrete: h_k = Ã_k h_{k-1} + B̃_k x_k, y_k = C_k h_k + D x_k
    """
    def __init__(self, d_model, d_state=256, dropout_rate=0.3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input projection
        self.in_proj = nn.Linear(d_model, d_model * 2)
        
        # SSM parameters (input-dependent through projections)
        self.A_proj = nn.Linear(d_model, d_state)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Delta (step size) projection
        self.delta_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize
        nn.init.kaiming_normal_(self.in_proj.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.out_proj.weight, nonlinearity='relu')
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection and split
        x_proj = self.in_proj(x)
        x_main, x_gate = x_proj.chunk(2, dim=-1)
        
        # Compute input-dependent SSM parameters
        A = -F.softplus(self.A_proj(x_main))  # [B, T, d_state]
        B = self.B_proj(x_main)                # [B, T, d_state]
        C = self.C_proj(x_main)                # [B, T, d_state]
        
        # Compute step size
        delta = F.softplus(self.delta_proj(x_main))  # [B, T, d_model]
        
        # Discretize (zero-order hold approximation)
        # Ã = exp(Δ * A), simplified for efficiency
        deltaA = delta.unsqueeze(-1) * A.unsqueeze(-2)  # [B, T, d_model, d_state]
        
        # Sequential state update (selective scan)
        h = torch.zeros(batch_size, self.d_model, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            # h_k = Ã_k * h_{k-1} + B̃_k * x_k
            h = h * torch.exp(deltaA[:, t]) + B[:, t].unsqueeze(1) * x_main[:, t].unsqueeze(-1)
            # y_k = C_k * h_k
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # [B, T, d_model]
        
        # Add skip connection with D
        y = y + x_main * self.D
        
        # Gate and output projection
        y = y * F.silu(x_gate)
        y = self.out_proj(y)
        
        if self.training:
            y = self.dropout(y)
        
        return y


# =================== MAMBA BLOCK (Section 3.4) ===================
class MambaBlock(nn.Module):
    """
    Mamba block with residual connection.
    """
    def __init__(self, d_model, d_state=256, dropout_rate=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, dropout_rate)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        return x + self.ssm(self.norm(x))


# =================== MAIN MODEL (Figure 2) ===================
class HyperMamba(nn.Module):
    """
    HyperMamba: Hypergraph State-Space Models for Impact Fall Detection.
    
    Architecture (Section 3):
    1. Hypergraph Convolution (2 layers) for spatial modeling
    2. Mamba blocks for temporal modeling  
    3. Multi-scale temporal convolutions
    4. Classification head
    """
    def __init__(self, device='cuda', num_nodes=33, num_hyperedges=6,
                 hgcn_hidden=128, mamba_d_state=256, dropout_rate=0.3):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.num_hyperedges = num_hyperedges
        
        # Load hypergraph structure
        hypergraph = HypergraphAblation(num_node=num_nodes, ablation_mode='full')
        self.register_buffer('H', hypergraph.H)           # [33, 6]
        self.register_buffer('H_norm', hypergraph.H_norm) # [33, 33]
        
        # =================== Hypergraph Convolution Layers (Section 3.3) ===================
        # Input: 3 channels (x, y, z coordinates)
        self.hgcn1 = HypergraphConv(3, hgcn_hidden)
        self.hgcn2 = HypergraphConv(hgcn_hidden, hgcn_hidden)
        
        # =================== Mamba Block (Section 3.4) ===================
        # Reshape features for Mamba: [B, T, num_nodes * hgcn_hidden]
        mamba_dim = num_nodes * hgcn_hidden
        self.mamba = MambaBlock(mamba_dim, d_state=mamba_d_state, dropout_rate=dropout_rate)
        
        # =================== Multi-Scale Temporal Convolutions (Section 3.5) ===================
        # Kernel sizes: 9, 15, 20 for fine, medium, coarse patterns
        self.conv_z1 = nn.Conv2d(hgcn_hidden, 64, kernel_size=(9, 1), padding=(4, 0))
        self.conv_z2 = nn.Conv2d(64, 64, kernel_size=(15, 1), padding=(7, 0))
        self.pad_z3 = nn.ZeroPad2d((0, 0, 10, 9))
        self.conv_z3 = nn.Conv2d(64, 64, kernel_size=(20, 1))
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # =================== Classification Head (Section 3.5) ===================
        # Concatenated features: 64 * 3 = 192 channels
        self.conv_final_1 = nn.Conv2d(192, 64, kernel_size=(1, 1))
        self.conv_final_2 = nn.Conv2d(64, 32, kernel_size=(1, num_nodes))
        self.dense1 = nn.Linear(32, 32)
        self.dense2 = nn.Linear(32, 1)
        self.dropout_final = nn.Dropout(0.5)
        
        # Training components
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)

    def forward(self, x):
        """
        Forward pass through HyperMamba.
        
        Args:
            x: Input tensor [B, T, N=33, C=3] - skeleton sequences
        Returns:
            logits: [B, T] - frame-wise impact predictions
        """
        B, T, N, C = x.shape
        
        # =================== Hypergraph Convolution (Equation 2) ===================
        # Two layers of hypergraph convolution for spatial feature extraction
        x = self.hgcn1(x, self.H_norm)  # [B, T, N, 128]
        x = self.hgcn2(x, self.H_norm)  # [B, T, N, 128]
        
        # =================== Mamba Temporal Modeling (Equations 7-9) ===================
        # Reshape for Mamba: [B, T, N*C'] -> [B, T, 33*128]
        x_seq = x.reshape(B, T, -1)
        x_seq = self.mamba(x_seq)  # [B, T, 33*128]
        
        # Reshape back: [B, T, N, C']
        x = x_seq.reshape(B, T, N, -1)  # [B, T, 33, 128]
        
        # =================== Multi-Scale Temporal Convolutions (Equations 10-12) ===================
        # Permute for Conv2d: [B, C, T, N]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, 128, T, 33]
        
        z1 = self.dropout(F.relu(self.conv_z1(x)))      # [B, 64, T, 33] - kernel 9
        z2 = self.dropout(F.relu(self.conv_z2(z1)))     # [B, 64, T, 33] - kernel 15
        z3_padded = self.pad_z3(z2)
        z3 = self.dropout(F.relu(self.conv_z3(z3_padded)))  # [B, 64, T, 33] - kernel 20
        
        # Concatenate multi-scale features (Equation 13)
        z = torch.cat([z1, z2, z3], dim=1)  # [B, 192, T, 33]
        
        # =================== Classification (Equation 14) ===================
        x = F.relu(self.conv_final_1(z))    # [B, 64, T, 33]
        x = self.conv_final_2(x)            # [B, 32, T, 1]
        x = x.squeeze(-1).permute(0, 2, 1)  # [B, T, 32]
        
        x = F.relu(self.dense1(x))          # [B, T, 32]
        if self.training:
            x = self.dropout_final(x)
        x = self.dense2(x).squeeze(-1)      # [B, T]
        
        return x

    @torch.no_grad()
    def predict(self, x):
        """Predict without gradient computation."""
        self.eval()
        return self(x)

    @torch.no_grad()
    def predict_proba(self, test_loader):
        """Predict probabilities for test set."""
        self.eval()
        self.to(self.device)
        all_probs = []
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(self.device)
            logits = self(batch_x)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
        return torch.cat(all_probs, dim=0).numpy()

    def fit(self, train_loader, val_loader, epochs=300, patience=15):
        """Train the model with early stopping."""
        self.to(self.device)
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=patience//2)
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()
                train_loss += loss.item()

            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    val_loss += self.criterion(self(batch_x), batch_y).item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                torch.save(best_state, 'best_hypermamba_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

        if best_state is not None:
            self.load_state_dict(best_state)
        return best_val_loss

    @torch.no_grad()
    def evaluate(self, test_loader):
        """Evaluate model on test set."""
        self.eval()
        self.to(self.device)
        total_loss = 0
        all_preds, all_targets = [], []

        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            logits = self(batch_x)
            loss = self.criterion(logits, batch_y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0).flatten()
        all_targets = np.concatenate(all_targets, axis=0).flatten()

        # Compute metrics
        tn = np.sum((1 - all_preds) * (1 - all_targets))
        fp = np.sum(all_preds * (1 - all_targets))
        specificity = tn / (tn + fp + 1e-10)

        metrics = {
            'loss': total_loss / len(test_loader),
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds, zero_division=0),
            'recall': recall_score(all_targets, all_preds, zero_division=0),
            'specificity': specificity,
            'f1': f1_score(all_targets, all_preds, zero_division=0)
        }
        return metrics

    def save(self, filepath):
        """Save model state."""
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        """Load model state."""
        self.load_state_dict(torch.load(filepath, map_location=self.device))
        self.to(self.device)

    def summary(self):
        """Print model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\nHyperMamba Model Summary")
        print("=" * 50)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Number of nodes (J): {self.num_nodes}")
        print(f"Number of hyperedges (E): {self.num_hyperedges}")
        print("\nArchitecture:")
        print("-" * 50)
        print("1. Hypergraph Convolution: 2 layers (128 hidden)")
        print("2. Mamba Block: 1 layer (d_state=256)")
        print("3. Multi-scale Temporal Conv: kernels {9, 15, 20}")
        print("4. Classification Head: FC layers")
        print("=" * 50)