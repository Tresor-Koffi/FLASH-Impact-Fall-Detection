"""
Occlusion Robustness Experiment - Standalone Version
Run this file directly: python test_occlusion.py
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# Import your modules
from HGCN.data_processing import FallDataLoader
from HGCN.hmamba import HyperMamba


def random_joint_dropout(skeleton, dropout_rate):
    """Randomly drop joints by setting coordinates to zero."""
    B, T, J, C = skeleton.shape
    device = skeleton.device
    
    mask = torch.ones(B, T, J, device=device)
    
    for b in range(B):
        for t in range(T):
            num_drop = int(J * dropout_rate)
            if num_drop > 0:
                drop_indices = torch.randperm(J, device=device)[:num_drop]
                mask[b, t, drop_indices] = 0
    
    return skeleton * mask.unsqueeze(-1)


def add_gaussian_noise(skeleton, sigma):
    """Add Gaussian noise to joint coordinates."""
    noise = torch.randn_like(skeleton) * sigma
    return skeleton + noise


def main():
    print("="*60)
    print("Occlusion Robustness Experiment for Reviewer #2 Q1")
    print("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\n1. Loading data...")
    data_loader = FallDataLoader("Data/")
    
    # Create test loader
    test_x = data_loader.scaled_x
    test_y = data_loader.scaled_y
    
    test_dataset = TensorDataset(
        torch.FloatTensor(test_x),
        torch.FloatTensor(test_y)
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"   Test samples: {len(test_dataset)}")
    
    # Load trained model
    print("\n2. Loading trained HyperMamba model...")
    model = HyperMamba(device=device).to(device)
    
    model_path = "saved_models/best_hypermamba_model.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"   Loaded model from {model_path}")
    else:
        print(f"   ERROR: No trained model found at {model_path}")
        print("   Please train the model first using: python GCN/train.py")
        return
    
    model.eval()
    
    # Test different dropout rates
    print("\n" + "="*60)
    print("3. Testing Random Joint Dropout (Occlusion)")
    print("="*60)
    
    dropout_rates = [0.0, 0.1, 0.2, 0.3]
    baseline_acc = None
    
    for rate in dropout_rates:
        print(f"\n  Dropout rate: {rate*100:.0f}%")
        all_preds = []
        all_labels = []
        
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            masked_x = random_joint_dropout(batch_x, rate)
            
            with torch.no_grad():
                logits = model(masked_x)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0).flatten()
        all_labels = np.concatenate(all_labels, axis=0).flatten()
        
        acc = accuracy_score(all_labels, all_preds)
        
        if rate == 0.0:
            baseline_acc = acc
            print(f"    Accuracy: {acc*100:.2f}% (baseline)")
        else:
            drop = (baseline_acc - acc) * 100
            print(f"    Accuracy: {acc*100:.2f}% (drop: {drop:.2f}%)")
    
    # Test Gaussian noise
    print("\n" + "="*60)
    print("4. Testing Gaussian Noise (Pose Estimation Error)")
    print("="*60)
    
    sigma_values = [0.0, 0.05, 0.10, 0.15]
    
    for sigma in sigma_values:
        print(f"\n  Noise sigma: {sigma}")
        all_preds = []
        all_labels = []
        
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            noisy_x = add_gaussian_noise(batch_x, sigma)
            
            with torch.no_grad():
                logits = model(noisy_x)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0).flatten()
        all_labels = np.concatenate(all_labels, axis=0).flatten()
        
        acc = accuracy_score(all_labels, all_preds)
        
        if sigma == 0.0:
            print(f"    Accuracy: {acc*100:.2f}% (baseline)")
        else:
            drop = (baseline_acc - acc) * 100
            print(f"    Accuracy: {acc*100:.2f}% (drop: {drop:.2f}%)")
    
    print("\n" + "="*60)
    print("✅ Experiment Complete")
    print("="*60)


if __name__ == "__main__":
    main()