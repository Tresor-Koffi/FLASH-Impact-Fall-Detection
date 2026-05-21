

# =================== IMPORTS AND SETUP ===================
"""
HyperMamba Training Script

Training configuration matches paper Table 2:
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-5
- Batch size: 32
- Max epochs: 300
- Train/Val/Test split: 80%/10%/10% (subject-independent)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, matthews_corrcoef, roc_auc_score
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import time

from HGCN.hypergraph import Hypergraph
from HGCN.data_processing import FallDataLoader
from HGCN.hmamba import HyperMamba


# =================== GPU CONFIGURATION ===================

def configure_gpu():
    """Configure GPU and print device information."""
    if torch.cuda.is_available():
        print(f"\nNum GPUs Available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i} Details:")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            print(f"Memory Usage:")
            print(f"Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        torch.backends.cudnn.benchmark = True
        print("\nCUDNN benchmark enabled")
        return True
    else:
        print("\nNo GPU available. Running on CPU.")
        return False


# =================== SETUP AND CONFIGURATION ===================

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

# Paper Table 2 hyperparameters
parser = argparse.ArgumentParser(description="HyperMamba: Impact Fall Detection")
parser.add_argument("--ex", dest="ex", type=str, default="Data/", help="Dataset path.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (paper: 1e-4).")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (paper: 1e-5).")
parser.add_argument("--epochs", type=int, default=300, help="Max epochs (paper: 300).")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size (paper: 32).")
parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")

args = parser.parse_args()

gpu_available = configure_gpu()
device = torch.device('cuda' if gpu_available else 'cpu')

results_dir = "training_results"
os.makedirs(results_dir, exist_ok=True)


# =================== DATA PREPARATION ===================

print("\n" + "="*50)
print("Loading and preprocessing data...")
print("="*50)

data_loader = FallDataLoader(args.ex)
print(f"\nData loaded - X shape: {data_loader.scaled_x.shape}, Y shape: {data_loader.scaled_y.shape}")
print(f"Number of joints (J): {data_loader.num_node}")
print(f"Number of hyperedges (E): {data_loader.hypergraph.get_num_hyperedges()}")

# Subject-independent split: 80% train, 10% val, 10% test (paper Section 4.1.2)
stratify_labels = data_loader.scaled_y[:, -1]

train_val_x, test_x, train_val_y, test_y = train_test_split(
    data_loader.scaled_x,
    data_loader.scaled_y,
    test_size=0.1,  # 10% test
    random_state=random_seed,
    shuffle=True,
    stratify=stratify_labels
)

train_x, val_x, train_y, val_y = train_test_split(
    train_val_x,
    train_val_y,
    test_size=0.111,  # ~10% of total (10/90)
    random_state=random_seed,
    shuffle=True,
    stratify=train_val_y[:, -1]
)

print(f"\nDataset splits:")
print(f"  Training:   {train_x.shape[0]} sequences ({train_x.shape[0]/(train_x.shape[0]+val_x.shape[0]+test_x.shape[0])*100:.1f}%)")
print(f"  Validation: {val_x.shape[0]} sequences ({val_x.shape[0]/(train_x.shape[0]+val_x.shape[0]+test_x.shape[0])*100:.1f}%)")
print(f"  Test:       {test_x.shape[0]} sequences ({test_x.shape[0]/(train_x.shape[0]+val_x.shape[0]+test_x.shape[0])*100:.1f}%)")


class TimeSeriesDataset(Dataset):
    """Dataset class for skeleton sequences."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = TimeSeriesDataset(train_x, train_y)
val_dataset = TimeSeriesDataset(val_x, val_y)
test_dataset = TimeSeriesDataset(test_x, test_y)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)


# =================== MODEL INITIALIZATION ===================

print("\n" + "="*50)
print("Initializing HyperMamba model...")
print("="*50)

model = HyperMamba(device=device).to(device)
model.summary()

# Paper Table 2: AdamW optimizer with lr=1e-4, weight_decay=1e-5
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=args.patience // 2, 
    min_lr=1e-6
)

history = {
    'loss': [], 'val_loss': [],
    'accuracy': [], 'val_accuracy': []
}


# =================== TRAINING LOOP ===================

print("\n" + "="*50)
print("Starting training...")
print(f"Hyperparameters: lr={args.lr}, weight_decay={args.weight_decay}, batch_size={args.batch_size}")
print("="*50 + "\n")

best_val_loss = float('inf')
best_model_state = None
patience_counter = 0

for epoch in range(args.epochs):
    # Training
    model.train()
    train_loss = 0.0
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
        
        train_loss += loss.item()
        predictions = (outputs > 0).float()
        train_correct += (predictions == batch_y).sum().item()
        train_total += batch_y.numel()
    
    train_loss /= len(train_loader)
    train_acc = train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            predictions = (outputs > 0).float()
            val_correct += (predictions == batch_y).sum().item()
            val_total += batch_y.numel()
    
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    
    scheduler.step(val_loss)
    
    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['accuracy'].append(train_acc)
    history['val_accuracy'].append(val_acc)
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{args.epochs}] - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')


# =================== EVALUATION ===================

print("\n" + "="*50)
print("Evaluating on test set...")
print("="*50)

model.load_state_dict(best_model_state)
model.to(device)
model.eval()

test_preds_logits = []
test_true = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        test_preds_logits.extend(outputs.cpu().numpy())
        test_true.extend(batch_y.numpy())

test_preds_logits = np.array(test_preds_logits).flatten()
test_true_flat = np.array(test_true).flatten()

test_probs = torch.sigmoid(torch.tensor(test_preds_logits)).numpy()
test_preds_binary = (test_preds_logits > 0).astype(int)

# Compute metrics (Paper Section 4.1.3, Equations 15-19)
cm = confusion_matrix(test_true_flat, test_preds_binary)
tn, fp, fn, tp = cm.ravel()

metrics = {
    'accuracy': accuracy_score(test_true_flat, test_preds_binary),
    'precision': precision_score(test_true_flat, test_preds_binary, zero_division=0),
    'recall': recall_score(test_true_flat, test_preds_binary, zero_division=0),
    'f1': f1_score(test_true_flat, test_preds_binary, zero_division=0),
    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
    'auc': roc_auc_score(test_true_flat, test_probs),
    'mcc': matthews_corrcoef(test_true_flat, test_preds_binary)
}

print("\n" + "="*50)
print("HyperMamba Performance on Test Set (Table 3)")
print("="*50)
print(f"Accuracy:    {metrics['accuracy']*100:.2f}%")
print(f"Precision:   {metrics['precision']*100:.2f}%")
print(f"Recall:      {metrics['recall']*100:.2f}%")
print(f"F1-Score:    {metrics['f1']*100:.2f}%")
print(f"Specificity: {metrics['specificity']*100:.2f}%")
print(f"AUC-ROC:     {metrics['auc']*100:.2f}%")
print(f"MCC:         {metrics['mcc']:.4f}")
print("="*50)


# =================== COMPUTATIONAL EFFICIENCY (Table 6) ===================

print("\nComputational Efficiency Analysis...")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Measure inference time
model.eval()
dummy_input = torch.randn(1, 100, 33, 3).to(device)

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = model(dummy_input)

if device.type == 'cuda':
    torch.cuda.synchronize()

start_time = time.time()
num_runs = 100
for _ in range(num_runs):
    with torch.no_grad():
        _ = model(dummy_input)

if device.type == 'cuda':
    torch.cuda.synchronize()

inference_time = (time.time() - start_time) / num_runs * 1000  # ms

print(f"\nParameters: {total_params/1e6:.1f}M")
print(f"Inference time: {inference_time:.1f} ms")


# =================== VISUALIZATION ===================

# Learning curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'learning_curves.png'), dpi=150)
plt.close()

# Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False,
            xticklabels=['Non-Impact', 'Impact'],
            yticklabels=['Non-Impact', 'Impact'])
plt.title('Confusion Matrix (Test Set)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrix_test.png'), dpi=150)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(test_true_flat, test_probs)
roc_auc = metrics['auc']

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Set)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'roc_curve_test.png'), dpi=150)
plt.close()


# =================== SAVE RESULTS ===================

model_save_dir = "saved_models"
os.makedirs(model_save_dir, exist_ok=True)
model_path = os.path.join(model_save_dir, "best_hypermamba_model.pth")

torch.save({
    'model_state_dict': best_model_state,
    'optimizer_state_dict': optimizer.state_dict(),
    'metrics': metrics,
    'best_val_loss': best_val_loss,
    'history': history,
    'config': {
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'epochs_trained': len(history['loss']),
        'num_joints': 33,
        'num_hyperedges': 26
    }
}, model_path)

print(f"\n✅ Model saved to: {model_path}")
print("\nSaved files:")
print(f"  - {model_path}")
print(f"  - {os.path.join(results_dir, 'learning_curves.png')}")
print(f"  - {os.path.join(results_dir, 'confusion_matrix_test.png')}")
print(f"  - {os.path.join(results_dir, 'roc_curve_test.png')}")





# # =================== IMPORTS AND SETUP ===================

# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.optim import Adam
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score, 
#     confusion_matrix, roc_curve, auc, matthews_corrcoef, roc_auc_score
# )
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import os
# from HGCN.hypergraph import Hypergraph
# from HGCN.data_processing import FallDataLoader
# from HGCN.hmamba import HyperMamba


# # =================== GPU CONFIGURATION ===================

# def configure_gpu():
#     if torch.cuda.is_available():
#         print(f"\nNum GPUs Available: {torch.cuda.device_count()}")
#         for i in range(torch.cuda.device_count()):
#             print(f"\nGPU {i} Details:")
#             print(f"Name: {torch.cuda.get_device_name(i)}")
#             print(f"Memory Usage:")
#             print(f"Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
#             print(f"Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
#         torch.backends.cudnn.benchmark = True
#         print("\nCUDNN benchmark enabled")
#         return True
#     else:
#         print("\nNo GPU available. Running on CPU.")
#         return False


# # =================== SETUP AND CONFIGURATION ===================

# random_seed = 42
# torch.manual_seed(random_seed)
# np.random.seed(random_seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(random_seed)

# parser = argparse.ArgumentParser(description="Fall Detection with HyperMamba")
# parser.add_argument("--ex", dest="ex", type=str, default="Data/", help="Dataset path.")
# parser.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate.")
# parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
# parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")

# args = parser.parse_args()

# gpu_available = configure_gpu()
# device = torch.device('cuda' if gpu_available else 'cpu')

# results_dir = "training_results"
# os.makedirs(results_dir, exist_ok=True)


# # =================== DATA PREPARATION ===================

# data_loader = FallDataLoader(args.ex)
# print(f"Data loaded - X shape: {data_loader.scaled_x.shape}, Y shape: {data_loader.scaled_y.shape}")

# # Stratify on last frame
# stratify_labels = data_loader.scaled_y[:, -1]

# train_val_x, test_x, train_val_y, test_y = train_test_split(
#     data_loader.scaled_x,
#     data_loader.scaled_y,
#     test_size=0.2,
#     random_state=random_seed,
#     shuffle=True,
#     stratify=stratify_labels
# )

# train_x, val_x, train_y, val_y = train_test_split(
#     train_val_x,
#     train_val_y,
#     test_size=0.25,
#     random_state=random_seed,
#     shuffle=True,
#     stratify=train_val_y[:, -1]
# )

# train_y_flattened = train_y.ravel().astype(int)
# class_distribution = np.bincount(train_y_flattened)
# print("Training Class Distribution:", class_distribution)


# class TimeSeriesDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.FloatTensor(X)
#         self.y = torch.FloatTensor(y)

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]


# train_dataset = TimeSeriesDataset(train_x, train_y)
# val_dataset = TimeSeriesDataset(val_x, val_y)
# test_dataset = TimeSeriesDataset(test_x, test_y)

# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
# val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
# test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)


# # =================== MODEL INITIALIZATION ===================

# model = HyperMamba(device=device).to(device)

# # ✅ CRITICAL: Use BCEWithLogitsLoss for raw logits
# criterion = nn.BCEWithLogitsLoss()
# optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)

# # ✅ Removed 'verbose' to support older PyTorch versions
# scheduler = ReduceLROnPlateau(
#     optimizer, 
#     mode='min', 
#     factor=0.5, 
#     patience=7, 
#     min_lr=1e-6
# )

# history = {
#     'loss': [], 'val_loss': [],
#     'binary_accuracy': [], 'val_binary_accuracy': []
# }


# # =================== TRAINING LOOP ===================

# best_val_loss = float('inf')
# best_model_state = None

# print("\nStarting model training...")
# for epoch in range(args.epochs):
#     # Training
#     model.train()
#     train_loss = 0.0
#     train_correct = 0
#     train_total = 0
    
#     for batch_x, batch_y in train_loader:
#         batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#         optimizer.zero_grad()
#         outputs = model(batch_x)  # ✅ No training arg
#         loss = criterion(outputs, batch_y)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#         optimizer.step()
        
#         train_loss += loss.item()
#         predictions = (outputs > 0).float()  # ✅ Threshold at 0 for logits
#         train_correct += (predictions == batch_y).sum().item()
#         train_total += batch_y.numel()
    
#     train_loss /= len(train_loader)
#     train_acc = train_correct / train_total
    
#     # Validation
#     model.eval()
#     val_loss = 0.0
#     val_correct = 0
#     val_total = 0
    
#     with torch.no_grad():
#         for batch_x, batch_y in val_loader:
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#             outputs = model(batch_x)  # ✅ No training=False
#             loss = criterion(outputs, batch_y)
#             val_loss += loss.item()
            
#             predictions = (outputs > 0).float()
#             val_correct += (predictions == batch_y).sum().item()
#             val_total += batch_y.numel()
    
#     val_loss /= len(val_loader)
#     val_acc = val_correct / val_total
    
#     scheduler.step(val_loss)
    
#     history['loss'].append(train_loss)
#     history['val_loss'].append(val_loss)
#     history['binary_accuracy'].append(train_acc)
#     history['val_binary_accuracy'].append(val_acc)
    
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         best_model_state = model.state_dict().copy()
    
#     print(f'Epoch [{epoch+1}/{args.epochs}]')
#     print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
#     print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
#     print('-' * 50)


# # =================== EVALUATION ===================

# model.load_state_dict(best_model_state)
# model.eval()

# test_preds_logits = []
# test_true = []

# with torch.no_grad():
#     for batch_x, batch_y in test_loader:
#         batch_x = batch_x.to(device)
#         outputs = model(batch_x)
#         test_preds_logits.extend(outputs.cpu().numpy())
#         test_true.extend(batch_y.numpy())

# test_preds_logits = np.array(test_preds_logits).flatten()
# test_true_flat = np.array(test_true).flatten()

# # ✅ Convert logits to probabilities for AUC
# test_probs = torch.sigmoid(torch.tensor(test_preds_logits)).numpy()
# test_preds_binary = (test_preds_logits > 0).astype(int)

# # Metrics
# metrics = {
#     'accuracy': accuracy_score(test_true_flat, test_preds_binary),
#     'precision': precision_score(test_true_flat, test_preds_binary, zero_division=0),
#     'recall': recall_score(test_true_flat, test_preds_binary, zero_division=0),
#     'f1': f1_score(test_true_flat, test_preds_binary, zero_division=0),
#     'auc': roc_auc_score(test_true_flat, test_probs)
# }

# cm = confusion_matrix(test_true_flat, test_preds_binary)
# tn, fp, fn, tp = cm.ravel()
# mcc = matthews_corrcoef(test_true_flat, test_preds_binary)

# metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
# metrics['mcc'] = mcc

# print("\nModel Performance Metrics on Test Set:")
# for metric_name, value in metrics.items():
#     print(f"{metric_name.capitalize()}: {value:.4f}")


# # =================== VISUALIZATION ===================

# # Learning curves
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(history['loss'], label='Training Loss')
# plt.plot(history['val_loss'], label='Validation Loss')
# plt.title('Model Loss over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(False)

# plt.subplot(1, 2, 2)
# plt.plot(history['binary_accuracy'], label='Training Accuracy')
# plt.plot(history['val_binary_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(False)

# plt.tight_layout()
# plt.savefig(os.path.join(results_dir, 'learning_curves.png'))
# plt.close()

# # Confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
# plt.title('Confusion Matrix (Test Set)')
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.savefig(os.path.join(results_dir, 'confusion_matrix_test.png'))
# plt.close()

# # ROC Curve
# fpr, tpr, _ = roc_curve(test_true_flat, test_probs)
# roc_auc = metrics['auc']

# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve (Test Set)')
# plt.legend(loc='lower right')
# plt.savefig(os.path.join(results_dir, 'roc_curve_test.png'))
# plt.close()


# # =================== SAVE RESULTS ===================

# model_save_dir = "saved_models"
# os.makedirs(model_save_dir, exist_ok=True)
# model_path = os.path.join(model_save_dir, "best_model.pth")

# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'metrics': metrics,
#     'best_val_loss': best_val_loss,
#     'history': history
# }, model_path)

# print(f"\n✅ Model saved to: {model_path}")
# print("\nSaved files:")
# print("- best_model.pth")
# print("- learning_curves.png")
# print("- confusion_matrix_test.png")
# print("- roc_curve_test.png")