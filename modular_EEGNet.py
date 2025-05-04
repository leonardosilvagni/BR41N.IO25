import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_augmentation import balance_data
from preprocessing import get_epochs_from_file

# Ensure output directory exists
os.makedirs('EEGNet_Sup_Class', exist_ok=True)

# ------------------------ Models and Loss -----------------------
class EEGNet(nn.Module):
    def __init__(self, num_channels=8, num_timepoints=226, dropout_rate=0.5, emb_dim=128, num_classes=4):
        super(EEGNet, self).__init__()
        # block 1
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8, momentum=0.01, affine=True, eps=1e-3, track_running_stats=False)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(8, 1), groups=8, bias=False),
            nn.BatchNorm2d(16, track_running_stats=False),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.Dropout(dropout_rate)
        )
        self.flatten = nn.Flatten()
        # Calculate feature dimension
        self.feature_dim = 16 * ((num_timepoints + 3) // 4) * 2
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.flatten(x)
        emb = self.projector(x)
        emb = F.normalize(emb, dim=1)
        logits = self.classifier(emb)
        return emb, logits

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask

        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        loss = -mean_log_prob_pos.mean()
        return loss

# ------------------------ Utility Functions -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_sampler(y):
    unique_classes = torch.unique(y)
    class_to_index = {cls.item(): idx for idx, cls in enumerate(unique_classes)}
    class_sample_count = torch.tensor([(y == cls).sum() for cls in unique_classes], dtype=torch.float)
    weights = 1.0 / class_sample_count
    sample_weights = torch.tensor([weights[class_to_index[int(label.item())]] for label in y])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def normalize_subject(X):
    trials, channels, samples = X.shape
    X_reshaped = X.transpose(1, 0, 2).reshape(channels, -1)
    X_reshaped = (X_reshaped - X_reshaped.mean(axis=1, keepdims=True)) / (X_reshaped.std(axis=1, keepdims=True) + 1e-6)
    return X_reshaped.reshape(channels, trials, samples).transpose(1, 0, 2)

# ------------------------ Data Preparation -----------------------

def prepare_general_data(ratio_augment):
    """
    Loads and prepares data across subjects S1 to S5.
    Returns train_loader, val_loader.
    """
    set_seed(42)
    subject_list = ['S1.mat', 'S2.mat', 'S3.mat', 'S4.mat', 'S5.mat']
    X_trains, X_vals, y_trains, y_vals = [], [], [], []
    
    folder = 'p300-speller'
    for subj in subject_list:
        epochs = get_epochs_from_file(os.path.join(folder, subj))
        epochs.pick_types(eeg=True)
        X = epochs.get_data() * 1e3
        y = epochs.events[:, -1]
        X = normalize_subject(X)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        X_train, y_train = balance_data(X_train, y_train, ratio=ratio_augment)
        X_trains.append(X_train)
        X_vals.append(X_val)
        y_trains.append(y_train)
        y_vals.append(y_val)
    
    X_train = np.vstack(X_trains)
    X_val = np.vstack(X_vals)
    y_train = np.hstack(y_trains)
    y_val = np.hstack(y_vals)

    # Binary classification: converting labels (example: 1==1)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor((torch.tensor(y_train, dtype=torch.long)==1).long())
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor((torch.tensor(y_val, dtype=torch.long)==1).long())

    train_sampler = get_sampler(y_train)
    val_sampler = get_sampler(y_val)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, sampler=train_sampler)
    val_loader = DataLoader(dataset=val_dataset, batch_size=128, sampler=val_sampler)
    
    return train_loader, val_loader

def prepare_finetuning_data(subject, ratio_augment):
    """
    Loads and prepares data for a specific subject (e.g., "S1.mat").
    Returns train_loader, val_loader.
    """
    set_seed(42)
    folder = 'p300-speller'
    subj_file = os.path.join(folder, f"{subject}.mat")
    epochs = get_epochs_from_file(subj_file)
    epochs.pick_types(eeg=True)
    X, y, _ = balance_data(epochs, ratio=ratio_augment)
    X = normalize_subject(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor((torch.tensor(y_train, dtype=torch.long)==1).long())
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor((torch.tensor(y_val, dtype=torch.long)==1).long())

    train_sampler = get_sampler(y_train)
    val_sampler = get_sampler(y_val)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, sampler=train_sampler)
    val_loader = DataLoader(dataset=val_dataset, batch_size=128, sampler=val_sampler)
    
    return train_loader, val_loader

# ------------------------ Training Functions -----------------------

def train_supcon_ce(model, loader, optimizer, supcon_loss, ce_loss, device, alpha=0.5):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        emb, logits = model(X)
        loss_supcon = supcon_loss(emb, y)
        loss_ce = ce_loss(logits, y)
        loss = alpha * loss_supcon + (1 - alpha) * loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            _, logits = model(X)
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(y)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return accuracy_score(all_labels, all_preds)

def train_general_model(ratio_augment, num_epochs=50, emb_dim=64, device=None):
    """
    Trains the general model using the combined data from all subjects.
    Saves model at EEGNet_Sup_Class/best_model_64.pth
    Returns best validation accuracy.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = prepare_general_data(ratio_augment)
    
    model = EEGNet(num_channels=8, num_timepoints=226, emb_dim=emb_dim)
    model.to(device)
    model.load_state_dict(torch.load("EEGNet_Sup_Class/best_model_64.pth", map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    supcon_loss = SupConLoss(temperature=0.07).to(device)
    ce_loss = nn.CrossEntropyLoss().to(device)
    
    max_acc = 0
    for epoch in range(num_epochs):
        alpha = 0.5  # you can adjust or decay this as needed
        train_loss = train_supcon_ce(model, train_loader, optimizer, supcon_loss, ce_loss, device, alpha=alpha)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > max_acc:
            max_acc = val_acc
            torch.save(model.state_dict(), "EEGNet_Sup_Class/best_model_64.pth")
            print(f" --> Model saved with accuracy {max_acc:.4f}")
    return max_acc

def finetune_subject(subject, ratio_augment, num_epochs=50, emb_dim=64, device=None):
    """
    Finetunes the model for a specific subject (e.g., subject='S1').
    Loads the general model weights from EEGNet_Sup_Class/best_model_64.pth.
    Saves the finetuned model to EEGNet_Sup_Class/best_model_64_{subject}.pth.
    Returns best validation accuracy for the subject.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = prepare_finetuning_data(subject, ratio_augment)
    
    model = EEGNet(num_channels=8, num_timepoints=226, emb_dim=emb_dim)
    model.to(device)
    # Load general model weights
    model.load_state_dict(torch.load("EEGNet_Sup_Class/best_model_64.pth", map_location=device))
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    supcon_loss = SupConLoss(temperature=0.07).to(device)
    ce_loss = nn.CrossEntropyLoss().to(device)
    
    max_acc = 0
    for epoch in range(num_epochs):
        alpha = 0.5
        train_loss = train_supcon_ce(model, train_loader, optimizer, supcon_loss, ce_loss, device, alpha=alpha)
        val_acc = evaluate(model, val_loader, device)
        print(f"Subject {subject} Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > max_acc:
            max_acc = val_acc
            save_path = f"EEGNet_Sup_Class/best_model_64_{subject}.pth"
            torch.save(model.state_dict(), save_path)
            print(f" --> Model for {subject} saved with accuracy {max_acc:.4f}")
    return max_acc

def experiment_ratio_augment(ratio_values, num_epochs=50, emb_dim=64, device=None):
    """
    Experiments with different values of ratio_augment.
    For each ratio value, runs the general training and prints the best validation accuracy.
    """
    results = {}
    for ratio in ratio_values:
        print(f"\n--- Experimenting with ratio_augment = {ratio} ---")
        acc = train_general_model(ratio, num_epochs=num_epochs, emb_dim=emb_dim, device=device)
        results[ratio] = acc
        print(f"ratio_augment: {ratio}  ->  Best Val Acc: {acc:.4f}")
    return results

# ------------------------ Main Driver -----------------------
if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Example: Run general training with a specific ratio
    general_acc = train_general_model(ratio_augment=0.7, num_epochs=30, emb_dim=64, device=device)
    print(f"\nGeneral Model Best Validation Accuracy: {general_acc:.4f}")
    
    # Example: Fine-tune for subject S1
    subjects = ['S1', 'S2', 'S3', 'S4', 'S5']
    sbj_accs = {}
    for subject in subjects:
        subj_acc = finetune_subject(subject, ratio_augment=0.7, num_epochs=25, emb_dim=64, device=device)
        print(f"\nSubject {subject} Fine-tuning Best Validation Accuracy: {subj_acc:.4f}")
        sbj_accs[subject] = subj_acc
    formatted_sbj_accs = {k: f"{v:.4f}" for k, v in sbj_accs.items()}
    print("\nSubject Accuracies, 25 epochs finetuning:", formatted_sbj_accs, "\nGeneral model accuracy, across subjects:", f"{general_acc:.4f}")
    
    # Example: Experiment with different ratio_augment values
    #ratio_list = [0.7, 0.8, 1]
    #exp_results = experiment_ratio_augment(ratio_list, num_epochs=5, emb_dim=64, device=device)
    #print("\nExperiment Results:", exp_results)