#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_augmentation import balance_data
import os
from sklearn.metrics import accuracy_score
os.makedirs('EEGNet_Sup_Class', exist_ok=True)
class EEGNet(nn.Module):
    def __init__(self, num_channels=8, num_timepoints=226, dropout_rate=0.5, emb_dim=128, num_classes=4):
        super(EEGNet, self).__init__()
        # block 1
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8, momentum=0.01, affine=True, eps=1e-3, track_running_stats=False)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(num_channels, 1), groups=8, bias=False),
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
        self.feature_dim = 16 * ((num_timepoints + 3) // 4) * 2

        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # Input: (B, C, T)
        x = x.unsqueeze(1)  # → (B, 1, C, T)
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
        # mask out self-comparisons
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask

        # compute log-softmax
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # loss
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        loss = -mean_log_prob_pos.mean()
        return loss
#%%

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

#%%

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
    acc = accuracy_score(all_labels, all_preds)
    return acc

#%%%%%%%
from preprocessing import *
# from EEGNet_implementation import EEGNet, SupConLoss, train_supcon

import os
import numpy as np
from matplotlib import pyplot as plt
import random
import torch       
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import WeightedRandomSampler
 
import torch
# import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import WeightedRandomSampler

# ensure reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

subject_ids = [1, 2, 3, 4, 5]  # or however many subjects you have
folder = 'p300-speller'

sid = 1
filepath = os.path.join(folder, f"S{sid}.mat")

print(f"Loading subject {sid} from {filepath}")

epochs_S1 = get_epochs_from_file('p300-speller/S1.mat')
epochs_S1.pick_types(eeg=True)

epochs_S2 = get_epochs_from_file('p300-speller/S2.mat')
epochs_S2.pick_types(eeg=True)

epochs_S3 = get_epochs_from_file('p300-speller/S3.mat')
epochs_S3.pick_types(eeg=True)

epochs_S4 = get_epochs_from_file('p300-speller/S4.mat')
epochs_S4.pick_types(eeg=True)

epochs_S5 = get_epochs_from_file('p300-speller/S5.mat')
epochs_S5.pick_types(eeg=True)

#%%
# Standardize X: (trials, channels, samples)
def normalize_subject(X):
    # Z-score across trials, separately for each channel
    trials, channels, samples = X.shape
    X_reshaped = X.transpose(1, 0, 2).reshape(channels, -1)
    X_reshaped = (X_reshaped - X_reshaped.mean(axis=1, keepdims=True)) / (X_reshaped.std(axis=1, keepdims=True) + 1e-6)
    return X_reshaped.reshape(channels, trials, samples).transpose(1, 0, 2)
ratio_augment = 0.5
X_S1,y_S1,times_S1 = balance_data(epochs_S1,ratio=ratio_augment) # format is in (trials, channels, samples)
X_S1 = normalize_subject(X_S1)
X_train_S1, X_val_S1, y_train_S1, y_val_S1 = train_test_split(X_S1, y_S1, test_size=0.25, random_state=42, stratify=y_S1)

X_S2, y_S2, times_S2 = balance_data(epochs_S2,ratio=ratio_augment)  # format is in (trials, channels, samples)
X_S2 = normalize_subject(X_S2)
X_train_S2, X_val_S2, y_train_S2, y_val_S2 = train_test_split(X_S2, y_S2, test_size=0.25, random_state=42, stratify=y_S2)

X_S3, y_S3, times_S3 = balance_data(epochs_S3,ratio=ratio_augment)  # format is in (trials, channels, samples)
X_S3 = normalize_subject(X_S3)
X_train_S3, X_val_S3, y_train_S3, y_val_S3 = train_test_split(X_S3, y_S3, test_size=0.25, random_state=42, stratify=y_S3)

X_S4, y_S4, times_S4 = balance_data(epochs_S4,ratio=ratio_augment)  # format is in (trials, channels, samples)
X_S4 = normalize_subject(X_S4)
X_train_S4, X_val_S4, y_train_S4, y_val_S4 = train_test_split(X_S4, y_S4, test_size=0.25, random_state=42, stratify=y_S4)

X_S5, y_S5, times_S5 = balance_data(epochs_S5,ratio=ratio_augment)  # format is in (trials, channels, samples)
X_S5 = normalize_subject(X_S5)
X_train_S5, X_val_S5, y_train_S5, y_val_S5 = train_test_split(X_S5, y_S5, test_size=0.25, random_state=42, stratify=y_S5)

def get_sampler(y):
    unique_classes = torch.unique(y)
    class_to_index = {cls.item(): idx for idx, cls in enumerate(unique_classes)}
    
    class_sample_count = torch.tensor([(y == cls).sum() for cls in unique_classes], dtype=torch.float)
    weights = 1.0 / class_sample_count

    sample_weights = torch.tensor([weights[class_to_index[int(label.item())]] for label in y])

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

# stack X_S2 and X_S1
X_train = np.vstack((X_train_S1, X_train_S2, X_train_S3, X_train_S4, X_train_S5)) # (600, 8, 226)
X_val = np.vstack((X_val_S1, X_val_S2, X_val_S3, X_val_S4, X_val_S5)) # (400, 8, 226)
y_train = np.hstack((y_train_S1, y_train_S2, y_train_S3, y_train_S4, y_train_S5)) # (600,)
y_val = np.hstack((y_val_S1, y_val_S2, y_val_S3, y_val_S4, y_val_S5)) # (400,)

X_val.shape, y_val.shape
# X_train.shape, y_train.shape
# X_val_S1.shape, X_val_S2.shape, X_val_S3.shape, X_val_S4.shape, X_val_S5.shape
# y_val_S1.shape, y_val_S2.shape, y_val_S3.shape, y_val_S4.shape, y_val_S5.shape
#%%
X_train = torch.tensor(X_train, dtype=torch.float32) # (600, 8, 226)
y_train = torch.tensor(y_train, dtype=torch.long) # (600,)
y_train = torch.tensor((y_train == 1).long())

X_val = torch.tensor(X_val, dtype=torch.float32) # (400, 8, 226)
y_val = torch.tensor(y_val, dtype=torch.long) # (400,)
y_val = torch.tensor((y_val == 1).long())

train_sampler = get_sampler(y_train)
val_sampler = get_sampler(y_val)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

# create dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size= 128, sampler=train_sampler)
val_loader = DataLoader(dataset=val_dataset, batch_size= 128, sampler=val_sampler)
#%%
print(y_val)
#%%
from tqdm import tqdm#.tqdm
for batch in tqdm(val_loader):
    inputs, targets = batch
    # count targets = 0 and 1
    targets = targets.numpy()
    count_0 = np.sum(targets == 0)
    count_1 = np.sum(targets == 1)
    print(f"Count of 0: {count_0}, Count of 1: {count_1}")
      
#%%
############################# EEGNet portion ##################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EEGNet(num_channels=8, num_timepoints=226, emb_dim=64)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
supcon_loss = SupConLoss(temperature=0.07).to(device)
ce_loss = nn.CrossEntropyLoss().to(device)

#%%
# def decay_alpha(epoch, initial_alpha=0.7, final_alpha=0.3, num_epochs=20):
#     # Linear decay of alpha
#     return initial_alpha - (initial_alpha - final_alpha) * (epoch / num_epochs)

num_epochs = 50
max_acc = 0
initial_alpha = 0.7  # Start with a higher weight for SupCon loss
final_alpha = 0.3  #

for epoch in range(num_epochs):
    # alpha = decay_alpha(epoch, initial_alpha, final_alpha, num_epochs)
    alpha = 0.5

    train_loss = train_supcon_ce(model, train_loader, optimizer, supcon_loss, ce_loss, device, alpha=alpha)
    
    val_acc = evaluate(model, val_loader, device)
    print(f"\rEpoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}", end="", flush=True)

    if val_acc > max_acc:
        max_acc = val_acc
        torch.save(model.state_dict(), "EEGNet_Sup_Class/best_model_64.pth")
        print(f"\rModel saved with accuracy {max_acc:.4f}  - ALL", flush=False)

#%% FINETUNING FOR SPECIFIC SUBJECT
## S1
# finetune dataset
X_train_S1 = torch.tensor(X_train_S1, dtype=torch.float32) # (600, 8, 226)
y_train_S1 = torch.tensor(y_train_S1, dtype=torch.long) # (600,)
y_train_S1 = torch.tensor((y_train_S1 == 1).long())

X_val_S1 = torch.tensor(X_val_S1, dtype=torch.float32) # (400, 8, 226)
y_val_S1 = torch.tensor(y_val_S1, dtype=torch.long) # (400,)
y_val_S1 = torch.tensor((y_val_S1 == 1).long())

train_sampler_S1 = get_sampler(y_train_S1)
val_sampler_S1 = get_sampler(y_val_S1)

train_dataset_S1 = TensorDataset(X_train_S1, y_train_S1)
val_dataset_S1 = TensorDataset(X_val_S1, y_val_S1)

# create dataloader
train_loader_S1 = DataLoader(dataset=train_dataset_S1, batch_size= 128, sampler=train_sampler_S1)
val_loader_S1 = DataLoader(dataset=val_dataset_S1, batch_size= 128, sampler=val_sampler_S1)

num_epochs = 50
max_acc_S1 = 0
initial_alpha = 0.7  # Start with a higher weight for SupCon loss
final_alpha = 0.3  #

model_S1 = EEGNet(num_channels=8, num_timepoints=226, emb_dim=64)
model_S1.to(device)
model_S1.load_state_dict(torch.load("EEGNet_Sup_Class/best_model_64.pth"))

optimizer = optim.Adam(model.parameters(), lr=1e-5)  # lower learning rate for fine-tuning

for epoch in range(num_epochs):
    # alpha = decay_alpha(epoch, initial_alpha, final_alpha, num_epochs)
    alpha = 0.5

    train_loss = train_supcon_ce(model_S1, train_loader_S1, optimizer, supcon_loss, ce_loss, device, alpha=alpha)
    
    val_acc = evaluate(model_S1, val_loader_S1, device)
    print(f"\rEpoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}", end="", flush=True)

    if val_acc > max_acc_S1:
        max_acc_S1 = val_acc
        torch.save(model.state_dict(), "EEGNet_Sup_Class/best_model_64_S1.pth")
        print(f"\rModel saved with accuracy {max_acc_S1:.4f} - S1", flush=False)

#%%
## S2
# finetune dataset
X_train_S2 = torch.tensor(X_train_S2, dtype=torch.float32) # (600, 8, 226)
y_train_S2 = torch.tensor(y_train_S2, dtype=torch.long) # (600,)
y_train_S2 = torch.tensor((y_train_S2 == 1).long())

X_val_S2 = torch.tensor(X_val_S2, dtype=torch.float32) # (400, 8, 226)
y_val_S2 = torch.tensor(y_val_S2, dtype=torch.long) # (400,)
y_val_S2 = torch.tensor((y_val_S2 == 1).long())

train_sampler_S2 = get_sampler(y_train_S2)
val_sampler_S2 = get_sampler(y_val_S2)

train_dataset_S2 = TensorDataset(X_train_S2, y_train_S2)
val_dataset_S2 = TensorDataset(X_val_S2, y_val_S2)

# create dataloader
train_loader_S2 = DataLoader(dataset=train_dataset_S2, batch_size= 128, sampler=train_sampler_S2)
val_loader_S2 = DataLoader(dataset=val_dataset_S2, batch_size= 128, sampler=val_sampler_S2)

max_acc_S2 = 0
initial_alpha = 0.7  # Start with a higher weight for SupCon loss
final_alpha = 0.3  #

model_S2 = EEGNet(num_channels=8, num_timepoints=226, emb_dim=64)
model_S2.to(device)
model_S2.load_state_dict(torch.load("EEGNet_Sup_Class/best_model_64.pth"))

optimizer = optim.Adam(model.parameters(), lr=1e-5)  # lower learning rate for fine-tuning

for epoch in range(num_epochs):
    # alpha = decay_alpha(epoch, initial_alpha, final_alpha, num_epochs)
    alpha = 0.5

    train_loss = train_supcon_ce(model_S2, train_loader_S2, optimizer, supcon_loss, ce_loss, device, alpha=alpha)
    
    val_acc = evaluate(model_S2, val_loader_S2, device)
    print(f"\rEpoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}", end="", flush=True)

    if val_acc > max_acc_S2:
        max_acc_S2 = val_acc
        torch.save(model.state_dict(), "EEGNet_Sup_Class/best_model_64_S2.pth")
        print(f"\rModel saved with accuracy {max_acc_S2:.4f} - S2", flush=False)

# %%
## S3
# finetune dataset
X_train_S3 = torch.tensor(X_train_S3, dtype=torch.float32) # (600, 8, 226)
y_train_S3 = torch.tensor(y_train_S3, dtype=torch.long) # (600,)
y_train_S3 = torch.tensor((y_train_S3 == 1).long())

X_val_S3 = torch.tensor(X_val_S3, dtype=torch.float32) # (400, 8, 226)
y_val_S3 = torch.tensor(y_val_S3, dtype=torch.long) # (400,)
y_val_S3 = torch.tensor((y_val_S3 == 1).long())

train_sampler_S3 = get_sampler(y_train_S3)
val_sampler_S3 = get_sampler(y_val_S3)

train_dataset_S3 = TensorDataset(X_train_S3, y_train_S3)
val_dataset_S3 = TensorDataset(X_val_S3, y_val_S3)

# create dataloader
train_loader_S3 = DataLoader(dataset=train_dataset_S3, batch_size= 128, sampler=train_sampler_S3)
val_loader_S3 = DataLoader(dataset=val_dataset_S3, batch_size= 128, sampler=val_sampler_S3)

max_acc_S3 = 0
initial_alpha = 0.7  # Start with a higher weight for SupCon loss
final_alpha = 0.3  #

model_S3 = EEGNet(num_channels=8, num_timepoints=226, emb_dim=64)
model_S3.to(device)
model_S3.load_state_dict(torch.load("EEGNet_Sup_Class/best_model_64.pth"))

optimizer = optim.Adam(model.parameters(), lr=1e-5)  # lower learning rate for fine-tuning

for epoch in range(num_epochs):
    # alpha = decay_alpha(epoch, initial_alpha, final_alpha, num_epochs)
    alpha = 0.5

    train_loss = train_supcon_ce(model_S3, train_loader_S3, optimizer, supcon_loss, ce_loss, device, alpha=alpha)
    
    val_acc = evaluate(model_S3, val_loader_S3, device)
    print(f"\rEpoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}", end ="", flush=True)

    if val_acc > max_acc_S3:
        max_acc_S3 = val_acc
        torch.save(model.state_dict(), "EEGNet_Sup_Class/best_model_64_S3.pth")
        print(f"\rModel saved with accuracy {max_acc_S3:.4f} - S3", flush=False)

#%%
## S4
# finetune dataset
X_train_S4 = torch.tensor(X_train_S4, dtype=torch.float32) # (600, 8, 226)
y_train_S4 = torch.tensor(y_train_S4, dtype=torch.long) # (600,)
y_train_S4 = torch.tensor((y_train_S4 == 1).long())

X_val_S4 = torch.tensor(X_val_S4, dtype=torch.float32) # (400, 8, 226)
y_val_S4 = torch.tensor(y_val_S4, dtype=torch.long) # (400,)
y_val_S4 = torch.tensor((y_val_S4 == 1).long())

train_sampler_S4 = get_sampler(y_train_S4)
val_sampler_S4 = get_sampler(y_val_S4)

train_dataset_S4 = TensorDataset(X_train_S4, y_train_S4)
val_dataset_S4 = TensorDataset(X_val_S4, y_val_S4)

# create dataloader
train_loader_S4 = DataLoader(dataset=train_dataset_S4, batch_size= 128, sampler=train_sampler_S4)
val_loader_S4 = DataLoader(dataset=val_dataset_S4, batch_size= 128, sampler=val_sampler_S4)

max_acc_S4 = 0
initial_alpha = 0.7  # Start with a higher weight for SupCon loss
final_alpha = 0.3  #

model_S4 = EEGNet(num_channels=8, num_timepoints=226, emb_dim=64)
model_S4.to(device)
model_S4.load_state_dict(torch.load("EEGNet_Sup_Class/best_model_64.pth"))

optimizer = optim.Adam(model.parameters(), lr=1e-5)  # lower learning rate for fine-tuning

for epoch in range(num_epochs):
    # alpha = decay_alpha(epoch, initial_alpha, final_alpha, num_epochs)
    alpha = 0.5

    train_loss = train_supcon_ce(model_S4, train_loader_S4, optimizer, supcon_loss, ce_loss, device, alpha=alpha)
    
    val_acc = evaluate(model_S4, val_loader_S4, device)
    print(f"\rEpoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}", end="", flush=True)

    if val_acc > max_acc_S4:
        max_acc_S4 = val_acc
        torch.save(model.state_dict(), "EEGNet_Sup_Class/best_model_64_S4.pth")
        print(f"\rModel saved with accuracy {max_acc_S4:.4f} - S4", end="", flush=False)

#%%
## S5
# finetune dataset
X_train_S5 = torch.tensor(X_train_S5, dtype=torch.float32) # (600, 8, 226)
y_train_S5 = torch.tensor(y_train_S5, dtype=torch.long) # (600,)
y_train_S5 = torch.tensor((y_train_S5 == 1).long())

X_val_S5 = torch.tensor(X_val_S5, dtype=torch.float32) # (400, 8, 226)
y_val_S5 = torch.tensor(y_val_S5, dtype=torch.long) # (400,)
y_val_S5 = torch.tensor((y_val_S5 == 1).long())

train_sampler_S5 = get_sampler(y_train_S5)
val_sampler_S5 = get_sampler(y_val_S5)

train_dataset_S5 = TensorDataset(X_train_S5, y_train_S5)
val_dataset_S5 = TensorDataset(X_val_S5, y_val_S5)

# create dataloader
train_loader_S5 = DataLoader(dataset=train_dataset_S5, batch_size= 128, sampler=train_sampler_S5)
val_loader_S5 = DataLoader(dataset=val_dataset_S5, batch_size= 128, sampler=val_sampler_S5)

max_acc_S5 = 0
initial_alpha = 0.7  # Start with a higher weight for SupCon loss
final_alpha = 0.3  #

model_S5 = EEGNet(num_channels=8, num_timepoints=226, emb_dim=64)
model_S5.to(device)
model_S5.load_state_dict(torch.load("EEGNet_Sup_Class/best_model_64.pth"))

optimizer = optim.Adam(model.parameters(), lr=1e-5)  # lower learning rate for fine-tuning

for epoch in range(num_epochs):
    # alpha = decay_alpha(epoch, initial_alpha, final_alpha, num_epochs)
    alpha = 0.5

    train_loss = train_supcon_ce(model_S5, train_loader_S5, optimizer, supcon_loss, ce_loss, device, alpha=alpha)
    
    val_acc = evaluate(model_S5, val_loader_S5, device)
    print(f"\rEpoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}", end="", flush=True)

    if val_acc > max_acc_S5:
        max_acc_S5 = val_acc
        torch.save(model.state_dict(), "EEGNet_Sup_Class/best_model_64_S5.pth")
        print(f"\rModel saved with accuracy {max_acc_S5:.4f} - S4", flush=False)
# %%
 