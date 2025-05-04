#%%
from preprocessing import *
from EEGNet_Sup_Class.model.EEGNet_Sup_class import *
from EEGNet_Sup_Class.utils.utils import *

# import os
import numpy as np
from matplotlib import pyplot as plt
# import random
import torch       
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier 
# from sklearn.linear_model import LogisticRegression
# from torch.utils.data import DataLoader, TensorDataset
# from torch.utils.data import WeightedRandomSampler
 
import torch
import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# from torch.utils.data import WeightedRandomSampler
# from torch.utils.data import Sampler

import warnings
warnings.filterwarnings('ignore')

##
set_seed(42)

subject_ids = [1, 2, 3, 4, 5]  # or however many subjects you have
folder = 'p300-speller'
ratio_augment = 0.5

train_loader, val_loader = get_dataloaders(subject_ids, folder=folder, ratio_augment=ratio_augment)

#%% balanced data sanity check
print("Sanity check for balanced data:")
for batch in val_loader:
    inputs, targets = batch
    # count targets = 0 and 1
    targets = targets.numpy()
    count_0 = np.sum(targets == 0)
    count_1 = np.sum(targets == 1)
    print(f"        Count of 0: {count_0}, Count of 1: {count_1}")
      
#%%########################### EEGNet portion ##################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%######### PRETRAINING ############
num_epochs = 1
max_acc = 0
save_model_path = "EEGNet_Sup_Class/checkpoints/best_model_64.pth"
model = EEGNet(num_channels=8, num_timepoints=226, emb_dim=64)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
supcon_loss = SupConLoss(temperature=0.07).to(device)
ce_loss = nn.CrossEntropyLoss().to(device)

initial_alpha = 0.7  
final_alpha = 0.3 

for epoch in range(num_epochs):
    # alpha = decay_alpha(epoch, initial_alpha, final_alpha, num_epochs)
    alpha = 0.5

    train_loss = train_supcon_ce(model, train_loader, optimizer, supcon_loss, ce_loss, device, alpha=alpha)
    
    val_acc = evaluate(model, val_loader, device)
    print(f"\rEpoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}", end=" ", flush=True)

    if val_acc > max_acc:
        max_acc = val_acc
        torch.save(model.state_dict(), save_model_path)
        print(f"\rModel saved with accuracy {max_acc:.4f}  -  {save_model_path}", end="\n", flush=False)

#%%######### FINETUNING FOR SPECIFIC SUBJECT ############
subject_ids = [1]  
num_epochs = 1
max_acc = 0

load_model_path = "EEGNet_Sup_Class/checkpoints/best_model_64.pth"
save_model_path = "EEGNet_Sup_Class/checkpoints/best_model_64_S1_finetuned.pth"

finetune_model(subject_ids, EEGNet, load_model_path, save_model_path, device, num_epochs=num_epochs)

#%% Evaluate the model
subject_ids = [1]  # Adjust subject IDs as necessary
train_loader, val_loader = get_dataloaders(subject_ids, folder=folder, ratio_augment=ratio_augment)

# Load the trained model
model_S1 = EEGNet(num_channels=8, num_timepoints=226, emb_dim=64)
model_S1.to(device)
model_S1.load_state_dict(torch.load("EEGNet_Sup_Class/best_model_64_S1.pth"))

# Extract embeddings for training and validation sets
train_embs, train_labels = extract_embeddings(model_S1, train_loader, device)
val_embs, val_labels = extract_embeddings(model_S1, val_loader, device)

# Visualize embeddings for training and validation sets using UMAP
create_umap_plot(train_embs, train_labels, title="3D UMAP of EEGNet Embeddings (Train) - S1")
create_umap_plot(val_embs, val_labels, title="3D UMAP of EEGNet Embeddings (Validation) - S1")

# %%