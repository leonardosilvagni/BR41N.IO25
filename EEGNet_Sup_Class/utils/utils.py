import umap.umap_ as umap
import plotly.graph_objects as go

import os

import torch   
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Sampler
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import WeightedRandomSampler
import random

import numpy as np

from preprocessing import *
from data_augmentation import *
from ..model.EEGNet_Sup_class import *

class ShuffledWeightedSampler(Sampler):
    def __init__(self, weights, num_samples, replacement=True):
        """
        weights: a 1-D torch tensor of weights for each sample.
        num_samples: number of samples to draw in each epoch.
        replacement: whether sampling is done with replacement.
        """
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        # Draw samples according to the weights.
        indices = torch.multinomial(self.weights, self.num_samples, self.replacement)
        # Shuffle the sampled indices
        shuffled_indices = indices[torch.randperm(len(indices))]
        return iter(shuffled_indices.tolist())

    def __len__(self):
        return self.num_samples
    
# ensure reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_subject(X):
    # Z-score across trials, separately for each channel
    trials, channels, samples = X.shape
    X_reshaped = X.transpose(1, 0, 2).reshape(channels, -1)
    X_reshaped = (X_reshaped - X_reshaped.mean(axis=1, keepdims=True)) / (X_reshaped.std(axis=1, keepdims=True) + 1e-6)
    return X_reshaped.reshape(channels, trials, samples).transpose(1, 0, 2)

def get_sampler(y):
    unique_classes = torch.unique(y)
    class_to_index = {cls.item(): idx for idx, cls in enumerate(unique_classes)}
    
    class_sample_count = torch.tensor([(y == cls).sum() for cls in unique_classes], dtype=torch.float)
    weights = 1.0 / class_sample_count

    sample_weights = torch.tensor([weights[class_to_index[int(label.item())]] for label in y])

    # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    sampler = ShuffledWeightedSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def get_dataloaders(subject_ids, folder='p300-speller', batch_size=128, ratio_augment=0.5):
    X_train_all, X_val_all = [], []
    y_train_all, y_val_all = [], []

    for sid in subject_ids:
        filepath = os.path.join(folder, f"S{sid}.mat")
        print(f"Loading subject {sid} from {filepath}")
        
        epochs = get_epochs_from_file(filepath)
        epochs.pick_types(eeg=True)

        X = epochs.get_data() *1000
        y = epochs.events[:, -1] 

        X = normalize_subject(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        X_train, y_train = balance_data(X_train, y_train, ratio=ratio_augment)

        print(f"    Subject {sid} - Train shape: {X_train.shape}, Val shape: {X_val.shape}")
        X_train_all.append(X_train)
        X_val_all.append(X_val)
        y_train_all.append(y_train)
        y_val_all.append(y_val)

    # Stack all subjects' data
    X_train = torch.tensor(np.vstack(X_train_all), dtype=torch.float32)
    y_train = torch.tensor((np.hstack(y_train_all) == 1).astype(np.longlong))  # Binary classification

    X_val = torch.tensor(np.vstack(X_val_all), dtype=torch.float32)
    y_val = torch.tensor((np.hstack(y_val_all) == 1).astype(np.longlong))

    # Samplers
    train_sampler = get_sampler(y_train)
    val_sampler = get_sampler(y_val)

    # Datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader


def finetune_model(subject_ids, model_class, load_model_path, save_model_path, device, num_epochs=50, batch_size=128, initial_alpha=0.7, final_alpha=0.3, ratio_augment=0.5, folder='p300-speller'):
    print('Info finetuning:')
    print('     Subject IDs:', subject_ids)
    print('     Load model from:', load_model_path)
    print('     Save model to:', save_model_path)
    train_loader, val_loader = get_dataloaders(subject_ids, folder=folder, ratio_augment=ratio_augment)

    model = model_class(num_channels=8, num_timepoints=226, emb_dim=64)  # Adjust based on your model class
    model.to(device)

    model.load_state_dict(torch.load(load_model_path))
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)  # Lower learning rate for fine-tuning
    supcon_loss = SupConLoss(temperature=0.07).to(device)
    ce_loss = nn.CrossEntropyLoss().to(device)
    
    max_acc = 0

    print('Finetuning...')
    for epoch in range(num_epochs):
        alpha = 0.5 
        # alpha = decay_alpha(epoch, initial_alpha, final_alpha, num_epochs)
        
        train_loss = train_supcon_ce(model, train_loader, optimizer, supcon_loss, ce_loss, device, alpha=alpha)
        
        val_acc = evaluate(model, val_loader, device)
        print(f"\r      Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}", end="", flush=True)

        if val_acc > max_acc:
            max_acc = val_acc
            torch.save(model.state_dict(), save_model_path)
            print(f"\r      Model saved with accuracy {max_acc:.4f} - {save_model_path}", flush=False)


def extract_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            emb, _ = model(X)  # only embeddings
            all_embeddings.append(emb.cpu())
            all_labels.append(y)

    all_embeddings = torch.cat(all_embeddings).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return all_embeddings, all_labels

def create_umap_plot(embeddings, labels, title="UMAP Plot", unique_labels=None):
    # Apply UMAP to reduce dimensionality to 3D
    umap_model = umap.UMAP(n_components=3, random_state=42)
    emb_3d = umap_model.fit_transform(embeddings)

    # Prepare color mapping for unique labels
    if unique_labels is None:
        unique_labels = np.unique(labels)
    colors = [plt.cm.viridis(i / len(unique_labels)) for i in range(len(unique_labels))]

    # Create Plotly 3D scatter plot
    traces = []
    for i, label in enumerate(unique_labels):
        indices = labels == label
        traces.append(go.Scatter3d(
            x=emb_3d[indices, 0],
            y=emb_3d[indices, 1],
            z=emb_3d[indices, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=f'rgb({int(colors[i][0]*255)}, {int(colors[i][1]*255)}, {int(colors[i][2]*255)})',
                opacity=0.7
            ),
            name=f'Class {label}'
        ))

    # Create and display the figure
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        legend_title="Classes",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

def decay_alpha(epoch, initial_alpha=0.7, final_alpha=0.3, num_epochs=20):
    # Linear decay of alpha
    return initial_alpha - (initial_alpha - final_alpha) * (epoch / num_epochs)
