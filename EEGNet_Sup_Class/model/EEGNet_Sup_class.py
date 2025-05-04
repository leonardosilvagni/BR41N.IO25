import torch
import torch.nn as nn
import torch.nn.functional as F
from data_augmentation import balance_data
import os
from sklearn.metrics import accuracy_score

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
    acc = accuracy_score(all_labels, all_preds)
    return acc
