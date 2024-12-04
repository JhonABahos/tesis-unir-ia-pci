# modules/training.py

import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Entrena el modelo por una época.
    """
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.squeeze(1).to(device).long()

        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def evaluate(model, dataloader, device):
    """
    Evalúa el modelo y calcula precisión, recall y F1.
    """
    model.eval()
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.squeeze(1).to(device).long()

            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_masks.append(masks.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    all_preds_flat = all_preds.flatten()
    all_masks_flat = all_masks.flatten()

    precision = precision_score(all_masks_flat, all_preds_flat, average='weighted', zero_division=0)
    recall = recall_score(all_masks_flat, all_preds_flat, average='weighted', zero_division=0)
    f1 = f1_score(all_masks_flat, all_preds_flat, average='weighted', zero_division=0)

    return precision, recall, f1
