#modules.training_pipeline.py

import os
import torch
from torch.utils.data import DataLoader
from modules.config import (
    DEVICE, BATCH_SIZE, NUM_CLASSES, NUM_EPOCHS, LEARNING_RATE, IMAGE_TRANSFORM, MASK_TRANSFORM
)
from modules.data_processing.dataset import PavementDataset
from modules.models import HybridCNNTransformer
from modules.models.training import train_one_epoch, evaluate

def train_model(data_dir):
    """
    Configura, entrena y evalúa el modelo.
    :param data_dir: Directorio base de datos.
    :return: Modelo entrenado.
    """
    # Preparar datasets y dataloaders
    train_dataset = PavementDataset(
        images_dir=os.path.join(data_dir, 'train/images'),
        masks_dir=os.path.join(data_dir, 'train/masks'),
        transform=IMAGE_TRANSFORM,
        target_transform=MASK_TRANSFORM
    )

    val_dataset = PavementDataset(
        images_dir=os.path.join(data_dir, 'val/images'),
        masks_dir=os.path.join(data_dir, 'val/masks'),
        transform=IMAGE_TRANSFORM,
        target_transform=MASK_TRANSFORM
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Configurar modelo, pérdida y optimizador
    model = HybridCNNTransformer(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Entrenamiento
    for epoch in range(NUM_EPOCHS):
        epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")
        precision, recall, f1 = evaluate(model, val_loader, DEVICE)
        print(f"Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    return model
