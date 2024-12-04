# main.py

import os
import torch
from torch.utils.data import DataLoader

from modules.config import (
    DATA_DIR, DEDUCT_VALUES_FILE, IMAGE_TRANSFORM, MASK_TRANSFORM, DEVICE, BATCH_SIZE, NUM_CLASSES, NUM_EPOCHS, LEARNING_RATE
)
from modules.dataset import PavementDataset
from modules.models import get_deeplabv3_model
from modules.utils import classify_damage, calculate_pci
from modules.deduct_values import load_deduct_values
from modules.damages import DAMAGES
from modules.training import train_one_epoch, evaluate

def main():
    # Cargar valores deducidos
    deduct_values = load_deduct_values(DEDUCT_VALUES_FILE)

    # Validar estructura de datos
    required_dirs = ['train/images', 'train/masks', 'val/images', 'val/masks']
    for subdir in required_dirs:
        path = os.path.join(DATA_DIR, subdir)
        if not os.path.exists(path) or not os.listdir(path):
            raise FileNotFoundError(f"No se encontraron archivos en la carpeta: {path}")

    # Preparar datasets y dataloaders
    train_dataset = PavementDataset(
        images_dir=os.path.join(DATA_DIR, 'train/images'),
        masks_dir=os.path.join(DATA_DIR, 'train/masks'),
        transform=IMAGE_TRANSFORM,
        target_transform=MASK_TRANSFORM
    )

    val_dataset = PavementDataset(
        images_dir=os.path.join(DATA_DIR, 'val/images'),
        masks_dir=os.path.join(DATA_DIR, 'val/masks'),
        transform=IMAGE_TRANSFORM,
        target_transform=MASK_TRANSFORM
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Configurar modelo, pérdida y optimizador
    model = get_deeplabv3_model(NUM_CLASSES).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Entrenamiento
    for epoch in range(NUM_EPOCHS):
        epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")

        precision, recall, f1 = evaluate(model, val_loader, DEVICE)
        print(f"Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    # Validación en un ejemplo
    model.eval()
    with torch.no_grad():
        image, mask = val_dataset[0]
        input_image = image.unsqueeze(0).to(DEVICE)
        output = model(input_image)['out']
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        damage_info = classify_damage(pred_mask, DAMAGES)
        pci = calculate_pci(damage_info, deduct_values)

        print("\nInformación de Daños:")
        for cls, info in damage_info.items():
            print(f"Clase {cls} - {info['nombre']}:")
            print(f"  Área total del daño: {info['area']:.2f} {info['unidad']}")
            print(f"  Densidad: {info['density']:.2f}%")
            print(f"  Severidad: {info['severity']}")
            print(f"  Valor Deducido: {info['deduct_value']}")
            print(f"  Cantidad de áreas dañadas: {info['count']}")

        print(f"\nÍndice de Condición del Pavimento (PCI): {pci:.2f}")


if __name__ == "__main__":
    main()
