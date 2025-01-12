#modules/inference.py
import os

import torch
from modules.config import DEVICE, DEDUCT_VALUES_FILE
from modules.utils.damages import DAMAGES
from modules.utils.deduct_values import load_deduct_values
from modules.utils.pci_calculator import classify_damage, calculate_pci


def evaluate_example(model, data_dir):
    """
    Evalúa un ejemplo y calcula el PCI.
    :param model: Modelo entrenado.
    :param data_dir: Directorio base de datos.
    """
    from modules.data_processing.pavement_dataset import PavementDataset
    from modules.config import IMAGE_TRANSFORM, MASK_TRANSFORM

    val_dataset = PavementDataset(
        images_dir=os.path.join(data_dir, 'val/images'),
        masks_dir=os.path.join(data_dir, 'val/masks'),
        transform=IMAGE_TRANSFORM,
        target_transform=MASK_TRANSFORM
    )

    # Cargar valores deducidos
    deduct_values = load_deduct_values(DEDUCT_VALUES_FILE)

    # Evaluar un ejemplo
    model.eval()
    with torch.no_grad():
        image, mask = val_dataset[0]
        input_image = image.unsqueeze(0).to(DEVICE)
        output = model(input_image)['out']
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        # Clasificar daños y calcular PCI
        damage_info = classify_damage(pred_mask, DAMAGES)
        pci = calculate_pci(damage_info, deduct_values)

        # Mostrar resultados
        print("\nInformación de Daños:")
        for cls, info in damage_info.items():
            print(f"Clase {cls} - {info['nombre']}:")
            print(f"  Área total del daño: {info['area']:.2f} {info['unidad']}")
            print(f"  Densidad: {info['density']:.2f}%")
            print(f"  Severidad: {info['severity']}")
            print(f"  Valor Deducido: {info['deduct_value']}")
            print(f"  Cantidad de áreas dañadas: {info['count']}")

        print(f"\nÍndice de Condición del Pavimento (PCI): {pci:.2f}")
