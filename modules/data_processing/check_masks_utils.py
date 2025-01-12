#modules.data_processing.check_masks_utils.py

import os

def check_missing_masks(data_dir):
    """
    Verifica si todas las imágenes tienen máscaras correspondientes en los conjuntos de entrenamiento y validación.
    """
    # Directorios de entrenamiento
    train_images_dir = os.path.join(data_dir, 'train/images')
    train_masks_dir = os.path.join(data_dir, 'train/masks')

    # Verificar máscaras de entrenamiento
    print("Verificando máscaras de entrenamiento...")
    train_image_files = sorted(os.listdir(train_images_dir))
    train_mask_files = sorted(os.listdir(train_masks_dir))
    train_missing_masks = [
        img for img in train_image_files if f"mask_{os.path.splitext(img)[0]}.jpg" not in train_mask_files
    ]

    if train_missing_masks:
        print(f"Faltan máscaras para {len(train_missing_masks)} imágenes en entrenamiento:")
        print(train_missing_masks)
    else:
        print("Todas las máscaras de entrenamiento están generadas.")

    # Directorios de validación
    val_images_dir = os.path.join(data_dir, 'val/images')
    val_masks_dir = os.path.join(data_dir, 'val/masks')

    # Verificar máscaras de validación
    print("\nVerificando máscaras de validación...")
    val_image_files = sorted(os.listdir(val_images_dir))
    val_mask_files = sorted(os.listdir(val_masks_dir))
    val_missing_masks = [
        img for img in val_image_files if f"mask_{os.path.splitext(img)[0]}.jpg" not in val_mask_files
    ]

    if val_missing_masks:
        print(f"Faltan máscaras para {len(val_missing_masks)} imágenes en validación:")
        print(val_missing_masks)
    else:
        print("Todas las máscaras de validación están generadas.")

    # Verificar máscaras adicionales en entrenamiento
    print("\nVerificando máscaras adicionales en entrenamiento...")
    train_extra_masks = [
        mask for mask in train_mask_files if f"{mask.replace('mask_', '').split('.')[0]}.jpg" not in train_image_files
    ]

    if train_extra_masks:
        print(f"Hay {len(train_extra_masks)} máscaras en exceso en entrenamiento:")
        print(train_extra_masks)
    else:
        print("No hay máscaras adicionales en entrenamiento.")

    # Verificar máscaras adicionales en validación
    print("\nVerificando máscaras adicionales en validación...")
    val_extra_masks = [
        mask for mask in val_mask_files if f"{mask.replace('mask_', '').split('.')[0]}.jpg" not in val_image_files
    ]

    if val_extra_masks:
        print(f"Hay {len(val_extra_masks)} máscaras en exceso en validación:")
        print(val_extra_masks)
    else:
        print("No hay máscaras adicionales en validación.")