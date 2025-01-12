#modules.data_processing.check_and_generate_masks.py

import os
from modules.data_processing.segmentation_and_mask_generator import process_images

def process_all_images(data_dir):
    """
    Procesa todas las imágenes en los directorios de entrenamiento y validación.
    Solo genera máscaras para imágenes que no tienen máscaras existentes.
    """
    # Directorios de entrada y salida
    train_image_dir = os.path.join(data_dir, 'train/images')
    train_mask_dir = os.path.join(data_dir, 'train/masks')

    # Crear carpetas necesarias si no existen
    os.makedirs(train_mask_dir, exist_ok=True)

    # Identificar imágenes que necesitan máscaras
    images_to_process = []
    for img_file in os.listdir(train_image_dir):
        if img_file.endswith(('.jpg', '.png')):
            mask_file = f"mask_{os.path.splitext(img_file)[0]}.jpg"
            if mask_file not in os.listdir(train_mask_dir):
                images_to_process.append(img_file)

    if images_to_process:
        print(f"Generando máscaras para {len(images_to_process)} imágenes en {train_image_dir}...")
        process_images(train_image_dir, os.path.join(data_dir, 'train/segmented_images'), train_mask_dir)
    else:
        print("Todas las máscaras de entrenamiento ya están generadas.")

    # Repetir para validación
    val_image_dir = os.path.join(data_dir, 'val/images')
    val_mask_dir = os.path.join(data_dir, 'val/masks')

    os.makedirs(val_mask_dir, exist_ok=True)

    val_images_to_process = []
    for img_file in os.listdir(val_image_dir):
        if img_file.endswith(('.jpg', '.png')):
            mask_file = f"mask_{os.path.splitext(img_file)[0]}.jpg"
            if mask_file not in os.listdir(val_mask_dir):
                val_images_to_process.append(img_file)

    if val_images_to_process:
        print(f"Generando máscaras para {len(val_images_to_process)} imágenes en {val_image_dir}...")
        process_images(val_image_dir, os.path.join(data_dir, 'val/segmented_images'), val_mask_dir)
    else:
        print("Todas las máscaras de validación ya están generadas.")
