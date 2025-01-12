#modules/data_processing_post/postprocessing_refine_segmented_masks.py

# Postprocesamiento de máscaras segmentadas para eliminar objetos pequeños, delgados y no relacionados con la carretera.

import os
import cv2
import numpy as np
from PIL import Image

def remove_small_objects(mask, min_size=500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            new_mask[labels == i] = 255
    print(f"[INFO] remove_small_objects: Objetos encontrados: {num_labels - 1}, Tamaño mínimo: {min_size}")
    return new_mask

def remove_thin_objects(mask, min_width=10):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_width, 1))
    thickened_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    print(f"[INFO] remove_thin_objects: Se engrosaron objetos con ancho mínimo de: {min_width}")
    return thickened_mask

def filter_borders_and_non_road(mask, min_area=1000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)
    areas_conservadas = 0
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
            areas_conservadas += 1
    print(f"[INFO] filter_borders_and_non_road: Áreas conservadas: {areas_conservadas}, Área mínima: {min_area}")
    return filtered_mask

def process_existing_masks(masks_dir, images_dir, output_mask_dir, output_segmented_dir, min_size=100, min_width=3, min_area=200):
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_segmented_dir, exist_ok=True)

    for filename in os.listdir(masks_dir):
        if filename.endswith(('.jpg', '.png')):
            mask_path = os.path.join(masks_dir, filename)
            image_path = os.path.join(images_dir, filename.replace("mask_", ""))
            output_mask_path = os.path.join(output_mask_dir, filename)
            output_segmented_path = os.path.join(output_segmented_dir, f"segmented_{filename.replace('mask_', '')}")

            # Cargar máscara e imagen
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            image = Image.open(image_path).convert("RGB") if os.path.exists(image_path) else None

            if image is None:
                print(f"[WARN] No se encontró la imagen original para {filename}.")
                continue

            print(f"[INFO] Procesando: {filename}, Dimensiones máscara: {mask.shape}, Dimensiones imagen: {image.size}")

            # Redimensionar la imagen para igualar la máscara
            image = image.resize((mask.shape[1], mask.shape[0]))
            image_array = np.array(image)

            # Postprocesamiento de la máscara
            print(f"[INFO] Valores de máscara antes: Min: {np.min(mask)}, Max: {np.max(mask)}")
            mask = remove_small_objects(mask, min_size)
            mask = remove_thin_objects(mask, min_width)
            mask = filter_borders_and_non_road(mask, min_area)
            print(f"[INFO] Valores de máscara después: Min: {np.min(mask)}, Max: {np.max(mask)}")

            # Guardar máscara postprocesada
            cv2.imwrite(output_mask_path, mask)
            print(f"[INFO] Máscara postprocesada guardada en: {output_mask_path}")

            # Generar imagen segmentada
            segmented_image = np.where(mask[:, :, None] == 255, image_array, 0)
            segmented_image = Image.fromarray(segmented_image)
            segmented_image.save(output_segmented_path)
            print(f"[INFO] Imagen segmentada guardada en: {output_segmented_path}")

if __name__ == "__main__":
    data_dir = "C:/Users/nesto/OneDrive/Documents/PycharmProjects/py_tesis_ia_pci/data"

    # Rutas para entrenamiento
    train_masks_dir = os.path.join(data_dir, 'train/masks')
    train_images_dir = os.path.join(data_dir, 'train/images')
    train_postprocessed_mask_dir = os.path.join(data_dir, 'train/postprocessed_masks')
    train_postprocessed_segmented_dir = os.path.join(data_dir, 'train/postprocessed_segmented_images')

    # Rutas para validación
    val_masks_dir = os.path.join(data_dir, 'val/masks')
    val_images_dir = os.path.join(data_dir, 'val/images')
    val_postprocessed_mask_dir = os.path.join(data_dir, 'val/postprocessed_masks')
    val_postprocessed_segmented_dir = os.path.join(data_dir, 'val/postprocessed_segmented_images')

    # Procesar entrenamiento
    print("Postprocesando máscaras de entrenamiento y generando imágenes segmentadas...")
    process_existing_masks(train_masks_dir, train_images_dir, train_postprocessed_mask_dir, train_postprocessed_segmented_dir)
    print("Postprocesamiento de entrenamiento finalizado.\n")

    # Procesar validación
    print("Postprocesando máscaras de validación y generando imágenes segmentadas...")
    process_existing_masks(val_masks_dir, val_images_dir, val_postprocessed_mask_dir, val_postprocessed_segmented_dir)
    print("Postprocesamiento de validación finalizado.")

