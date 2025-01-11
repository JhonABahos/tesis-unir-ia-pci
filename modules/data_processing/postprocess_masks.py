# module.postprocess_masks.py

import os
import cv2
import numpy as np
from modules.config import DATA_DIR

def remove_small_objects(mask, min_size=500):
    """
    Elimina objetos pequeños de una máscara binaria.
    :param mask: Máscara binaria (numpy array).
    :param min_size: Tamaño mínimo en píxeles para conservar un objeto.
    :return: Máscara binaria con objetos pequeños eliminados.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Crear una nueva máscara que conserve solo los objetos grandes
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num_labels):  # Ignorar el fondo (etiqueta 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            new_mask[labels == i] = 255

    return new_mask


def remove_thin_objects(mask, min_width=10):
    """
    Elimina objetos delgados de una máscara binaria.
    :param mask: Máscara binaria (numpy array).
    :param min_width: Ancho mínimo en píxeles para conservar un objeto.
    :return: Máscara binaria con objetos delgados eliminados.
    """
    # Usar morfología para eliminar objetos delgados
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_width, 1))
    thickened_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return thickened_mask


def filter_borders_and_non_road(mask, min_area=1000):
    """
    Filtra áreas no relacionadas con la carretera, como bordillos y bordes.
    :param mask: Máscara binaria (numpy array).
    :param min_area: Área mínima en píxeles para conservar un objeto.
    :return: Máscara binaria con objetos irrelevantes eliminados.
    """
    # Detectar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)

    for contour in contours:
        # Filtrar por área
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return filtered_mask


def process_existing_masks(masks_dir, output_dir, min_size=500, min_width=10, min_area=1000):
    """
    Postprocesa las máscaras existentes para eliminar objetos pequeños, delgados y bordillos.
    :param masks_dir: Directorio donde están las máscaras actuales.
    :param output_dir: Directorio donde se guardarán las máscaras postprocesadas.
    :param min_size: Tamaño mínimo en píxeles para conservar un objeto.
    :param min_width: Ancho mínimo en píxeles para conservar un objeto.
    :param min_area: Área mínima en píxeles para filtrar bordillos y no-carretera.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(masks_dir):
        if filename.endswith(('.jpg', '.png')):
            mask_path = os.path.join(masks_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Cargar la máscara como imagen en escala de grises
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Aplicar los filtros en secuencia
            mask = remove_small_objects(mask, min_size)
            mask = remove_thin_objects(mask, min_width)
            mask = filter_borders_and_non_road(mask, min_area)

            # Guardar la nueva máscara
            cv2.imwrite(output_path, mask)
            print(f"Máscara postprocesada guardada en: {output_path}")


if __name__ == "__main__":
    # Configura los directorios
    train_masks_dir = os.path.join(DATA_DIR, 'train/masks')
    postprocessed_masks_dir = os.path.join(DATA_DIR, 'train/postprocessed_masks')

    # Procesar máscaras
    print("Postprocesando máscaras existentes...")
    process_existing_masks(train_masks_dir, postprocessed_masks_dir, min_size=500, min_width=10, min_area=1000)
    print("Postprocesamiento completado.")
