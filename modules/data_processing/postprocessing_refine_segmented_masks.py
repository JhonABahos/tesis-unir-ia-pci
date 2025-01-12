#module.data_processing.postprocessing_refine_segmented_masks.py

import os
import cv2
import numpy as np

def remove_small_objects(mask, min_size=500):
    """
    Elimina objetos pequeños de una máscara binaria.

    :param mask: Máscara binaria en formato numpy array (escala de grises).
    :param min_size: Tamaño mínimo en píxeles para conservar un objeto.
    :return: Máscara binaria con los objetos pequeños eliminados.
    """
    # Identifica los objetos conectados en la máscara
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Crea una nueva máscara que solo conserva los objetos grandes
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num_labels):  # Ignora el fondo (etiqueta 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:  # Conserva solo objetos grandes
            new_mask[labels == i] = 255  # Marca el objeto en blanco
    return new_mask

def remove_thin_objects(mask, min_width=10):
    """
    Elimina objetos delgados de la máscara usando operaciones morfológicas.

    :param mask: Máscara binaria en formato numpy array (escala de grises).
    :param min_width: Ancho mínimo en píxeles para conservar un objeto.
    :return: Máscara binaria con los objetos delgados eliminados.
    """
    # Crea un elemento estructurante de forma rectangular con el ancho mínimo especificado
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_width, 1))

    # Aplica una operación de cierre morfológico para engrosar objetos delgados
    thickened_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return thickened_mask

def filter_borders_and_non_road(mask, min_area=1000):
    """
    Filtra áreas no relacionadas con la carretera, como bordillos y bordes.

    :param mask: Máscara binaria en formato numpy array (escala de grises).
    :param min_area: Área mínima en píxeles para conservar un objeto.
    :return: Máscara binaria con objetos irrelevantes eliminados.
    """
    # Encuentra los contornos de los objetos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crea una nueva máscara vacía
    filtered_mask = np.zeros_like(mask)

    for contour in contours:
        # Dibuja solo los contornos que cumplen con el área mínima
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
    return filtered_mask

def process_existing_masks(masks_dir, output_dir, min_size=200, min_width=5, min_area=500):
    #process_existing_masks(masks_dir, output_dir, min_size=500, min_width=10, min_area=1000)

    """
    Postprocesa las máscaras existentes eliminando objetos pequeños, delgados y bordillos.

    :param masks_dir: Directorio donde están las máscaras actuales.
    :param output_dir: Directorio donde se guardarán las máscaras postprocesadas.
    :param min_size: Tamaño mínimo en píxeles para conservar un objeto.
    :param min_width: Ancho mínimo en píxeles para conservar un objeto.
    :param min_area: Área mínima en píxeles para conservar objetos relevantes.
    """
    # Crear la carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(masks_dir):
        if filename.endswith(('.jpg', '.png')):
            mask_path = os.path.join(masks_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Cargar la máscara en escala de grises
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Aplicar las funciones de postprocesamiento en secuencia
            mask = remove_small_objects(mask, min_size)
            mask = remove_thin_objects(mask, min_width)
            mask = filter_borders_and_non_road(mask, min_area)

            # Guardar la nueva máscara postprocesada
            cv2.imwrite(output_path, mask)
            print(f"Máscara postprocesada guardada en: {output_path}")

if __name__ == "__main__":
    # Directorio base de datos
    data_dir = "C:/Users/nesto/OneDrive/Documents/PycharmProjects/py_tesis_ia_pci/data"

    # Postprocesar máscaras del conjunto de entrenamiento
    train_masks_dir = os.path.join(data_dir, 'train/masks')
    train_postprocessed_dir = os.path.join(data_dir, 'train/postprocessed_masks')
    print("Postprocesando máscaras de entrenamiento...")
    process_existing_masks(train_masks_dir, train_postprocessed_dir)

    # Postprocesar máscaras del conjunto de validación
    val_masks_dir = os.path.join(data_dir, 'val/masks')
    val_postprocessed_dir = os.path.join(data_dir, 'val/postprocessed_masks')
    print("Postprocesando máscaras de validación...")
    process_existing_masks(val_masks_dir, val_postprocessed_dir)

    print("Postprocesamiento completado para entrenamiento y validación.")
