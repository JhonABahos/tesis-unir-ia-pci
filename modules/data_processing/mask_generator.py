#modules.data_processing.mask_generator.py

import os
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from PIL import Image
import numpy as np
from skimage.morphology import remove_small_objects
import cv2

# Cargar modelo DeepLabV3 preentrenado para segmentación
weights = DeepLabV3_ResNet101_Weights.DEFAULT
road_model = deeplabv3_resnet101(weights=weights)
road_model.eval()

# Transformaciones necesarias para la entrada al modelo
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def segment_road(image_path, output_path, min_object_size=5000):
    """
    Segmenta el área de la carretera en una imagen, eliminando elementos irrelevantes.
    :param image_path: Ruta de la imagen original.
    :param output_path: Ruta para guardar la imagen segmentada.
    :param min_object_size: Tamaño mínimo en píxeles para conservar un objeto.
    """
    # Cargar y preprocesar la imagen
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # Guardar tamaño original (ancho, alto)
    input_tensor = preprocess(image).unsqueeze(0)

    # Inferencia del modelo
    with torch.no_grad():
        output = road_model(input_tensor)['out'][0]
    output_predictions = torch.argmax(output, dim=0).cpu().numpy()

    # Crear máscara binaria para la clase "road" (carretera)
    road_mask = (output_predictions == 0).astype(np.uint8)

    # Eliminar objetos pequeños
    road_mask = remove_small_objects(road_mask.astype(bool), min_size=min_object_size).astype(np.uint8)

    # Aplicar filtro para bordillos y objetos delgados
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(road_mask)
    for contour in contours:
        if cv2.contourArea(contour) > min_object_size:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Redimensionar la máscara al tamaño original de la imagen
    road_mask_resized = cv2.resize(filtered_mask, original_size, interpolation=cv2.INTER_NEAREST)

    # Aplicar máscara a la imagen original
    image_array = np.array(image)
    segmented_image = image_array * road_mask_resized[:, :, None]

    # Guardar la imagen segmentada
    segmented_image = Image.fromarray(segmented_image)
    segmented_image.save(output_path)
    print(f"Carretera segmentada guardada en: {output_path}")

    return output_path

def generate_damage_masks(segmented_image_path, mask_output_path):
    """
    Genera máscaras de daños en la carretera segmentada.
    :param segmented_image_path: Ruta de la imagen segmentada (solo carretera).
    :param mask_output_path: Ruta para guardar la máscara de daños.
    """
    # Cargar y preprocesar la imagen segmentada
    image = Image.open(segmented_image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0)

    # Inferencia del modelo de segmentación
    with torch.no_grad():
        output = road_model(input_tensor)['out'][0]
    damage_predictions = torch.argmax(output, dim=0).cpu().numpy()

    # Verificar si el valor máximo es mayor a 0 antes de normalizar
    if damage_predictions.max() > 0:
        damage_mask = (damage_predictions * 255 / damage_predictions.max()).astype(np.uint8)
    else:
        damage_mask = np.zeros_like(damage_predictions, dtype=np.uint8)

    # Guardar la máscara de daños
    mask_image = Image.fromarray(damage_mask)
    mask_image.save(mask_output_path)
    print(f"Máscara de daños generada en: {mask_output_path}")

def process_images(image_dir, road_output_dir, mask_output_dir):
    """
    Procesa un conjunto de imágenes, segmentando la carretera y generando máscaras de daños.
    :param image_dir: Directorio de imágenes originales.
    :param road_output_dir: Directorio donde se guardarán las imágenes segmentadas (carretera).
    :param mask_output_dir: Directorio donde se guardarán las máscaras de daños.
    """
    os.makedirs(road_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_dir, filename)
            segmented_image_path = os.path.join(road_output_dir, f"segmented_{filename}")
            segment_road(image_path, segmented_image_path)

            mask_output_path = os.path.join(mask_output_dir, f"mask_{os.path.splitext(filename)[0]}.jpg")
            generate_damage_masks(segmented_image_path, mask_output_path)

import os
from modules.data_processing.mask_generator import process_images

if __name__ == "__main__":
    # Directorios de entrada y salida (usando rutas absolutas)
    image_dir = "C:/Users/nesto/OneDrive/Documents/PycharmProjects/py_tesis_ia_pci/data/train/images"
    road_output_dir = "C:/Users/nesto/OneDrive/Documents/PycharmProjects/py_tesis_ia_pci/data/train/segmented_images"
    mask_output_dir = "C:/Users/nesto/OneDrive/Documents/PycharmProjects/py_tesis_ia_pci/data/train/masks"

    # Crear las carpetas si no existen
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(road_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    print("Iniciando la segmentación y generación de máscaras...")
    process_images(image_dir, road_output_dir, mask_output_dir)
    print("Segmentación y generación de máscaras completada.")
