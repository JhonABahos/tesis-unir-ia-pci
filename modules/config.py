#modules.config.py

import os
import torch
import torchvision.transforms as T

# === Configuración del proyecto ===

# Rutas base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DEDUCT_VALUES_FILE = os.path.join(DATA_DIR, 'deduct_values.json')

# Validación de rutas
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"La carpeta de datos no existe: {DATA_DIR}")
if not os.path.exists(DEDUCT_VALUES_FILE):
    raise FileNotFoundError(f"El archivo deduct_values.json no existe: {DEDUCT_VALUES_FILE}")

# Tamaño estándar de las imágenes y máscaras
IMAGE_SIZE = (512, 512)

# Transformaciones para imágenes de entrada
IMAGE_TRANSFORM = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformaciones para máscaras (categóricas)
MASK_TRANSFORM = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.Lambda(lambda img: torch.tensor(img, dtype=torch.long))  # Máscara categórica
])

# === Parámetros de entrenamiento ===

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_CLASSES = 20  # Según el número de clases de daños
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4

# Ruta al modelo GPT4All
BASE_DIR = "D:/Proyectos_BC/ia_tesis_pci"
#MODEL_PATH = os.path.join(BASE_DIR, "models", "gpt4all-falcon-newbpe-q4_0.gguf")
MODEL_PATH = os.path.join(BASE_DIR, "models", "Meta-Llama-3-8B-Instruct.Q4_0.gguf")

# Parámetros del modelo
MAX_TOKENS = 500  # Número máximo de tokens por respuesta
TEMPERATURE = 0.2  # Nivel de aleatoriedad de las respuestas

# === Documentación adicional ===
# - BASE_DIR: Carpeta base del proyecto.
# - DATA_DIR: Carpeta donde se encuentran los datos (imágenes, máscaras, deduct_values.json).
# - IMAGE_SIZE: Dimensiones a las que se redimensionan las imágenes y máscaras.
# - DEVICE: Selecciona automáticamente GPU (cuda) si está disponible, o CPU.
# - NUM_CLASSES: Número total de clases, incluyendo el fondo (clase 0).
# - MASK_TRANSFORM: Se asegura de que las máscaras sean categóricas (valores enteros por clase).
# - IMAGE_TRANSFORM: Transformaciones para las imágenes de entrada al modelo.
# - BATCH_SIZE: Tamaño del lote para el entrenamiento.
# - NUM_EPOCHS: Número de épocas para entrenar el modelo.
