# modules/config.py

import os
import torch
import torchvision.transforms as T

# Establecer rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DEDUCT_VALUES_FILE = os.path.join(DATA_DIR, 'deduct_values.json')

# Transformaciones
IMAGE_TRANSFORM = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

MASK_TRANSFORM = T.Compose([
    T.Resize((512, 512)),
    T.PILToTensor()  # Asegúrate de convertir las máscaras correctamente
])

# Otros parámetros
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_CLASSES = 21  # Según el número de clases de daños
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
