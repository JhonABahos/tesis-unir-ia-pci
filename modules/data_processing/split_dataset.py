# modules/split_dataset.py
import os
import random
import shutil

# Directorio base de imágenes descargadas
base_dir = r"/data/all_images"
train_dir = r"/data/train/images"
val_dir = r"/data/val/images"

# Crear carpetas si no existen
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Obtener lista de imágenes y mezclarlas aleatoriamente
all_images = [f for f in os.listdir(base_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(all_images)

# Calcular la cantidad de imágenes para entrenamiento (80%) y validación (20%)
train_count = round(len(all_images) * 0.8) # 80% para entrenamiento
val_count = len(all_images) - train_count # 20% para validación

# Dividir en conjuntos de entrenamiento y validación
train_images = all_images[:train_count]
val_images = all_images[train_count:]

print(f"Total de imágenes: {len(all_images)}")
print(f"Imágenes de entrenamiento: {len(train_images)}")
print(f"Imágenes de validación: {len(val_images)}")

# Copiar imágenes a las carpetas correspondientes
for img in train_images:
    shutil.copy(os.path.join(base_dir, img), os.path.join(train_dir, img))

for img in val_images:
    shutil.copy(os.path.join(base_dir, img), os.path.join(val_dir, img))

print(f"Distribución completa: {len(train_images)} en entrenamiento, {len(val_images)} en validación.")
