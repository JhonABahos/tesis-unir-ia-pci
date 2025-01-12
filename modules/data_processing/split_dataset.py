# modules/data_processing/split_dataset.py
import os
import random
import shutil

# Directorio base de imágenes descargadas
base_dir = r"C:\Users\nesto\OneDrive\Documents\PycharmProjects\py_tesis_ia_pci\data\all_images"
train_dir = r"C:\Users\nesto\OneDrive\Documents\PycharmProjects\py_tesis_ia_pci\data\train\images"
val_dir = r"C:\Users\nesto\OneDrive\Documents\PycharmProjects\py_tesis_ia_pci\data\val\images"


# Crear carpetas si no existen
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Obtener lista de imágenes y mezclarlas aleatoriamente
all_images = [f for f in os.listdir(base_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(all_images)

# Listar imágenes ya existentes en las carpetas de destino
existing_train_images = set(os.listdir(train_dir))
existing_val_images = set(os.listdir(val_dir))

# Verificar imágenes ya presentes en cualquier carpeta (train o val)
existing_images = existing_train_images.union(existing_val_images)

# Imprimir estadísticas iniciales
print(f"Total de imágenes en `all_images`: {len(all_images)}")
print(f"Imágenes actuales en `train/images`: {len(existing_train_images)}")
print(f"Imágenes actuales en `val/images`: {len(existing_val_images)}")
print(f"Imágenes actualmente distribuidas (train o val): {len(existing_images)}")

# Filtrar imágenes que no están ni en train ni en val
new_images = [img for img in all_images if img not in existing_images]

# Calcular la cantidad de imágenes para entrenamiento (80%) y validación (20%)
train_count = round(len(new_images) * 0.8)  # 80% para entrenamiento
val_count = len(new_images) - train_count  # 20% para validación

# Dividir en conjuntos de entrenamiento y validación
train_images = new_images[:train_count]
val_images = new_images[train_count:]

print(f"Total de imágenes nuevas a distribuir: {len(new_images)}")
print(f"Nuevas imágenes asignadas a `train/images`: {len(train_images)}")
print(f"Nuevas imágenes asignadas a `val/images`: {len(val_images)}")

# Copiar imágenes a las carpetas correspondientes
for img in train_images:
    shutil.copy(os.path.join(base_dir, img), os.path.join(train_dir, img))

for img in val_images:
    shutil.copy(os.path.join(base_dir, img), os.path.join(val_dir, img))

# Imprimir distribución final
final_train_images = set(os.listdir(train_dir))
final_val_images = set(os.listdir(val_dir))

print(f"\nDistribución completa finalizada:")
print(f"Total de imágenes en `train/images`: {len(final_train_images)}")
print(f"Total de imágenes en `val/images`: {len(final_val_images)}")
