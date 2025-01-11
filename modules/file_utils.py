#module.file_utils.py

import os
import shutil

def move_images_to_root_folder(root_dir, image_extensions=None):
    """
    Mueve todas las imágenes de las subcarpetas dentro de root_dir a root_dir directamente.

    :param root_dir: Directorio raíz que contiene las subcarpetas con imágenes.
    :param image_extensions: Lista de extensiones de archivo a considerar como imágenes.
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    for root, dirs, files in os.walk(root_dir):
        # Evitar mover archivos desde la carpeta raíz
        if root == root_dir:
            continue

        for file in files:
            if file.lower().endswith(tuple(image_extensions)):
                source_path = os.path.join(root, file)
                target_path = os.path.join(root_dir, file)

                # Renombrar si ya existe un archivo con el mismo nombre
                if os.path.exists(target_path):
                    base, ext = os.path.splitext(file)
                    i = 1
                    while os.path.exists(target_path):
                        target_path = os.path.join(root_dir, f"{base}_{i}{ext}")
                        i += 1

                shutil.move(source_path, target_path)
                print(f"Movido: {source_path} -> {target_path}")

        # Intentar eliminar la carpeta si está vacía
        try:
            if not os.listdir(root):  # Verifica que la carpeta esté vacía
                os.rmdir(root)
                print(f"Carpeta eliminada: {root}")
        except PermissionError:
            print(f"Permiso denegado para eliminar la carpeta: {root}")
        except Exception as e:
            print(f"No se pudo eliminar la carpeta {root}: {e}")

# Ruta al directorio raíz
root_directory = r'C:\Users\nesto\OneDrive\Documents\PycharmProjects\py_tesis_ia_pci\data\train\images'

# Ejecutar la función
move_images_to_root_folder(root_directory)
