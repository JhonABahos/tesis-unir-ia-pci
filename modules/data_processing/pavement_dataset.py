# modules.data_processing.pavement_dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset


class PavementDataset(Dataset):
    """
    Dataset para cargar imágenes y máscaras correspondientes para la segmentación de pavimentos.

    :param images_dir: Directorio que contiene las imágenes.
    :param masks_dir: Directorio que contiene las máscaras correspondientes.
    :param transform: Transformaciones a aplicar a las imágenes.
    :param target_transform: Transformaciones a aplicar a las máscaras.
    """
    def __init__(self, images_dir, masks_dir, transform=None, target_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_transform = target_transform

        # Filtrar y ordenar imágenes para asegurar la consistencia
        self.images = sorted([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])

    def __len__(self):
        """
        Devuelve la cantidad de imágenes en el dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Devuelve la imagen y la máscara correspondiente en la posición `idx`.

        :param idx: Índice del elemento a obtener.
        :return: Imagen transformada y máscara transformada.
        """
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx])

        # Asegurarse de que las rutas existen
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Archivo no encontrado: {img_path} o {mask_path}")

        # Cargar la imagen y la máscara
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Aplicar transformaciones si están definidas
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask
