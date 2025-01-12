# utils/data_processing/check_masks_utils.py

import os
from colorama import Fore, Style


def check_missing_and_extra_masks(images_dir, masks_dir, set_name):
    """
    Verifica imágenes sin máscaras y máscaras adicionales en un conjunto específico.
    """
    valid_extensions = ('.jpg', '.png', '.jpeg')

    print(f"\n{Fore.BLUE}--- Verificando {set_name} ---{Style.RESET_ALL}")

    # Verificar si los directorios existen
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"{Fore.RED}Los directorios {images_dir} o {masks_dir} no existen. Saltando...{Style.RESET_ALL}")
        return [], []

    # Filtrar archivos válidos
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(valid_extensions) and not f.startswith("__")])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png') and not f.startswith("mask___")])

    # Buscar imágenes sin máscaras
    missing_masks = [
        img for img in image_files if f"mask_{os.path.splitext(img)[0]}.png" not in mask_files
    ]

    if missing_masks:
        print(f"{Fore.RED}Faltan máscaras para {len(missing_masks)} imágenes en {set_name}:{Style.RESET_ALL}")
        for missing in missing_masks[:10]:  # Mostrar solo los primeros 10 archivos faltantes
            print(f"  - {missing}")
        if len(missing_masks) > 10:
            print(
                f"{Fore.YELLOW}Mostrando solo los primeros 10 de {len(missing_masks)} imágenes sin máscara...{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}Todas las máscaras de {set_name} están generadas.{Style.RESET_ALL}")

    # Buscar máscaras adicionales sin imágenes correspondientes
    extra_masks = [
        mask for mask in mask_files if f"{mask.replace('mask_', '').split('.')[0]}.jpg" not in image_files
    ]

    if extra_masks:
        print(f"{Fore.YELLOW}Hay {len(extra_masks)} máscaras en exceso en {set_name}:{Style.RESET_ALL}")
        for extra in extra_masks[:10]:  # Mostrar solo los primeros 10 archivos sobrantes
            print(f"  - {extra}")
        if len(extra_masks) > 10:
            print(
                f"{Fore.YELLOW}Mostrando solo los primeros 10 de {len(extra_masks)} máscaras en exceso...{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}No hay máscaras adicionales en {set_name}.{Style.RESET_ALL}")

    return missing_masks, extra_masks


def check_missing_masks(data_dir):
    """
    Verifica si todas las imágenes tienen máscaras correspondientes en los conjuntos de entrenamiento y validación.
    """
    total_missing = 0
    total_extra = 0

    # Verificación explícita para 'train' y 'val' únicamente
    sets_to_check = ["train", "val"]

    for set_name in sets_to_check:
        images_dir = os.path.join(data_dir, f"{set_name}/images")
        masks_dir = os.path.join(data_dir, f"{set_name}/masks")

        # Mostrar mensaje claro de revisión solo en estas carpetas
        print(f"\nRevisando solo la carpeta '{set_name}'...")

        missing, extra = check_missing_and_extra_masks(images_dir, masks_dir, set_name)
        total_missing += len(missing)
        total_extra += len(extra)

    print(f"\n{Fore.CYAN}--- Resumen ---{Style.RESET_ALL}")
    print(f"Total de imágenes sin máscara: {total_missing}")
    print(f"Total de máscaras en exceso: {total_extra}")
    if total_missing == 0 and total_extra == 0:
        print(f"{Fore.GREEN}¡Todo está en orden!{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}Hay problemas que debes revisar.{Style.RESET_ALL}")


# Ejemplo de uso
if __name__ == "__main__":
    # Establecer un directorio base
    data_dir = "/data"

    # Asegurarnos de excluir explícitamente "all_images"
    if "all_images" in os.listdir(data_dir):
        print(f"{Fore.MAGENTA}Ignorando carpeta 'all_images'...{Style.RESET_ALL}")

    check_missing_masks(data_dir)
