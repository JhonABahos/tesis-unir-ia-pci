# modules/data_processing/convert_seg_exit_to_masks.py

import numpy as np
import cv2
import os


def parse_seg_file(seg_file_path):
    """
    Lee un archivo .seg y genera una máscara binaria correspondiente.

    :param seg_file_path: Ruta del archivo .seg a procesar.
    :return: Máscara binaria como una matriz numpy.
    """
    with open(seg_file_path, 'r') as file:
        lines = file.readlines()

    # Extraer ancho y alto
    width = int(lines[1].split()[-1])  # Línea con "width"
    height = int(lines[2].split()[-1])  # Línea con "height"

    # Crear una máscara vacía con dimensiones width x height
    mask = np.zeros((height, width), dtype=np.uint8)

    # Encontrar la sección 'data' y procesar las líneas de datos
    data_start_index = lines.index("data\n") + 1
    for line in lines[data_start_index:]:
        values = list(map(int, line.split()))
        row, col_start, col_end = values[0], values[2], values[3]
        mask[row, col_start:col_end + 1] = 255  # Marca la grieta con valor 255 (blanco)

    return mask


def generate_masks_from_seg_files(input_dir, output_dir):
    """
    Genera máscaras binarias en formato PNG a partir de archivos .seg.

    :param input_dir: Directorio con archivos .seg.
    :param output_dir: Directorio donde se guardarán las máscaras generadas.
    """
    os.makedirs(output_dir, exist_ok=True)

    for seg_file in os.listdir(input_dir):
        if seg_file.endswith(".seg"):
            seg_path = os.path.join(input_dir, seg_file)
            mask = parse_seg_file(seg_path)

            # Guardar la máscara como imagen PNG
            mask_filename = os.path.splitext(seg_file)[0] + ".png"
            mask_output_path = os.path.join(output_dir, mask_filename)
            cv2.imwrite(mask_output_path, mask)

            print(f"Máscara generada: {mask_output_path}")


if __name__ == "__main__":
    # Directorio de entrada con archivos .seg
    input_dir = r"C:\Users\nesto\OneDrive\Documents\PycharmProjects\py_tesis_ia_pci\data\all_images\seg"

    # Directorio de salida donde se guardarán las máscaras PNG
    output_dir = r"C:\Users\nesto\OneDrive\Documents\PycharmProjects\py_tesis_ia_pci\data\all_images\masks"

    print("Generando máscaras a partir de archivos .seg...")
    generate_masks_from_seg_files(input_dir, output_dir)
    print("Generación de máscaras completada.")
