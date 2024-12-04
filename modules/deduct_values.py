# modules/deduct_values.py

import json
import os


def load_deduct_values(file_path=None):
    """
    Carga los valores deducidos desde un archivo JSON.

    :param file_path: Ruta del archivo JSON. Si no se especifica, se busca en una ruta por defecto.
    :return: Diccionario con los valores deducidos.
    """
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), '../data/deduct_values.json')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

    with open(file_path, 'r') as f:
        deduct_values = json.load(f)
    return deduct_values


def interpolate_deduct_value(deduct_values, class_id, severity, density):
    """
    Interpola el valor deducido para una densidad que no está directamente en el archivo JSON.

    :param deduct_values: Diccionario cargado desde deduct_values.json.
    :param class_id: ID de la clase de daño (como '1', '6', etc.).
    :param severity: Severidad del daño ('B', 'M', 'A').
    :param density: Densidad para la cual calcular el valor deducido.
    :return: Valor deducido interpolado.
    """
    keys = sorted(map(float, deduct_values[class_id][severity].keys()))
    for i in range(len(keys) - 1):
        if keys[i] <= density <= keys[i + 1]:
            x0, x1 = keys[i], keys[i + 1]
            y0 = deduct_values[class_id][severity][str(x0)]
            y1 = deduct_values[class_id][severity][str(x1)]
            return y0 + (density - x0) * (y1 - y0) / (x1 - x0)
    # Si está fuera del rango, devuelve el valor más cercano.
    return deduct_values[class_id][severity][str(keys[0])] if density < keys[0] else deduct_values[class_id][severity][
        str(keys[-1])]
