# modules/pci_calculator.py

import cv2
import numpy as np
import torch
from modules.deduct_values import load_deduct_values, interpolate_deduct_value
from modules.damages import get_damage_info


def classify_damage(mask, damages_dict):
    """
    Clasifica los daños en una máscara segmentada y calcula métricas clave.

    :param mask: Máscara segmentada (Tensor o ndarray).
    :param damages_dict: Diccionario de clases de daños.
    :return: Diccionario con información de los daños detectados.
    """
    damage_info = {}

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    mask: np.ndarray

    classes = np.unique(mask)
    classes = classes[classes != 0]  # Excluir el fondo (clase 0)

    for cls in classes:
        damage_mask = (mask == cls).astype(np.uint8)
        contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_damage_area = sum(cv2.contourArea(cnt) for cnt in contours)
        total_area = mask.shape[0] * mask.shape[1]
        density = (total_damage_area / total_area) * 100

        # Ajustar umbrales de severidad según criterios
        if density < 1:
            severity = 'B'
        elif density < 5:
            severity = 'M'
        else:
            severity = 'A'

        damage_info[cls] = {
            'nombre': get_damage_info(cls)['nombre'],
            'unidad': get_damage_info(cls)['unidad'],
            'area': total_damage_area,
            'density': density,
            'severity': severity,
            'count': len(contours)
        }

    return damage_info


def calculate_pci(damage_info, deduct_values):
    """
    Calcula el Índice de Condición del Pavimento (PCI) basado en la información de daños.

    :param damage_info: Diccionario con información de daños detectados.
    :param deduct_values: Diccionario con valores deducidos cargados desde deduct_values.json.
    :return: Índice de Condición del Pavimento (PCI).
    """
    deduct_values_list = []

    for cls, info in damage_info.items():
        severity = info['severity']
        density = info['density']

        # Obtener el valor deducido interpolado
        deduct_value = interpolate_deduct_value(deduct_values, str(cls), severity, density)
        deduct_values_list.append(deduct_value)

        # Almacenar el valor deducido en la información de daño
        info['deduct_value'] = deduct_value

    # Ordenar los valores deducidos en orden descendente
    deduct_values_list.sort(reverse=True)

    # Sumar todos los valores deducidos
    total_deduct_value = sum(deduct_values_list)

    # Número de valores deducidos mayores a 2
    c = len([dv for dv in deduct_values_list if dv > 2])

    # Calcular el valor deducido corregido
    if c <= 1:
        corrected_deduct_value = total_deduct_value
    else:
        maximum_deduct_value = max(deduct_values_list)
        total_deduct_value_minus_max = total_deduct_value - maximum_deduct_value
        corrected_deduct_value = maximum_deduct_value + (
            total_deduct_value_minus_max * (100 - maximum_deduct_value)
        ) / (100 - maximum_deduct_value)

    # Asegurar que el valor deducido corregido no exceda 100
    corrected_deduct_value = min(corrected_deduct_value, 100)

    # Calcular PCI
    pci = 100 - corrected_deduct_value
    return pci


