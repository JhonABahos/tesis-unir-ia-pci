# modules.damages.py

"""
Este módulo define las clases de daños en pavimentos y sus características asociadas (unidad de medida).
"""

DAMAGES = {
    1: {'nombre': 'Piel de cocodrilo', 'unidad': 'm²'},
    2: {'nombre': 'Exudación', 'unidad': 'm²'},
    3: {'nombre': 'Agrietamiento en bloque', 'unidad': 'm²'},
    4: {'nombre': 'Abultamientos y hundimientos', 'unidad': 'ml'},
    5: {'nombre': 'Corrugación', 'unidad': 'm²'},
    6: {'nombre': 'Depresión', 'unidad': 'm²'},
    7: {'nombre': 'Grieta de borde', 'unidad': 'ml'},
    8: {'nombre': 'Grieta de reflexión de junta', 'unidad': 'ml'},
    9: {'nombre': 'Desnivel carril / berma', 'unidad': 'ml'},
    10: {'nombre': 'Grietas longitudinales y transversales', 'unidad': 'ml'},
    11: {'nombre': 'Parcheo', 'unidad': 'm²'},
    12: {'nombre': 'Pulimiento de agregados', 'unidad': 'm²'},
    13: {'nombre': 'Huecos', 'unidad': 'Cantidad'},
    14: {'nombre': 'Cruce de vía férrea', 'unidad': 'm²'},
    15: {'nombre': 'Ahuellamiento', 'unidad': 'm²'},
    16: {'nombre': 'Desplazamiento', 'unidad': 'm²'},
    17: {'nombre': 'Grieta parabólica (slippage)', 'unidad': 'm²'},
    18: {'nombre': 'Hinchamiento', 'unidad': 'm²'},
    19: {'nombre': 'Raveling (Pérdida de partículas gruesas)', 'unidad': 'm²'},
    20: {'nombre': 'Weathering (Meteorización)', 'unidad': 'm²'}
}

def get_damage_info(cls):
    """
    Obtiene información de una clase de daño específica.

    :param cls: Clase del daño (entero).
    :return: Diccionario con información del daño (nombre y unidad). Si no existe, retorna 'Desconocido'.
    """
    return DAMAGES.get(cls, {'nombre': 'Desconocido', 'unidad': 'Desconocido'})
