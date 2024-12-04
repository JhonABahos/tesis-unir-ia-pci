# modules/models.py

"""
Este módulo define funciones para construir modelos de segmentación.
"""

import torch
from torch import nn
from torchvision import models


def get_deeplabv3_model(num_classes, pretrained=True):
    """
    Crea un modelo DeepLabV3 con ResNet-101 como backbone.

    :param num_classes: Número de clases para la segmentación.
    :param pretrained: Booleano que indica si usar pesos preentrenados en ImageNet.
    :return: Modelo DeepLabV3 modificado.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)

    # Reemplazar la última capa de clasificación para el número de clases deseado
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    # Verificar si aux_classifier está presente (algunos modelos no la tienen)
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    return model
