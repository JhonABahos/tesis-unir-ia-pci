# modules.models.py

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


class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(HybridCNNTransformer, self).__init__()
        # Backbone CNN (ResNet-50)
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Eliminar capas FC y avgpool

        # Vision Transformer (ViT)
        self.transformer = models.vit_b_16(pretrained=pretrained)
        self.transformer.heads = nn.Identity()  # Eliminar la capa de clasificación

        # Fusionar salidas de CNN y Transformer
        self.conv_fusion = nn.Conv2d(2048, 768, kernel_size=1)  # Adaptar canales ResNet a ViT
        self.fc = nn.Conv2d(768, num_classes, kernel_size=1)  # Capa de clasificación final

    def forward(self, x):
        # Paso 1: Características locales con CNN
        cnn_features = self.backbone(x)  # Salida: [B, 2048, H/32, W/32]

        # Paso 2: Reducción de canales para encajar con ViT
        cnn_features = self.conv_fusion(cnn_features)  # Salida: [B, 768, H/32, W/32]

        # Paso 3: Aplanar para ViT
        b, c, h, w = cnn_features.shape
        transformer_input = cnn_features.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, 768]

        # Paso 4: Contexto global con Transformer
        transformer_output = self.transformer.encoder(transformer_input)  # [B, H*W, 768]

        # Paso 5: Reconstruir salida y clasificar
        transformer_output = transformer_output.permute(0, 2, 1).view(b, 768, h, w)  # [B, 768, H/32, W/32]
        output = self.fc(transformer_output)  # [B, num_classes, H/32, W/32]
        return output
