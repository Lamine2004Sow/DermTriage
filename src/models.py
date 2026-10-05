import torch
from torch import nn
from torchvision import models

from .classes import NUM_CLASSES

TRAINABLE = ("head", "layer4", "all")


def build_model(name="resnet18", pretrained=True, trainable="head"):
    """
    Construit un ResNet-18 à NUM_CLASSES sorties.
    pretrained : poids ImageNet (IMAGENET1K_V1).
    trainable : "head" (seule fc apprend), "layer4" (layer4 + fc) ou "all".
    """
    if name != "resnet18":
        raise ValueError(f"Architecture inconnue : {name}")
    if trainable not in TRAINABLE:
        raise ValueError(f"trainable doit être dans {TRAINABLE}, reçu : {trainable}")

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    for param in model.parameters():
        param.requires_grad = trainable == "all"
    if trainable == "layer4":
        for param in model.layer4.parameters():
            param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
