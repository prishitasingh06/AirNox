"""
Modular PyTorch image classification backbones for bioaerosol microscopy.
Supports ResNet18 (default lightweight baseline) and EfficientNet-B0.
"""

import torch
import torch.nn as nn
from torchvision import models


# def get_device() -> torch.device:
#     """Selects Apple Silicon MPS GPU on Mac, CUDA if available, otherwise CPU."""
#     if torch.backends.mps.is_available():
#         return torch.device("mps")
#     if torch.cuda.is_available():
#         return torch.device("cuda")
#     return torch.device("cpu")

def get_device() -> torch.device:
    """Forcing CPU on Mac to avoid MPS linear layer driver incompatibility."""
    return torch.device("cpu")

def build_classifier(arch: str = "resnet18", num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_features, num_classes)
        )
        return model

    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unsupported architecture '{arch}'. Choose 'resnet18' or 'efficientnet_b0'.")
