"""Minimal ResNet builder for the Space — trimmed from src/model.py.

Only what's needed to reconstruct the architecture before loading the
trained checkpoint; TinyVGG isn't included since this Space only serves
the winning model (ResNet18, frozen backbone).
"""

from torch import nn
from torchvision import models


def build_resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)  # architecture only; real weights loaded from checkpoint
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    return model
