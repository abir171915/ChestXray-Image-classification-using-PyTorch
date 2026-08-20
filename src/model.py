"""Model definitions: the from-scratch TinyVGG baseline and a ResNet transfer-learning builder."""

import torch
from torch import nn
from torchvision import models

RESNET_BUILDERS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
}


class TinyVGG(nn.Module):
    """TinyVGG architecture, replicating the CNN Explainer reference model.

    Expects 64x64 input images. `hidden_units` controls the width of every
    conv layer; the classifier's input size (`hidden_units * 13 * 13`) is
    derived from that input resolution and this exact architecture, so it
    will need recomputing if either changes.
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 13 * 13, output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.classifier(x)


def build_resnet(
    variant: str, num_classes: int, freeze_backbone: bool = True, pretrained: bool = True
) -> nn.Module:
    """Builds a ResNet with its final layer replaced for `num_classes`.

    `variant` must be one of RESNET_BUILDERS.keys() (currently "resnet18", "resnet34").
    When `freeze_backbone=True` (the default, and what this project found works best
    for this dataset size), only the new `fc` layer is left trainable.

    `pretrained=False` skips downloading ImageNet weights entirely — use this when
    you're about to load a fine-tuned checkpoint anyway (e.g. for inference), since
    the downloaded weights would just be immediately overwritten.
    """
    if variant not in RESNET_BUILDERS:
        raise ValueError(f"Unknown ResNet variant '{variant}'. Choose from {list(RESNET_BUILDERS)}.")

    builder, weights = RESNET_BUILDERS[variant]
    model = builder(weights=weights if pretrained else None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    return model
