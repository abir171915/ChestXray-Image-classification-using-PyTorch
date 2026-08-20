"""Standalone inference on a single chest X-ray image.

Hardcoded to ResNet18 (frozen backbone) since that's the best-performing
model in this project (see README) — this script is for using the result,
not for choosing between architectures.

Usage:
    python -m src.predict --image path/to/xray.jpeg
    python -m src.predict --image path/to/xray.jpeg --checkpoint checkpoints/resnet18_frozen.pth
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.model import build_resnet
from src.transforms import RESNET_TRANSFORM

CLASS_NAMES = ("NORMAL", "PNEUMONIA")
DEFAULT_CHECKPOINT = Path("checkpoints/resnet18_frozen.pth")


def load_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
    # pretrained=False: no point downloading ImageNet weights just to overwrite
    # them with our own checkpoint on the next line.
    model = build_resnet("resnet18", num_classes=len(CLASS_NAMES), freeze_backbone=True, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_proba(model: torch.nn.Module, image: Image.Image, device: str) -> dict[str, float]:
    """Returns {class_name: probability} for every class, given a PIL image.

    Takes an already-opened PIL Image (rather than a path) so this also works
    for images uploaded through an API, not just files on disk.
    """
    x = RESNET_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


def predict_image(model: torch.nn.Module, image_path: Path, device: str) -> tuple[str, float]:
    """Returns (predicted_class, confidence) for a single image on disk."""
    probs = predict_proba(model, Image.open(image_path), device)
    pred_class = max(probs, key=probs.get)
    return pred_class, probs[pred_class]


def main():
    parser = argparse.ArgumentParser(description="Predict NORMAL vs PNEUMONIA on a chest X-ray.")
    parser.add_argument("--image", type=Path, required=True, help="Path to a chest X-ray image.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint} "
            "(train one with `python -m src.train --model resnet18`)"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    pred_class, confidence = predict_image(model, args.image, device)

    print(f"{pred_class} ({confidence:.2%} confidence)")


if __name__ == "__main__":
    main()
