"""Shared image transforms.

Split out from train.py so that inference-only code (predict.py, the FastAPI
app) doesn't have to import the training loop — and its training-only
dependencies like tqdm — just to get a transform.
"""

from torchvision import transforms

TINYVGG_TRANSFORM = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor(),
])

RESNET_TRANSFORM = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
