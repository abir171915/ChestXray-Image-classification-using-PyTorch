"""Gradio app for the chest X-ray pneumonia classifier.

Loads the ResNet18 (frozen backbone) checkpoint from the public HF model
repo at startup, then serves predictions through a Gradio UI (which also
auto-exposes an API endpoint for the same function).

Full training methodology, evaluation, and other experiments:
https://github.com/<your-github-username>/Chest_Xray
"""

import spaces  # noqa: F401  -- must be imported before torch/anything CUDA-related,
                # this Space runs on ZeroGPU hardware and needs to intercept CUDA init,
                # even though this app never calls @spaces.GPU (CPU inference is fast enough).

import torch
import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from model import build_resnet18

CLASS_NAMES = ("NORMAL", "PNEUMONIA")
CHECKPOINT_REPO = "abir171/pneumonia-xray-resnet18"
CHECKPOINT_FILE = "resnet18_frozen.pth"

RESNET_TRANSFORM = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loaded once at import time, on CPU — cheap enough that the GPU isn't needed
# for this model, but ZeroGPU hardware requires at least one @spaces.GPU
# function to exist, so `predict` below claims the GPU during its call and
# moves the model over for the duration of that request.
checkpoint_path = hf_hub_download(repo_id=CHECKPOINT_REPO, filename=CHECKPOINT_FILE)
model = build_resnet18(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()


@spaces.GPU
def predict(image: Image.Image) -> dict[str, float]:
    if image is None:
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    x = RESNET_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Chest X-ray"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Chest X-Ray Pneumonia Classifier",
    description=(
        "ResNet18, transfer learning with a frozen backbone, trained on a patient-level "
        "(leak-free) split with data augmentation. 91% test accuracy. "
        "Upload a chest X-ray to see the model's prediction."
    ),
)

if __name__ == "__main__":
    demo.launch()
