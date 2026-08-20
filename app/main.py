"""FastAPI service wrapping the ResNet18 pneumonia classifier for inference.

The model is loaded once at startup, not per-request — reloading a ~45MB
checkpoint on every prediction would make each request slow for no reason.

Run locally:
    uvicorn app.main:app --reload

Then:
    curl -X POST -F "file=@xray.jpeg" http://127.0.0.1:8000/predict
"""

from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.predict import DEFAULT_CHECKPOINT, load_model, predict_proba

model_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = Path(DEFAULT_CHECKPOINT)
    if not checkpoint.exists():
        raise RuntimeError(
            f"Checkpoint not found at {checkpoint}. "
            "Train one with `python -m src.train --model resnet18`."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_state["device"] = device
    model_state["model"] = load_model(checkpoint, device)

    yield

    model_state.clear()


app = FastAPI(
    title="Chest X-Ray Pneumonia Classifier",
    description="ResNet18 (frozen backbone), trained on a patient-level, leak-free split. "
    "See the project README for full methodology and results.",
    lifespan=lifespan,
)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image = Image.open(file.file)
        image.load()  # force-read now, so a corrupt file fails here, not mid-inference
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image.")

    probabilities = predict_proba(model_state["model"], image, model_state["device"])
    predicted_class = max(probabilities, key=probabilities.get)

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=probabilities[predicted_class],
        probabilities=probabilities,
    )
