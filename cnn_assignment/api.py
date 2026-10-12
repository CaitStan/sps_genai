"""Add router to your EXISTING FastAPI app; do not replace that app."""
import os
import warnings
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from .common import CLASSES, DEFAULT_CHECKPOINT, image_transform
from .model import AssignmentCNN

router = APIRouter(prefix="/cnn", tags=["Assignment 2 - CIFAR-10"])
MAX_BYTES = 10 * 1024 * 1024


def checkpoint_path():
    return Path(os.environ.get("CNN_MODEL_PATH", str(DEFAULT_CHECKPOINT))).expanduser()


@lru_cache(maxsize=1)
def load_model(path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("classes") != list(CLASSES) or checkpoint.get("image_size") != 64:
        raise ValueError("Checkpoint classes or image size do not match this model")
    model = AssignmentCNN()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


@router.get("/health")
def health():
    return {"checkpoint_exists": checkpoint_path().is_file(), "classes": list(CLASSES),
            "input_size": [64, 64, 3]}


@router.post("/predict")
def predict(file: UploadFile = File(...)):
    # FastAPI runs this synchronous endpoint in its thread pool.
    content = file.file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image must be at most 10 MB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                tensor = image_transform()(image).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning):
        raise HTTPException(status_code=400, detail="Upload a valid image, such as PNG or JPEG")
    try:
        model = load_model(str(checkpoint_path().resolve()))
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model missing. Run: python -m cnn_assignment.train")
    except Exception as exc:
        # Model loading errors must not prevent the old app's routes from starting.
        import logging
        logging.getLogger(__name__).exception("Cannot load CNN checkpoint")
        raise HTTPException(status_code=503, detail="Model checkpoint could not be loaded") from exc
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(probabilities.argmax().item())
    return {"class_id": index, "class_name": CLASSES[index],
            "confidence": float(probabilities[index].item()),
            "probabilities": {name: float(probabilities[i].item()) for i, name in enumerate(CLASSES)}}
