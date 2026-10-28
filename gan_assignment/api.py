import logging
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock

import torch
from fastapi import APIRouter, HTTPException, Query, Response

from helper_lib.model import Generator
from helper_lib.generator import generate_samples, samples_png

router = APIRouter(prefix="/gan", tags=["Assignment 3 - MNIST GAN"])
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "gan" / "mnist_gan.pt"
_generation_lock = Lock()


def checkpoint_path():
    return Path(os.environ.get("GAN_MODEL_PATH", str(DEFAULT_CHECKPOINT))).expanduser().resolve()


@lru_cache(maxsize=1)
def load_generator(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("latent_dim") != 100 or checkpoint.get("image_shape") != [1, 28, 28]:
        raise ValueError("GAN checkpoint metadata does not match the architecture")
    generator = Generator()
    generator.load_state_dict(checkpoint["generator_state_dict"], strict=True)
    generator.eval()
    return generator


@router.get("/health")
def health():
    return {"checkpoint_exists": checkpoint_path().is_file(), "latent_dim": 100,
            "image_shape": [1, 28, 28], "conditional": False}


@router.get("/generate", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
def generate(num_samples: int = Query(16, ge=1, le=64),
             seed: int | None = Query(None, ge=0, le=2147483647)):
    """Generate a PNG grid of random handwritten digits. A seed repeats a result."""
    try:
        # Serialize mode changes to the cached model, protecting BatchNorm state.
        with _generation_lock:
            generator = load_generator(str(checkpoint_path()))
            samples = generate_samples(generator, "cpu", num_samples, seed=seed, show=False)
            content = samples_png(samples)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="GAN weights missing. Run: python -m gan_assignment.train")
    except Exception as exc:
        logging.getLogger(__name__).exception("GAN generation failed")
        raise HTTPException(status_code=503, detail="GAN could not load or generate images") from exc
    return Response(content=content, media_type="image/png",
                    headers={"Cache-Control": "no-store", "Content-Disposition": 'inline; filename="mnist_digits.png"'})
