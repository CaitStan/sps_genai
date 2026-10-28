import math
from io import BytesIO

import torch
from PIL import Image
from torchvision.utils import make_grid


def generate_samples(model, device="cpu", num_samples=10, *, seed=None, show=True):
    """Generate [0,1] CPU tensors and optionally plot a grid, as in Module 6.

    Accept a GAN wrapper or its Generator. Labels are not inputs: this is an
    unconditional GAN. Supplying a seed reproduces noise without resetting
    the global random-number generator.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    generator = model.generator if hasattr(model, "generator") else model
    requested = torch.device(device)
    actual = next(generator.parameters()).device
    if requested.type != actual.type or (requested.index is not None and requested.index != actual.index):
        raise ValueError("Move the model to the requested device before generating")
    rng = torch.Generator(device="cpu")
    rng.seed() if seed is None else rng.manual_seed(seed)
    noise = torch.randn(num_samples, generator.latent_dim, generator=rng).to(device)
    was_training = generator.training
    generator.eval()  # Use learned BatchNorm running statistics for inference.
    try:
        with torch.inference_mode():
            samples = ((generator(noise).cpu() + 1) / 2).clamp(0, 1)
    finally:
        generator.train(was_training)
    if show:
        import matplotlib.pyplot as plt
        grid = make_grid(samples, nrow=min(8, num_samples), padding=2)
        plt.figure(figsize=(8, max(2, math.ceil(num_samples / 8))))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    return samples


def samples_png(samples, *, columns=8, scale=4):
    """Encode a grid of native 28x28 outputs as a PNG; scale only for display."""
    grid = make_grid(samples, nrow=min(columns, len(samples)), padding=2)
    pixels = (grid[0].clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    image = Image.fromarray(pixels)
    image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
