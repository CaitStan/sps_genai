"""Run: uv run python -m gan_assignment.train --epochs 20"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib import get_model, train_gan, generate_samples
from helper_lib.generator import samples_png

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "gan")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, threads and lr must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(args.device)
    print(f"Training MNIST GAN on {device}", flush=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    model = get_model("GAN").to(device)
    optimizers = {"generator": torch.optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.5, 0.999)),
                  "discriminator": torch.optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    def save_epoch(gan, row):
        sample_path = args.output_dir / f"samples_epoch_{row['epoch']:03d}.png"
        samples = generate_samples(gan, device, 64, seed=123, show=False)
        png = samples_png(samples)
        sample_path.write_bytes(png)
        (args.output_dir / "samples.png").write_bytes(png)
        checkpoint = {"generator_state_dict": {k: v.detach().cpu().clone() for k, v in gan.generator.state_dict().items()},
                      "discriminator_state_dict": {k: v.detach().cpu().clone() for k, v in gan.discriminator.state_dict().items()},
                      "latent_dim": 100, "image_shape": [1, 28, 28], "epoch": row["epoch"],
                      "dataset": "MNIST", "seed": args.seed}
        temporary = args.output_dir / "mnist_gan.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(args.output_dir / "mnist_gan.pt")
        report = {"dataset": "MNIST training split", "training_examples": len(dataset),
                  "epochs_completed": row["epoch"], "seed": args.seed, "sample_seed": 123,
                  "device": str(device), "batch_size": args.batch_size, "learning_rate": args.lr,
                  "adam_betas": [0.5, 0.999], "elapsed_seconds": time.perf_counter() - start,
                  "history": gan.history,
                  "note": "GAN losses are not classification accuracy. Final epoch checkpoint; no cherry-picked images."}
        (args.output_dir / "training_metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    train_gan(model, loader, nn.BCELoss(), optimizers, device, args.epochs, on_epoch_end=save_epoch)
    print(f"Saved trained GAN, metrics and sample grids to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
