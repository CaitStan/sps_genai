"""Run from the project root: python -m cnn_assignment.train --epochs 10"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from .common import CLASSES, DEFAULT_CHECKPOINT, ROOT, image_transform
from .model import AssignmentCNN


def evaluate(model, loader, device):
    model.eval()
    loss_total, correct, count = 0.0, 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_total += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
    return {"loss": loss_total / count, "accuracy": correct / count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, threads, and lr must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(args.device)
    print(f"Training on {device}. Downloading CIFAR-10 if needed.", flush=True)
    dataset = CIFAR10(args.data_dir, train=True, download=True, transform=image_transform())
    train_data, val_data = random_split(
        dataset, [45000, 5000], generator=torch.Generator().manual_seed(args.seed)
    )
    test_data = CIFAR10(args.data_dir, train=False, download=True, transform=image_transform())
    # Zero workers makes this portable across Windows, macOS, and notebooks.
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=0)
    model = AssignmentCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()  # Takes logits; no softmax in model.forward().
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history, best_val, best_epoch = [], -1.0, 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()  # Restore training mode after validation each epoch.
        total_loss, correct, count = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss / count,
               "train_accuracy": correct / count, "validation_loss": val["loss"],
               "validation_accuracy": val["accuracy"]}
        history.append(row)
        print(f"Epoch {epoch}/{args.epochs}: train={correct/count:.2%}, val={val['accuracy']:.2%}", flush=True)
        if val["accuracy"] > best_val:
            best_val, best_epoch = val["accuracy"], epoch
            # Portable CPU state, without pickling a model instance.
            checkpoint = {"model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                          "classes": list(CLASSES), "image_size": 64, "epoch": epoch,
                          "validation_accuracy": best_val}
            temporary = args.output.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(args.output)
    checkpoint = torch.load(args.output, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate(model, test_loader, device)  # Test set used only after selecting best validation epoch.
    report = {"epochs": args.epochs, "best_epoch": best_epoch, "seed": args.seed,
              "device": str(device), "batch_size": args.batch_size, "learning_rate": args.lr,
              "training_examples": 45000, "validation_examples": 5000, "test_examples": 10000,
              "history": history, "test": test, "elapsed_seconds": time.perf_counter() - start}
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Write a convenient real CIFAR-10 upload example.
    from PIL import Image
    Image.fromarray(test_data.data[0]).save(args.output.parent / "example_cifar10.png")
    print(f"Best epoch: {best_epoch}; test accuracy: {test['accuracy']:.2%}", flush=True)
    print(f"Saved {args.output}\nMetrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
