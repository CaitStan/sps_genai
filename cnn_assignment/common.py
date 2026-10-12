from pathlib import Path

from torchvision import transforms

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "models" / "cifar10_cnn.pt"


def image_transform():
    # Shared by training and deployment: CIFAR-10 is originally 32 x 32.
    return transforms.Compose([
        transforms.Resize((64, 64), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
