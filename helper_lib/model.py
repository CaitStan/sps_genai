"""Assignment 3 architecture. Tensor shapes use NCHW ordering."""
from torch import nn


class Generator(nn.Module):
    latent_dim = 100

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 7 * 7 * 128),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.layers(noise)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),  # Convert the final linear output to a real/fake probability.
        )

    def forward(self, images):
        return self.layers(images)


def initialize_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)


class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.apply(initialize_weights)
        self.history = []

    def forward(self, noise):
        return self.generator(noise)


def get_model(model_name):
    """Return models implemented in this repository; do not silently substitute."""
    name = model_name.upper()
    if name == "GAN":
        return GAN()
    if name == "CNN":
        from cnn_assignment.model import AssignmentCNN
        return AssignmentCNN()
    raise ValueError(f"Unknown model {model_name!r}. Available models: GAN, CNN")
