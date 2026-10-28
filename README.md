# sps_genai: word embeddings and CIFAR-10 classification

The CNN uses two convolutional layers, ReLU activations, max-pooling layers, and fully connected layers.
The classifier was trained on CIFAR-10 and achieved 64.47% test accuracy.
The trained model is integrated into the existing FastAPI application through a prediction endpoint.

## Assignment 3: MNIST GAN

The GAN follows the assigned 100-dimensional noise -> 7x7x128 -> 14x14x64 ->
28x28x1 generator architecture. `helper_lib/model.py`, `trainer.py`, and
`generator.py` implement the Module 6 reusable helpers. The existing CNN and
word-embedding endpoints are preserved.

```sh
uv sync
uv run python -m spacy download en_core_web_lg
uv run uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/gan/generate?num_samples=16&seed=42` for a PNG grid,
or use `/docs`. The original embedding route requires the spaCy model; install
it after `uv sync`, which may remove unlisted packages. Trained GAN weights and
sample grids are in `models/gan/`, so retraining is optional:

```sh
uv run python -m gan_assignment.train --epochs 20
uv run --extra test python verify_assignment3.py
```

The GAN is unconditional: it generates random digits rather than accepting a
chosen digit label. A fixed seed repeats the noise; it does not select a digit.
Included weights were trained for 20 epochs on all 60,000 MNIST training images.
Training losses and settings are recorded in `models/gan/training_metrics.json`.
GAN losses are not classification accuracy. Restart the API after retraining;
set `GAN_MODEL_PATH` to use a checkpoint in another location.

The repository did not contain the earlier Module 4 helper library. The added
`get_model` supports the new `GAN` and the existing `CNN`; unsupported model
names raise an error rather than returning an unrelated architecture.
