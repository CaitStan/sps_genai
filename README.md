# sps_genai: word embeddings and CIFAR-10 classification

The original `GET /` and `GET /embedding?word=hello` endpoints are preserved.
Assignment 2 adds `POST /cnn/predict` for image classification and
`GET /cnn/health` for checkpoint presence and class labels.

## Run the app

From this repository, using the existing Python 3.13/uv setup:

```sh
uv sync
uv run python -m spacy download en_core_web_lg
uv run uvicorn main:app --reload
```

The spaCy model is required by the original embedding endpoint. The download
command ensures it is installed after `uv sync`, which can remove packages that
are not listed in the project dependencies. It reuses the download cache.

Open http://127.0.0.1:8000/docs. Under `POST /cnn/predict`, choose **Try it out**,
upload `models/example_cifar10.png`, and choose **Execute**. The response contains
the predicted class, a softmax confidence score, and probabilities for all ten
classes. The supplied image is a cat. Confidence is not a calibrated guarantee;
the classifier always picks one of the ten CIFAR-10 classes.

Trained weights are included, so no retraining is needed to run this example.
The saved model achieved **64.47% test accuracy (6,447/10,000)** after ten epochs.
Epoch 8 had the highest validation accuracy, **64.74%**, and was selected before
evaluating the separate test set. Full measurements are in
`models/cifar10_cnn.metrics.json`.

## Train the classifier

```sh
uv run python -m cnn_assignment.train --epochs 10
```

This downloads CIFAR-10 to `data/`, resizes 32x32 RGB images to 64x64, normalizes
each channel with mean/std 0.5, and trains on 45,000 examples. It reserves 5,000
examples for validation and evaluates the selected checkpoint on the 10,000
official test examples. Training uses Adam, learning rate 0.001, batch size 128,
and seed 42. Add `--device cpu` to force CPU use. Otherwise CUDA or Apple MPS is
used when available. Runtime and exact results vary by hardware and libraries.

Architecture: Conv(3->16, 3x3, stride 1, padding 1), ReLU, MaxPool(2,2),
Conv(16->32, 3x3, stride 1, padding 1), ReLU, MaxPool(2,2), Flatten(8192),
Linear(8192->100), ReLU, Linear(100->10). Total: 825,398 trainable parameters.
The model returns logits for cross-entropy loss. The API applies softmax.

The trainer saves `models/cifar10_cnn.pt`, metrics, and an example image. Restart
the server after retraining because the API caches the loaded model. Set
`CNN_MODEL_PATH` before starting the server to use another checkpoint location.
Missing or invalid weights produce HTTP 503 at prediction time; model loading is
lazy and does not replace the original application startup or routes.

## Checks and theory answers

```sh
uv sync --extra test
uv run --extra test python verify_assignment2.py
```

The checks cover every layer shape, parameter count, a gradient update, image
uploads, checkpoint loading, invalid input, and preservation of a sample route.
The six written answers are in `THEORY_ANSWERS.txt`.

Implementation references: [PyTorch modules](https://docs.pytorch.org/docs/stable/notes/modules.html),
[gradient modes](https://docs.pytorch.org/docs/stable/notes/autograd), and
[FastAPI routers](https://fastapi.tiangolo.com/tutorial/bigger-applications/).
