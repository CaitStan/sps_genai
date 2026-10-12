# sps_genai: word embeddings and CIFAR-10 classification

The original `GET /` and `GET /embedding?word=hello` endpoints are preserved.
Assignment 2 adds `POST /cnn/predict` for image classification and
`GET /cnn/health` for checkpoint presence and class labels.

The saved model achieved **64.47% test accuracy (6,447/10,000)** after ten epochs.
Epoch 8 had the highest validation accuracy, **64.74%**, and was selected before
evaluating the separate test set. Full measurements are in
`models/cifar10_cnn.metrics.json`.
