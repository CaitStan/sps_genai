"""Offline checks: python verify_assignment2.py (requires httpx)."""
import os
import tempfile
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from cnn_assignment.api import load_model, router
from cnn_assignment.common import CLASSES
from cnn_assignment.model import AssignmentCNN


def main():
    torch.set_num_threads(2)
    model = AssignmentCNN()
    x = torch.randn(2, 3, 64, 64)
    expected = [(2,16,64,64), (2,16,64,64), (2,16,32,32),
                (2,32,32,32), (2,32,32,32), (2,32,16,16),
                (2,8192), (2,100), (2,100), (2,10)]
    for layer, shape in zip(model.layers, expected):
        x = layer(x)
        assert tuple(x.shape) == shape
    assert sum(p.numel() for p in model.parameters()) == 825398
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = model.layers[0].weight.detach().clone()
    loss = torch.nn.functional.cross_entropy(x, torch.tensor([0, 1]))
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.layers[0].weight)
    model.train()
    training_output = model(torch.ones(1, 3, 64, 64))
    model.eval()
    torch.testing.assert_close(training_output, model(torch.ones(1, 3, 64, 64)))

    app = FastAPI()
    @app.get("/existing-example")
    def old_route():
        return {"still_here": True}
    app.include_router(router)
    client = TestClient(app)
    buffer = BytesIO()
    Image.new("RGB", (32, 32), "red").save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    old_path = os.environ.get("CNN_MODEL_PATH")
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.pt"
            os.environ["CNN_MODEL_PATH"] = str(path)
            load_model.cache_clear()
            assert client.get("/existing-example").json() == {"still_here": True}
            assert client.get("/cnn/health").json()["checkpoint_exists"] is False
            assert client.post("/cnn/predict", files={"file": ("x.png", image_bytes)}).status_code == 503
            torch.save({"model_state_dict": model.state_dict(), "classes": list(CLASSES), "image_size": 64}, path)
            response = client.post("/cnn/predict", files={"file": ("x.png", image_bytes)})
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["class_name"] in CLASSES
            assert abs(sum(body["probabilities"].values()) - 1) < 1e-5
            assert client.post("/cnn/predict", files={"file": ("bad.txt", b"invalid")}).status_code == 400
            assert client.post("/cnn/predict", files={"file": ("large.png", b"x" * (10 * 1024 * 1024 + 1))}).status_code == 413
    finally:
        load_model.cache_clear()
        if old_path is None:
            os.environ.pop("CNN_MODEL_PATH", None)
        else:
            os.environ["CNN_MODEL_PATH"] = old_path
    print("PASS: all architecture shapes, 825398 parameters, gradient update, mode equivalence, checkpoint loading,")
    print("prediction, probabilities, invalid/oversized uploads, missing model, and existing-route preservation.")


if __name__ == "__main__":
    main()
