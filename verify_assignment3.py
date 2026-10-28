"""Offline GAN architecture/training/API checks: uv run --extra test python verify_assignment3.py"""
import os
import tempfile
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from helper_lib import get_model, train_gan, generate_samples
from gan_assignment.api import router, load_generator


def main():
    torch.set_num_threads(2)
    torch.manual_seed(1)
    model = get_model("GAN")
    x = torch.randn(4, 100)
    expected_g = [(4,6272), (4,128,7,7), (4,64,14,14), (4,64,14,14),
                  (4,64,14,14), (4,1,28,28), (4,1,28,28)]
    for layer, shape in zip(model.generator.layers, expected_g):
        x = layer(x)
        assert tuple(x.shape) == shape
    assert x.min() >= -1 and x.max() <= 1
    expected_d = [(4,64,14,14), (4,64,14,14), (4,128,7,7), (4,128,7,7),
                  (4,128,7,7), (4,6272), (4,1), (4,1)]
    for layer, shape in zip(model.discriminator.layers, expected_d):
        x = layer(x)
        assert tuple(x.shape) == shape
    assert x.min() >= 0 and x.max() <= 1
    assert sum(p.numel() for p in model.generator.parameters()) == 765761
    assert sum(p.numel() for p in model.discriminator.parameters()) == 138817

    before_g = model.generator.layers[0].weight.detach().clone()
    before_d = model.discriminator.layers[0].weight.detach().clone()
    data = TensorDataset(torch.rand(5,1,28,28) * 2 - 1, torch.zeros(5,dtype=torch.long))
    optimizers = {"generator": torch.optim.Adam(model.generator.parameters(), lr=0.0002),
                  "discriminator": torch.optim.Adam(model.discriminator.parameters(), lr=0.0002)}
    returned = train_gan(model, DataLoader(data,batch_size=4), nn.BCELoss(), optimizers, epochs=1)
    assert returned is model and model.history[0]["examples"] == 5
    assert not torch.equal(before_g,model.generator.layers[0].weight)
    assert not torch.equal(before_d,model.discriminator.layers[0].weight)
    assert all(p.requires_grad for p in model.discriminator.parameters())
    assert torch.isfinite(torch.tensor(model.history[0]["generator_loss"]))

    bn = model.generator.layers[3]
    before_mean = bn.running_mean.clone()
    before_rng = torch.random.get_rng_state().clone()
    samples = generate_samples(model,"cpu",3,seed=9,show=False)
    assert samples.shape == (3,1,28,28) and samples.min() >= 0 and samples.max() <= 1
    torch.testing.assert_close(samples,generate_samples(model,"cpu",3,seed=9,show=False))
    assert not torch.equal(samples,generate_samples(model,"cpu",3,seed=10,show=False))
    assert torch.equal(before_rng,torch.random.get_rng_state())
    assert model.generator.training and torch.equal(before_mean,bn.running_mean)

    app = FastAPI()
    @app.get("/existing")
    def existing():
        return {"preserved": True}
    app.include_router(router)
    original_path = os.environ.get("GAN_MODEL_PATH")
    try:
        with tempfile.TemporaryDirectory() as directory, TestClient(app) as client:
            path = Path(directory) / "test.pt"
            os.environ["GAN_MODEL_PATH"] = str(path)
            load_generator.cache_clear()
            assert client.get('/existing').json() == {"preserved": True}
            assert client.get('/gan/generate').status_code == 503
            torch.save({"generator_state_dict": model.generator.state_dict(), "latent_dim":100,
                        "image_shape":[1,28,28]},path)
            response = client.get('/gan/generate?num_samples=4&seed=9')
            assert response.status_code == 200 and response.headers['content-type']=='image/png'
            with Image.open(BytesIO(response.content)) as image:
                assert image.format == 'PNG' and image.size == (488,128)
            assert client.get('/gan/generate?num_samples=4&seed=9').content == response.content
            for query in ['num_samples=0','num_samples=65','seed=-1','seed=abc']:
                assert client.get('/gan/generate?'+query).status_code == 422
            assert client.get('/gan/health').json()['checkpoint_exists'] is True
    finally:
        load_generator.cache_clear()
        if original_path is None:
            os.environ.pop('GAN_MODEL_PATH',None)
        else:
            os.environ['GAN_MODEL_PATH'] = original_path
    print('PASS: exact GAN shapes/parameters, training both networks including a short final batch,')
    print('sample range, seeds, preserved RNG/BatchNorm state, PNG API, bounds, missing weights, existing route.')


if __name__ == '__main__':
    main()
