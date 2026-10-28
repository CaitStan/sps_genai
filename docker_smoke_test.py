"""Exercise the real HTTP server: docker compose exec -T api python docker_smoke_test.py"""
import json
import struct
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as response:
        return response.headers, response.read()


def main():
    assert json.loads(get("/")[1]) == {"message": "Word Embedding API"}
    embedding = json.loads(get("/embedding?word=hello")[1])
    assert embedding["word"] == "hello" and len(embedding["embedding"]) == 300
    assert any(embedding["embedding"])
    for path in ["/cnn/health", "/gan/health"]:
        assert json.loads(get(path)[1])["checkpoint_exists"]
    headers, image = get("/gan/generate?num_samples=16&seed=42")
    assert headers.get_content_type() == "image/png" and image[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", image[16:24]) == (968, 248)
    assert get("/gan/generate?num_samples=16&seed=42")[1] == image
    boundary = "assignment-docker-check"
    sample = Path(__file__).parent / "models" / "example_cifar10.png"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"cat.png\"\r\n"
            "Content-Type: image/png\r\n\r\n").encode() + sample.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(BASE + "/cnn/predict", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        prediction = json.load(response)
    assert prediction["class_name"] == "cat"
    paths = json.loads(get("/openapi.json")[1])["paths"]
    assert all(p in paths for p in ["/", "/embedding", "/cnn/health", "/cnn/predict", "/gan/health", "/gan/generate"])
    assert get("/docs")[0].get_content_type() == "text/html"
    print("PASS: Docker HTTP server, docs, original embedding, CNN prediction, GAN PNG generation, repeatable seed, and all six routes.")


if __name__ == "__main__":
    main()
