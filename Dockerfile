FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU wheels avoid a CUDA/GPU requirement on the instructor's machine.
RUN python -m pip install --no-cache-dir \
    torch==2.14.1 torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/cpu
COPY requirements-docker.txt ./
RUN python -m pip install --no-cache-dir -r requirements-docker.txt \
    && python -m pip check

COPY main.py docker_smoke_test.py ./
COPY cnn_assignment/ ./cnn_assignment/
COPY gan_assignment/ ./gan_assignment/
COPY helper_lib/ ./helper_lib/
COPY models/cifar10_cnn.pt models/example_cifar10.png ./models/
COPY models/gan/mnist_gan.pt ./models/gan/

# Verify bundled model files and the old embedding dependency during the build.
RUN python -c "import main; from cnn_assignment.api import load_model, checkpoint_path; from gan_assignment.api import load_generator, checkpoint_path as gan_path; load_model(str(checkpoint_path())); load_generator(str(gan_path())); print('All bundled models load successfully')" \
    && useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=12 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=3)" || exit 1
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
