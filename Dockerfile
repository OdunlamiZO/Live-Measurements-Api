FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU-only first (avoids pulling the 2 GB CUDA build)
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest, skipping torch/torchvision (already installed above)
RUN grep -vE "^torch==|^torchvision==" requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
