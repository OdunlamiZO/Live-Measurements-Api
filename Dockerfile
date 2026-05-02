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

RUN python -c "\
import urllib.request, os; \
urllib.request.urlretrieve(\
'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',\
'pose_landmarker_heavy.task')"

EXPOSE 5000

# 1 worker — each worker loads the ML models into memory, so more workers
# means proportionally more RAM. Timeout raised to 120s for inference time.
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]
