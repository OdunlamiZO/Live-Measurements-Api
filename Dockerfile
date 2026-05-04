FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgles2 \
    libegl1 \
    libegl-mesa0 \
    libgbm1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the pose model at build time so the container starts instantly
RUN python -c "\
import urllib.request; \
urllib.request.urlretrieve(\
'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',\
'pose_landmarker_heavy.task')"

EXPOSE 5000

# 1 worker — each worker loads the ML models into memory, so more workers
# means proportionally more RAM. Timeout raised to 120s for inference time.
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]
