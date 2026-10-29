# PCB defect inspection API (YOLO + tiled inference, TensorRT INT8/FP16 or ONNX Runtime)
# Host requirements: NVIDIA driver R580+ and GPU support for containers
# (Docker Desktop + WSL2 on Windows, NVIDIA Container Toolkit on Linux).
ARG BASE_IMAGE=nvidia/cuda:13.4.3-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv curl \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /cache /models && chown -R appuser /cache
USER appuser

ENV MODEL_DIR=/models \
    ENGINE_CACHE=/cache \
    BACKEND=trt \
    PRECISION=fp16

EXPOSE 8000
# First start builds the TensorRT engine for this GPU (1-3 min), hence the long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
