# ============================================================
# AI Assets Toolbox — RunPod Serverless Worker
# ============================================================
# Base image: NVIDIA CUDA 12.1.1 devel on Ubuntu 22.04
# ============================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git wget curl \
    libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && pip install --upgrade pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Copy backend source code
# ---------------------------------------------------------------------------
COPY . /app/

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# HuggingFace cache — point to the network volume so weights are persisted
ENV HF_HOME=/runpod-volume/hf_cache
ENV TRANSFORMERS_CACHE=/runpod-volume/hf_cache
ENV DIFFUSERS_CACHE=/runpod-volume/hf_cache

# Ensure Python can find the backend modules
ENV PYTHONPATH=/app

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
RUN sed -i 's/\r$//' /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
