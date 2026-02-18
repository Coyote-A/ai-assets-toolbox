# ============================================================
# AI Assets Toolbox — RunPod Serverless Worker
# ============================================================
# Base image: RunPod PyTorch 2.2 with CUDA 12.1 on Ubuntu 22.04
# ============================================================
FROM runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

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
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
