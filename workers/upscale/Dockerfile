# ============================================================
# AI Assets Toolbox — Upscale Worker
# ============================================================
# Base image: NVIDIA CUDA 12.1.1 devel on Ubuntu 22.04
# ============================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    wget \
    libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && python -m pip install --upgrade pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

# Network volume path (LoRAs, outputs, etc.)
ENV RUNPOD_VOLUME_PATH=/runpod-volume

# Ensure Python can find the worker modules
ENV PYTHONPATH=/app

# ---------------------------------------------------------------------------
# Pre-download models into the Docker image via wget
# Each model in its own RUN layer for Docker layer caching.
# Layer ordering: pip install → model downloads → code copy
# This ensures code changes do NOT invalidate the expensive model cache layers.
# ---------------------------------------------------------------------------

# 1. Illustrious-XL v2.0 — single safetensors file
RUN mkdir -p /app/models/illustrious-xl \
    && wget -q --show-progress -O /app/models/illustrious-xl/illustrious-xl-v2.0.safetensors \
        "https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0/resolve/main/illustriousXL_v20.safetensors" \
    && echo "Illustrious-XL v2.0 downloaded successfully"

# 2. ControlNet Tile SDXL — config + weights
RUN mkdir -p /app/models/controlnet-tile \
    && wget -q --show-progress -O /app/models/controlnet-tile/config.json \
        "https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0/resolve/main/config.json" \
    && wget -q --show-progress -O /app/models/controlnet-tile/diffusion_pytorch_model.safetensors \
        "https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors" \
    && echo "ControlNet Tile SDXL downloaded successfully"

# 3. SDXL VAE fp16-fix — config + weights
RUN mkdir -p /app/models/sdxl-vae \
    && wget -q --show-progress -O /app/models/sdxl-vae/config.json \
        "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/config.json" \
    && wget -q --show-progress -O /app/models/sdxl-vae/diffusion_pytorch_model.safetensors \
        "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/diffusion_pytorch_model.safetensors" \
    && echo "SDXL VAE fp16-fix downloaded successfully"

# 4. IP-Adapter weights + CLIP ViT-H image encoder
RUN mkdir -p /app/models/ip-adapter \
    && wget -q --show-progress -O /app/models/ip-adapter/ip-adapter-plus_sdxl_vit-h.safetensors \
        "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors" \
    && echo "IP-Adapter weights downloaded successfully"

RUN mkdir -p /app/models/clip-vit-h \
    && wget -q --show-progress -O /app/models/clip-vit-h/config.json \
        "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/config.json" \
    && wget -q --show-progress -O /app/models/clip-vit-h/model.safetensors \
        "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/model.safetensors" \
    && wget -q --show-progress -O /app/models/clip-vit-h/preprocessor_config.json \
        "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/preprocessor_config.json" \
    && echo "CLIP ViT-H image encoder downloaded successfully"

# ---------------------------------------------------------------------------
# Copy worker source code
# (After model downloads so code changes don't bust the model cache layers)
# ---------------------------------------------------------------------------
COPY . /app/

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
RUN sed -i 's/\r$//' /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
