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
# Pre-download models into the Docker image
# Models are baked into the image so there is zero cold-start model loading.
# Layer ordering: pip install → model downloads → code copy
# This ensures code changes do NOT invalidate the expensive model cache layers.
# ---------------------------------------------------------------------------
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV DIFFUSERS_CACHE=/app/hf_cache
ENV HF_HUB_CACHE=/app/hf_cache

# Ensure huggingface-cli is available for model downloads
RUN python -m pip install --no-cache-dir "huggingface_hub[cli]"

# 1. Illustrious-XL (primary SDXL diffusion model) — precached
RUN huggingface-cli download OnomaAIResearch/Illustrious-XL-v2.0 \
    --cache-dir /app/hf_cache \
    && echo "Illustrious-XL downloaded successfully"

# 2. ControlNet Tile SDXL
RUN huggingface-cli download xinsir/controlnet-tile-sdxl-1.0 \
    --cache-dir /app/hf_cache \
    && echo "ControlNet Tile SDXL downloaded successfully"

# 3. SDXL VAE fp16-fix
RUN huggingface-cli download madebyollin/sdxl-vae-fp16-fix \
    --cache-dir /app/hf_cache \
    && echo "SDXL VAE fp16-fix downloaded successfully"

# 4. Qwen3-VL-8B-Instruct is NOT baked in — it uses RunPod's managed model caching.
# Set "Qwen/Qwen3-VL-8B-Instruct" as the Model in the RunPod endpoint configuration.
# RunPod will pre-download it to /runpod-volume/huggingface-cache/hub on host NVMe (free, no billing).

# 5. IP-Adapter SDXL weights + CLIP ViT-H image encoder (precached for Phase 4)
RUN huggingface-cli download h94/IP-Adapter \
    --cache-dir /app/hf_cache \
    && echo "IP-Adapter downloaded successfully"

# ---------------------------------------------------------------------------
# Copy backend source code
# (After model downloads so code changes don't bust the model cache layers)
# ---------------------------------------------------------------------------
COPY . /app/

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# HuggingFace cache — baked into the image at /app/hf_cache
# LoRAs are still loaded dynamically from the network volume at runtime
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV DIFFUSERS_CACHE=/app/hf_cache
ENV HF_HUB_CACHE=/app/hf_cache

# Network volume path (LoRAs, outputs, etc.)
ENV RUNPOD_VOLUME_PATH=/runpod-volume

# Ensure Python can find the backend modules
ENV PYTHONPATH=/app

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
RUN sed -i 's/\r$//' /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
