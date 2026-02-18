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
# Pre-download models into the Docker image
# Models are baked into the image so there is zero cold-start model loading.
# Layer ordering: pip install → model downloads → code copy
# This ensures code changes do NOT invalidate the expensive model cache layers.
# ---------------------------------------------------------------------------
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV DIFFUSERS_CACHE=/app/hf_cache

# 1. Illustrious-XL (primary SDXL diffusion model) — precached
RUN python -c "\
from diffusers import StableDiffusionXLImg2ImgPipeline; \
import torch; \
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained( \
    'OnomaAIResearch/Illustrious-xl-early-release-v0', \
    torch_dtype=torch.float16, \
    use_safetensors=True, \
    variant='fp16', \
); \
print('Illustrious-XL downloaded successfully') \
"

# 2. ControlNet Tile SDXL
RUN python -c "\
from diffusers import ControlNetModel; \
import torch; \
cn = ControlNetModel.from_pretrained( \
    'xinsir/controlnet-tile-sdxl-1.0', \
    torch_dtype=torch.float16, \
); \
print('ControlNet Tile SDXL downloaded successfully') \
"

# 3. SDXL VAE fp16-fix
RUN python -c "\
from diffusers import AutoencoderKL; \
import torch; \
vae = AutoencoderKL.from_pretrained( \
    'madebyollin/sdxl-vae-fp16-fix', \
    torch_dtype=torch.float16, \
); \
print('SDXL VAE fp16-fix downloaded successfully') \
"

# 4. Qwen2.5-VL-7B-Instruct (captioning + image editing)
# NOTE: Qwen2.5-VL-7B-Instruct (~15 GB) is NOT baked into this image.
# Configure it as a RunPod Cached Model on the endpoint so RunPod pre-warms
# it onto the network volume automatically.  The model_manager will find it
# at the RunPod cache path (RUNPOD_MODEL_CACHE env var, default:
# /runpod-volume/huggingface-cache/hub) and fall back to a live HF download
# if the cache is absent.  Removing it here reduces the image from ~35 GB
# to ~15-20 GB.

# 5. IP-Adapter SDXL weights + CLIP ViT-H image encoder (precached for Phase 4)
RUN python -c "\
from huggingface_hub import hf_hub_download; \
path = hf_hub_download( \
    repo_id='h94/IP-Adapter', \
    filename='ip-adapter_sdxl_vit-h.safetensors', \
    subfolder='sdxl_models', \
); \
print(f'IP-Adapter SDXL weights downloaded to: {path}'); \
from transformers import CLIPVisionModelWithProjection; \
encoder = CLIPVisionModelWithProjection.from_pretrained( \
    'h94/IP-Adapter', \
    subfolder='models/image_encoder', \
); \
print('IP-Adapter CLIP image encoder downloaded successfully') \
"

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
