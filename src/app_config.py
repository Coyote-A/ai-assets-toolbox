"""
Shared Modal app instance, container images, and volume definitions.

All GPU service modules import ``app``, ``caption_image``, ``upscale_image``,
and ``lora_volume`` from here so that every ``@app.cls()`` decorator registers
against the same ``modal.App("ai-toolbox")`` instance.

The main entrypoint (``modal_app.py`` at the repo root) imports the service
classes from ``src.gpu.*`` to trigger their registration, then deploys with:

    modal deploy modal_app.py
"""

from __future__ import annotations

import modal

from src.services.model_registry import (
    LORAS_MOUNT_PATH,
    LORAS_VOLUME_NAME,
    MODELS_MOUNT_PATH,
    MODELS_VOLUME_NAME,
)

# ---------------------------------------------------------------------------
# Shared Modal app
# ---------------------------------------------------------------------------

app = modal.App("ai-toolbox")

# ---------------------------------------------------------------------------
# Caption image
# ---------------------------------------------------------------------------
# Lightweight pip-only image — model weights are loaded from the models volume
# at runtime (mounted at MODELS_MOUNT_PATH).

caption_image = (
    modal.Image.debian_slim(python_version="3.11")
    # System libraries required by some transformers / Pillow backends
    .apt_install("libgl1", "libglib2.0-0")
    # Python dependencies — pinned to minimum compatible versions
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "transformers>=4.53.0",
        "accelerate>=0.30.0",
        "safetensors>=0.4.0",
        "Pillow>=10.0.0",
        "qwen-vl-utils",
        "sentencepiece",
        "huggingface_hub",
    )
    # Mount the local `src` package so that `from src.xxx import ...` works
    # inside the container.
    .add_local_python_source("src")
)

# ---------------------------------------------------------------------------
# Upscale image
# ---------------------------------------------------------------------------
# Lightweight pip-only image — all model weights (Illustrious-XL, ControlNet
# Tile, SDXL VAE, IP-Adapter, CLIP ViT-H) are loaded from the models volume
# at runtime (mounted at MODELS_MOUNT_PATH).

upscale_image = (
    modal.Image.debian_slim(python_version="3.11")
    # System libraries required by Pillow / OpenCV backends
    .apt_install("libgl1", "libglib2.0-0", "wget")
    # Python dependencies
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "diffusers>=0.31.0",
        "transformers>=4.51.0",
        "accelerate>=0.30.0",
        "safetensors>=0.4.0",
        "Pillow>=10.0.0",
        "peft>=0.12.0",
        "controlnet-aux",
        "compel>=2.0.2",
        "huggingface_hub",
        "requests",
    )
    # Mount the local `src` package so that `from src.xxx import ...` works
    # inside the container.
    .add_local_python_source("src")
)

# ---------------------------------------------------------------------------
# Gradio image (lightweight CPU-only container for the UI)
# ---------------------------------------------------------------------------

gradio_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gradio>=6.6.0",
        "fastapi",
        "Pillow>=10.0.0",
        "numpy",
        "modal",
    )
    # Mount the local `src` package so that `from src.xxx import ...` works
    # inside the container.
    .add_local_python_source("src")
)

# ---------------------------------------------------------------------------
# Download image (lightweight CPU-only container for the download service)
# ---------------------------------------------------------------------------
# Used by the setup-wizard DownloadService to fetch model weights from
# HuggingFace and write them to the models volume.

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub[hf_transfer]>=0.25",
        "requests>=2.31",
        "safetensors>=0.4",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("src")
)

# ---------------------------------------------------------------------------
# Persistent volumes
# ---------------------------------------------------------------------------
# models_volume — base model weights downloaded by the setup wizard.
# lora_volume   — LoRA .safetensors files uploaded by the user.
# Both survive container restarts.

models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)
lora_volume = modal.Volume.from_name(LORAS_VOLUME_NAME, create_if_missing=True)

# ---------------------------------------------------------------------------
# Token storage (Modal Dict for server-side persistence)
# ---------------------------------------------------------------------------
# Stores API tokens (HuggingFace, CivitAI) keyed by browser session ID.
# This persists across server restarts, unlike Gradio's BrowserState which
# uses encryption keys that rotate on server restart.

token_store = modal.Dict.from_name("ai-toolbox-tokens", create_if_missing=True)

# ---------------------------------------------------------------------------
# Metadata extraction cache (Modal Dict for LLM response caching)
# ---------------------------------------------------------------------------
# Caches extracted metadata from model descriptions to avoid repeated LLM calls.
# Keyed by SHA256 hash of description text. Values are JSON dicts with
# extracted fields: trigger_words, recommended_weight, clip_skip, tags, usage_notes.

metadata_extraction_cache = modal.Dict.from_name(
    "ai-toolbox-metadata-cache", create_if_missing=True
)
