"""
Shared Modal app instance, container images, and volume definitions.

All GPU service modules import ``app``, ``caption_image``, ``upscale_image``,
and ``lora_volume`` from here so that every ``@app.cls()`` decorator registers
against the same ``modal.App("ai-toolbox")`` instance.

The main entrypoint (``modal_app.py`` at the repo root) imports the service
classes from ``src.gpu.*`` to trigger their registration, then deploys with:

    modal deploy modal_app.py
"""

import modal

# ---------------------------------------------------------------------------
# Shared Modal app
# ---------------------------------------------------------------------------

app = modal.App("ai-toolbox")

# ---------------------------------------------------------------------------
# Constants — model paths baked into the container images at build time
# ---------------------------------------------------------------------------

_CAPTION_MODEL_REPO = "Qwen/Qwen3-VL-2B-Instruct"
_CAPTION_MODEL_PATH = "/app/models/qwen3-vl-2b"

_MODELS_DIR = "/app/models"
_ILLUSTRIOUS_XL_PATH = f"{_MODELS_DIR}/illustrious-xl/Illustrious-XL-v2.0.safetensors"
_CONTROLNET_TILE_PATH = f"{_MODELS_DIR}/controlnet-tile"
_SDXL_VAE_PATH = f"{_MODELS_DIR}/sdxl-vae"
_IP_ADAPTER_PATH = f"{_MODELS_DIR}/ip-adapter/ip-adapter-plus_sdxl_vit-h.safetensors"
_CLIP_VIT_H_PATH = f"{_MODELS_DIR}/clip-vit-h"

# ---------------------------------------------------------------------------
# Caption image
# ---------------------------------------------------------------------------
# Qwen3-VL-2B-Instruct weights (~4 GB) are downloaded at image-build time so
# cold starts never need to fetch weights from HuggingFace.

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
    # Download Qwen3-VL-2B-Instruct weights into the image layer at build time.
    # snapshot_download fetches all model files (config, tokenizer, weights)
    # and stores them at _CAPTION_MODEL_PATH so the container never needs
    # network access to HuggingFace at runtime.
    .run_commands(
        f"python -c \""
        f"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{_CAPTION_MODEL_REPO}', local_dir='{_CAPTION_MODEL_PATH}')"
        f"\""
    )
    # Mount the local `src` package so that `from src.xxx import ...` works
    # inside the container.  Placed last so code changes don't bust the
    # expensive model-download cache layers above.
    .add_local_python_source("src")
)

# ---------------------------------------------------------------------------
# Upscale image
# ---------------------------------------------------------------------------
# All model weights (Illustrious-XL, ControlNet Tile, SDXL VAE, IP-Adapter,
# CLIP ViT-H) are downloaded at image-build time.

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
    # -----------------------------------------------------------------------
    # Download Illustrious-XL v2.0 single-file checkpoint
    # -----------------------------------------------------------------------
    .run_commands(
        f"mkdir -p {_MODELS_DIR}/illustrious-xl && "
        f"wget -q -O {_ILLUSTRIOUS_XL_PATH} "
        f"'https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0/resolve/main/Illustrious-XL-v2.0.safetensors'"
    )
    # -----------------------------------------------------------------------
    # Download ControlNet Tile SDXL (xinsir/controlnet-tile-sdxl-1.0)
    # -----------------------------------------------------------------------
    .run_commands(
        f"mkdir -p {_CONTROLNET_TILE_PATH} && "
        f"python -c \""
        f"from huggingface_hub import hf_hub_download; "
        f"hf_hub_download('xinsir/controlnet-tile-sdxl-1.0', 'config.json', local_dir='{_CONTROLNET_TILE_PATH}'); "
        f"hf_hub_download('xinsir/controlnet-tile-sdxl-1.0', 'diffusion_pytorch_model.safetensors', local_dir='{_CONTROLNET_TILE_PATH}')"
        f"\""
    )
    # -----------------------------------------------------------------------
    # Download SDXL VAE fp16-fix (madebyollin/sdxl-vae-fp16-fix)
    # -----------------------------------------------------------------------
    .run_commands(
        f"mkdir -p {_SDXL_VAE_PATH} && "
        f"python -c \""
        f"from huggingface_hub import hf_hub_download; "
        f"hf_hub_download('madebyollin/sdxl-vae-fp16-fix', 'config.json', local_dir='{_SDXL_VAE_PATH}'); "
        f"hf_hub_download('madebyollin/sdxl-vae-fp16-fix', 'diffusion_pytorch_model.safetensors', local_dir='{_SDXL_VAE_PATH}')"
        f"\""
    )
    # -----------------------------------------------------------------------
    # Download IP-Adapter Plus SDXL ViT-H
    # -----------------------------------------------------------------------
    .run_commands(
        f"mkdir -p {_MODELS_DIR}/ip-adapter && "
        f"wget -q -O {_IP_ADAPTER_PATH} "
        f"'https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors'"
    )
    # -----------------------------------------------------------------------
    # Download CLIP ViT-H-14 (laion/CLIP-ViT-H-14-laion2B-s32B-b79K)
    # -----------------------------------------------------------------------
    .run_commands(
        f"mkdir -p {_CLIP_VIT_H_PATH} && "
        f"python -c \""
        f"from huggingface_hub import hf_hub_download; "
        f"hf_hub_download('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', 'config.json', local_dir='{_CLIP_VIT_H_PATH}'); "
        f"hf_hub_download('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', 'model.safetensors', local_dir='{_CLIP_VIT_H_PATH}'); "
        f"hf_hub_download('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', 'preprocessor_config.json', local_dir='{_CLIP_VIT_H_PATH}')"
        f"\""
    )
    # Mount the local `src` package so that `from src.xxx import ...` works
    # inside the container.  Placed last so code changes don't bust the
    # expensive model-download cache layers above.
    .add_local_python_source("src")
)

# ---------------------------------------------------------------------------
# Gradio image (lightweight CPU-only container for the UI)
# ---------------------------------------------------------------------------

gradio_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gradio>=6.0.0",
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
# Persistent LoRA volume
# ---------------------------------------------------------------------------
# LoRA .safetensors files uploaded by the user are stored here and survive
# container restarts.  The volume is mounted at /vol/loras inside the upscale
# container.

lora_volume = modal.Volume.from_name("ai-toolbox-loras", create_if_missing=True)
