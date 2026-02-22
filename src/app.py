"""
AI Assets Toolbox — Unified Modal App

Deploy:  modal deploy src/app.py
Dev:     modal serve src/app.py

This single entrypoint registers:
  - CaptionService   (T4 GPU)   — Qwen3-VL-2B image captioning
  - UpscaleService   (A10G GPU)  — SDXL tile-based upscaling
  - DownloadService  (CPU)       — model downloads from HuggingFace to Volume
  - Gradio web UI    (CPU)       — served at the app's URL
"""
from __future__ import annotations

import modal
from src.app_config import app, gradio_image

# Import GPU services to register them with the app
from src.gpu import CaptionService, UpscaleService  # noqa: F401

# Import DownloadService to register it with the app
from src.services.download import DownloadService  # noqa: F401

# Import Gradio app creator
from src.ui import create_gradio_app


@app.function(
    image=gradio_image,
    scaledown_window=600,  # 10 min idle for UI
    timeout=3600,  # 1 hour max
)
@modal.concurrent(max_inputs=100)  # Gradio handles many users
@modal.asgi_app()
def web_ui():
    """Serve the Gradio UI as an ASGI app on Modal."""
    import gradio as gr
    blocks = create_gradio_app()
    # In Gradio 6.0, Blocks is no longer directly callable as an ASGI app.
    # Use gr.mount_gradio_app() to mount it onto a FastAPI app instead.
    from fastapi import FastAPI
    fastapi_app = FastAPI()
    fastapi_app = gr.mount_gradio_app(fastapi_app, blocks, path="/")
    return fastapi_app
