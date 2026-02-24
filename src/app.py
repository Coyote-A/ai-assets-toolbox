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
from src.app_config import app, gradio_image, lora_volume
from src.services.model_registry import LORAS_MOUNT_PATH

# Import GPU services to register them with the app
from src.gpu import CaptionService, UpscaleService  # noqa: F401

# Import DownloadService to register it with the app
from src.services.download import DownloadService  # noqa: F401

# Import Gradio app creator
from src.ui import create_gradio_app

# JavaScript patch to fix broken fullscreen button in Gradio.
# In Gradio 6.x, js/head params moved from Blocks constructor to mount_gradio_app().
FULLSCREEN_JS_PATCH = """
function() {
    console.log('[FULLSCREEN PATCH LOADED]');
    document.addEventListener('click', function(e) {
        const btn = e.target.closest('button[aria-label="Fullscreen"], button[title="Fullscreen"]');
        if (!btn) return;
        const root = btn.closest('.image-container, .gr-image, figure, .svelte-image');
        if (!root) return;
        const media = root.querySelector('img') || root.querySelector('video');
        if (!media) return;
        console.log('[FULLSCREEN INTERCEPTED]');
        e.preventDefault();
        e.stopImmediatePropagation();
        if (document.fullscreenElement) {
            (document.exitFullscreen || document.webkitExitFullscreen).call(document);
        } else {
            (media.requestFullscreen || media.webkitRequestFullscreen).call(media);
        }
    }, true);
}
"""


@app.function(
    image=gradio_image,
    scaledown_window=600,  # 10 min idle for UI
    timeout=3600,  # 1 hour max
    volumes={
        LORAS_MOUNT_PATH: lora_volume,  # For token storage
    },
)
@modal.concurrent(max_inputs=100)  # Gradio handles many users
@modal.asgi_app()
def web_ui():
    """Serve the Gradio UI as an ASGI app on Modal."""
    import gradio as gr
    blocks = create_gradio_app()
    # In Gradio 6.6, Blocks is no longer directly callable as an ASGI app.
    # Use gr.mount_gradio_app() to mount it onto a FastAPI app instead.
    # The js= parameter is the official way to inject JS in Gradio 6.x ASGI apps.
    from fastapi import FastAPI
    fastapi_app = FastAPI()
    fastapi_app = gr.mount_gradio_app(fastapi_app, blocks, path="/", js=FULLSCREEN_JS_PATCH)
    return fastapi_app
