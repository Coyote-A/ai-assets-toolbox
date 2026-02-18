"""
AI Assets Toolbox — Gradio frontend entry point.

Tabs
----
1. 🖼️ Tile Upscale        — main upscaling workflow
2. 🎭 Spritesheet Animation — placeholder (coming soon)
3. 🎛️ LoRA Manager         — upload / list / delete LoRAs on RunPod storage

Usage
-----
    cd frontend
    python app.py

Environment variables (or .env file):
    RUNPOD_API_KEY      — RunPod API key
    RUNPOD_ENDPOINT_ID  — RunPod serverless endpoint ID
"""
import sys
import os

# Ensure the frontend directory is on the Python path so that sibling
# modules (config, api_client, tiling, tabs.*) can be imported directly.
_FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

import gradio as gr

import config
from api_client import RunPodClient
from tabs.upscale_tab import create_upscale_tab
from tabs.spritesheet_tab import create_spritesheet_tab
from tabs.model_manager_tab import create_model_manager_tab

_CSS = """
/* ── Layout ─────────────────────────────────────────────────────────────── */
.gradio-container { max-width: 1400px; margin: auto; }
footer { display: none !important; }

/* ── App header ──────────────────────────────────────────────────────────── */
#app-header {
    text-align: center;
    padding: 12px 0 4px;
    border-bottom: 1px solid #2a2a3a;
    margin-bottom: 12px;
}
#app-header h1 { font-size: 1.6em; margin: 0; }
#app-header p  { color: #888; margin: 2px 0 0; font-size: 0.9em; }

/* ── Config warning banner ───────────────────────────────────────────────── */
#config-warning {
    background: #3a1a00;
    border: 1px solid #a04000;
    border-radius: 6px;
    padding: 8px 14px;
    margin-bottom: 10px;
    color: #ffb060;
}

/* ── Tile grid gallery ───────────────────────────────────────────────────── */
.tile-grid-gallery .thumbnail-item {
    border: 2px solid transparent;
    border-radius: 4px;
    transition: border-color 0.15s ease;
}
.tile-grid-gallery .thumbnail-item:hover {
    border-color: #4a80ff;
    cursor: pointer;
}
.tile-grid-gallery .thumbnail-item.selected {
    border-color: #1e6fff;
    box-shadow: 0 0 0 2px #1e6fff55;
}

/* ── Selected tile detail panel ──────────────────────────────────────────── */
.selected-tile-panel {
    border: 1px solid #2e3050;
    border-radius: 8px;
    padding: 10px 12px;
    margin-top: 8px;
    background: #12131e;
}

/* ── Action buttons ──────────────────────────────────────────────────────── */
.action-buttons { gap: 8px; }
.action-buttons button { font-size: 0.95em; padding: 10px 16px; }
.action-buttons button.primary {
    font-size: 1.05em;
    font-weight: 600;
    box-shadow: 0 2px 8px #1e6fff44;
}

/* ── Accordion headers ───────────────────────────────────────────────────── */
.gradio-accordion > .label-wrap {
    background: #1a1b2e;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.gradio-accordion > .label-wrap:hover { background: #22243a; }

/* ── Status bar ──────────────────────────────────────────────────────────── */
#status-bar textarea {
    font-size: 0.85em;
    color: #a0c8a0;
    background: #0e1a0e;
    border-color: #1e3a1e;
    border-radius: 4px;
}

/* ── Responsive: narrow screens ─────────────────────────────────────────── */
@media (max-width: 900px) {
    .gradio-container { max-width: 100%; padding: 0 8px; }
    .tile-grid-gallery { height: auto !important; }
}
"""


def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""

    client = RunPodClient(
        api_key=config.RUNPOD_API_KEY,
        endpoint_id=config.RUNPOD_ENDPOINT_ID,
    )

    with gr.Blocks(
        title="AI Assets Toolbox",
        theme=gr.themes.Soft(),
        css=_CSS,
    ) as demo:
        gr.Markdown(
            """
            # 🎨 AI Assets Toolbox
            GPU-accelerated tile-based image upscaling powered by Illustrious-XL on RunPod.
            """,
            elem_id="app-header",
        )

        # Warn if credentials are missing
        if not config.RUNPOD_API_KEY or not config.RUNPOD_ENDPOINT_ID:
            gr.Markdown(
                "⚠️ **Configuration missing** — set `RUNPOD_API_KEY` and "
                "`RUNPOD_ENDPOINT_ID` in your `.env` file or environment before "
                "making API calls.",
                elem_id="config-warning",
            )

        create_upscale_tab(client)
        create_spritesheet_tab()
        create_model_manager_tab(client)

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
