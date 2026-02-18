"""
AI Assets Toolbox — Gradio frontend entry point.

Tabs
----
1. 🖼️ Tile Upscale        — main upscaling workflow
2. 🎭 Spritesheet Animation — placeholder (coming soon)
3. 📦 Model Manager        — upload / list / delete models on RunPod storage

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


def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""

    client = RunPodClient(
        api_key=config.RUNPOD_API_KEY,
        endpoint_id=config.RUNPOD_ENDPOINT_ID,
    )

    with gr.Blocks(
        title="AI Assets Toolbox",
        theme=gr.themes.Soft(),
        css="""
            .gradio-container { max-width: 1400px; margin: auto; }
            footer { display: none !important; }
        """,
    ) as demo:
        gr.Markdown(
            """
            # 🎨 AI Assets Toolbox
            GPU-accelerated image upscaling and asset generation powered by RunPod.
            """
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
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
