"""
AI Assets Toolbox — Gradio frontend entry point.

Tabs
----
1. 🖼️ Tile Upscale        — main upscaling workflow
2. 🎭 Spritesheet Animation — placeholder (coming soon)

Usage
-----
    cd frontend
    python app.py

Environment variables (or .env file):
    RUNPOD_API_KEY              — RunPod API key
    RUNPOD_UPSCALE_ENDPOINT_ID  — RunPod upscale worker endpoint ID
    RUNPOD_CAPTION_ENDPOINT_ID  — RunPod caption worker endpoint ID
    RUNPOD_ENDPOINT_ID          — Legacy single-endpoint fallback (optional)
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

# Re-export new config names for convenience
from config import (  # noqa: F401
    RUNPOD_API_KEY,
    RUNPOD_UPSCALE_ENDPOINT_ID,
    RUNPOD_CAPTION_ENDPOINT_ID,
)

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

/* ── Tile grid (HTML/CSS component) ──────────────────────────────────────── */
#tile-grid-container {
    width: 100%;
}
/* Container: full image as background, tiles are absolute overlays */
.tile-grid-wrap {
    position: relative;
    width: 100%;
    border-radius: 8px;
    overflow: hidden;
    background-size: 100% 100%;
    background-repeat: no-repeat;
}
.tile-grid-wrap .tile {
    position: absolute;
    cursor: pointer;
    overflow: hidden;
    box-shadow: inset 0 0 0 0.5px rgba(255, 255, 255, 0.12);
    transition: box-shadow 0.15s ease;
    background-repeat: no-repeat;
    background-size: 100% 100%;
}
/* Selection dimming */
.tile-grid-wrap.has-selection .tile:not(.selected)::before {
    content: '';
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.45);
    pointer-events: none;
    z-index: 2;
    transition: background 0.15s ease;
}
.tile-grid-wrap .tile.selected {
    z-index: 3;
    box-shadow: inset 0 0 0 0.5px rgba(255, 255, 255, 0.12),
                inset 0 0 0 2px rgba(30, 111, 255, 0.8);
}
.tile-grid-wrap .tile[data-status=processed]::after {
    content: '';
    position: absolute;
    inset: 0;
    background: rgba(0, 200, 80, 0.08);
    pointer-events: none;
    z-index: 1;
}
.tile-grid-wrap .tile[data-status=processing] {
    animation: tile-pulse 1.2s ease-in-out infinite;
}
@keyframes tile-pulse {
    0%, 100% { box-shadow: inset 0 0 0 0.5px rgba(255,255,255,0.12), 0 0 0 0 rgba(255, 167, 38, 0.4); }
    50%       { box-shadow: inset 0 0 0 0.5px rgba(255,255,255,0.12), 0 0 0 4px rgba(255, 167, 38, 0); }
}
.tile-grid-wrap .tile:hover {
    box-shadow: inset 0 0 0 0.5px rgba(255, 255, 255, 0.12),
                inset 0 0 0 1.5px rgba(74, 128, 255, 0.6);
}
.tile-grid-wrap .tile-label {
    position: absolute;
    top: 3px;
    left: 3px;
    background: rgba(0, 0, 0, 0.7);
    color: #fff;
    font-size: 11px;
    padding: 1px 5px;
    border-radius: 3px;
    pointer-events: none;
    font-family: monospace;
    z-index: 5;
}
.tile-grid-wrap .tile-status-icon {
    position: absolute;
    bottom: 3px;
    right: 3px;
    font-size: 14px;
    pointer-events: none;
    z-index: 5;
}
.tile-grid-wrap .tile[data-status=processed] .tile-status-icon::after {
    content: '✓';
    color: #00c853;
    text-shadow: 0 0 3px rgba(0, 0, 0, 0.8);
}
.tile-grid-wrap .tile[data-status=processing] .tile-status-icon::after { content: '⏳'; }
/* Overlap zone strips */
.tile-grid-wrap .tile .overlap-zone {
    position: absolute;
    pointer-events: none;
    z-index: 4;
    display: none;
}
.tile-grid-wrap .tile.selected .overlap-zone { display: block; }
.tile-grid-wrap .tile .overlap-zone.left {
    left: 0; top: 0; bottom: 0;
    background: linear-gradient(to right, rgba(255, 165, 0, 0.35), transparent);
}
.tile-grid-wrap .tile .overlap-zone.right {
    right: 0; top: 0; bottom: 0;
    background: linear-gradient(to left, rgba(255, 165, 0, 0.35), transparent);
}
.tile-grid-wrap .tile .overlap-zone.top {
    left: 0; right: 0; top: 0;
    background: linear-gradient(to bottom, rgba(255, 165, 0, 0.35), transparent);
}
.tile-grid-wrap .tile .overlap-zone.bottom {
    left: 0; right: 0; bottom: 0;
    background: linear-gradient(to top, rgba(255, 165, 0, 0.35), transparent);
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
    .tile-grid-wrap .tile-label { font-size: 9px; }
}
"""


def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""

    client = RunPodClient(
        api_key=config.RUNPOD_API_KEY,
        upscale_endpoint_id=config.RUNPOD_UPSCALE_ENDPOINT_ID,
        caption_endpoint_id=config.RUNPOD_CAPTION_ENDPOINT_ID,
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
        if not config.RUNPOD_API_KEY or not config.RUNPOD_UPSCALE_ENDPOINT_ID:
            gr.Markdown(
                "⚠️ **Configuration missing** — set `RUNPOD_API_KEY` and "
                "`RUNPOD_UPSCALE_ENDPOINT_ID` (and optionally `RUNPOD_CAPTION_ENDPOINT_ID`) "
                "in your `.env` file or environment before making API calls.",
                elem_id="config-warning",
            )

        create_upscale_tab(client)
        create_spritesheet_tab()

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
