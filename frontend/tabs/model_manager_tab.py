"""
Model Manager tab — upload, list, and delete models on RunPod network storage.
"""
from __future__ import annotations

import sys
import os

# Allow imports from the frontend package root when running as a module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional, Tuple

import gradio as gr

from api_client import RunPodClient, RunPodError


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _refresh_models(
    model_type: str,
    client: RunPodClient,
) -> Tuple[List[List], str]:
    """Fetch the model list from RunPod and return (table_data, status_text)."""
    try:
        models = client.list_models(model_type)
        rows = [
            [m.get("name", ""), m.get("base_model", ""), f"{m.get('size_mb', 0):.1f} MB", m.get("path", "")]
            for m in models
        ]
        return rows, f"✅ Found {len(rows)} model(s) of type '{model_type}'."
    except RunPodError as exc:
        return [], f"❌ Error: {exc}"


def _upload_model(
    file_obj,
    model_type: str,
    base_model: str,
    client: RunPodClient,
) -> str:
    """Upload a model file to RunPod network storage."""
    if file_obj is None:
        return "⚠️ No file selected."
    try:
        with open(file_obj.name, "rb") as fh:
            file_data = fh.read()
        filename = os.path.basename(file_obj.name)
        result = client.upload_model(filename, model_type, file_data, base_model=base_model)
        return f"✅ Uploaded: {result.get('path', filename)} ({result.get('size_mb', '?')} MB)"
    except RunPodError as exc:
        return f"❌ Upload failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"❌ Unexpected error: {exc}"


def _delete_model(
    selected_rows: Optional[List],
    table_data: List[List],
    client: RunPodClient,
) -> Tuple[str, List[List]]:
    """Delete the selected model from RunPod network storage."""
    if not selected_rows:
        return "⚠️ No model selected.", table_data

    # Gradio Dataframe selection returns row indices
    try:
        row_idx = selected_rows[0] if isinstance(selected_rows[0], int) else int(selected_rows[0][0])
        path = table_data[row_idx][3]  # 4th column is the storage path
    except (IndexError, ValueError, TypeError):
        return "⚠️ Could not determine selected model path.", table_data

    try:
        client.delete_model(path)
        new_data = [r for i, r in enumerate(table_data) if i != row_idx]
        return f"✅ Deleted: {path}", new_data
    except RunPodError as exc:
        return f"❌ Delete failed: {exc}", table_data


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def create_model_manager_tab(client: RunPodClient) -> None:
    """Render the Model Manager tab inside a Gradio Blocks context."""
    with gr.Tab("📦 Model Manager"):
        gr.Markdown("## Model Manager")
        gr.Markdown(
            "Upload, list, and delete model checkpoints, LoRAs, and ControlNet weights "
            "stored on the RunPod network volume."
        )

        with gr.Row():
            model_type_dd = gr.Dropdown(
                choices=["lora", "checkpoint", "controlnet"],
                value="lora",
                label="Model Type",
                scale=2,
            )
            base_model_dd = gr.Dropdown(
                choices=["sdxl", "flux"],
                value="sdxl",
                label="Base Model (for upload)",
                scale=2,
            )
            refresh_btn = gr.Button("🔄 Refresh List", scale=1)

        # Model table
        model_table = gr.Dataframe(
            headers=["Name", "Base Model", "Size", "Storage Path"],
            datatype=["str", "str", "str", "str"],
            interactive=False,
            label="Available Models",
            row_count=(5, "dynamic"),
        )

        status_text = gr.Textbox(label="Status", interactive=False, lines=1)

        gr.Markdown("---")
        gr.Markdown("### Upload New Model")

        with gr.Row():
            upload_file = gr.File(
                label="Select model file (.safetensors, .ckpt, .pt, .bin)",
                file_types=[".safetensors", ".ckpt", ".pt", ".bin"],
                scale=4,
            )
            upload_btn = gr.Button("⬆️ Upload", scale=1)

        upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=1)

        gr.Markdown("---")
        gr.Markdown("### Delete Model")
        gr.Markdown(
            "_Select a row in the table above, then click Delete._"
        )

        with gr.Row():
            selected_row_state = gr.State(value=None)
            delete_btn = gr.Button("🗑️ Delete Selected", variant="stop")

        # ------------------------------------------------------------------
        # Event wiring
        # ------------------------------------------------------------------

        def on_refresh(model_type: str):
            rows, msg = _refresh_models(model_type, client)
            return rows, msg

        refresh_btn.click(
            fn=on_refresh,
            inputs=[model_type_dd],
            outputs=[model_table, status_text],
        )

        # Also refresh when model type dropdown changes
        model_type_dd.change(
            fn=on_refresh,
            inputs=[model_type_dd],
            outputs=[model_table, status_text],
        )

        def on_upload(file_obj, model_type: str, base_model: str):
            return _upload_model(file_obj, model_type, base_model, client)

        upload_btn.click(
            fn=on_upload,
            inputs=[upload_file, model_type_dd, base_model_dd],
            outputs=[upload_status],
        )

        # Track selected row via dataframe select event
        def on_row_select(evt: gr.SelectData):
            return [evt.index[0]] if evt.index else None

        model_table.select(
            fn=on_row_select,
            inputs=None,
            outputs=[selected_row_state],
        )

        def on_delete(selected_rows, table_data: List[List]):
            msg, new_data = _delete_model(selected_rows, table_data, client)
            return new_data, msg

        delete_btn.click(
            fn=on_delete,
            inputs=[selected_row_state, model_table],
            outputs=[model_table, status_text],
        )
