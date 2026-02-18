"""
LoRA Manager tab — upload, list, and delete LoRA adapters on RunPod network storage.

Only LoRAs for the Illustrious-XL (SDXL) base model are managed here.
The base model itself (illustrious-xl) is pre-cached in the Docker image and
does not need to be uploaded manually.
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

def _refresh_loras(client: RunPodClient) -> Tuple[List[List], str]:
    """Fetch the LoRA list from RunPod and return (table_data, status_text)."""
    try:
        models = client.list_models("lora")
        rows = [
            [m.get("name", ""), f"{m.get('size_mb', 0):.1f} MB", m.get("path", "")]
            for m in models
        ]
        return rows, f"✅ Found {len(rows)} LoRA(s)."
    except RunPodError as exc:
        return [], f"❌ Error: {exc}"


def _upload_lora(
    file_obj,
    client: RunPodClient,
) -> str:
    """Upload a LoRA file to RunPod network storage."""
    if file_obj is None:
        return "⚠️ No file selected."
    try:
        with open(file_obj.name, "rb") as fh:
            file_data = fh.read()
        filename = os.path.basename(file_obj.name)
        result = client.upload_model(filename, "lora", file_data, base_model="sdxl")
        return f"✅ Uploaded: {result.get('path', filename)} ({result.get('size_mb', '?')} MB)"
    except RunPodError as exc:
        return f"❌ Upload failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"❌ Unexpected error: {exc}"


def _delete_lora(
    selected_rows: Optional[List],
    table_data: List[List],
    client: RunPodClient,
) -> Tuple[str, List[List]]:
    """Delete the selected LoRA from RunPod network storage."""
    if not selected_rows:
        return "⚠️ No LoRA selected.", table_data

    # Gradio Dataframe selection returns row indices
    try:
        row_idx = selected_rows[0] if isinstance(selected_rows[0], int) else int(selected_rows[0][0])
        path = table_data[row_idx][2]  # 3rd column is the storage path
    except (IndexError, ValueError, TypeError):
        return "⚠️ Could not determine selected LoRA path.", table_data

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
    """Render the LoRA Manager tab inside a Gradio Blocks context."""
    with gr.Tab("🎛️ LoRA Manager"):
        gr.Markdown("## LoRA Manager")
        gr.Markdown(
            "Upload, list, and delete **LoRA adapters** stored on the RunPod network volume. "
            "LoRAs are applied on top of the pre-cached **Illustrious-XL** base model. "
            "The base model itself does not need to be uploaded here."
        )

        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh List", variant="secondary")

        # LoRA table
        lora_table = gr.Dataframe(
            headers=["Name", "Size", "Storage Path"],
            datatype=["str", "str", "str"],
            interactive=False,
            label="Available LoRAs",
            row_count=(5, "dynamic"),
        )

        status_text = gr.Textbox(label="Status", interactive=False, lines=1)

        gr.Markdown("---")
        gr.Markdown("### Upload New LoRA")
        gr.Markdown(
            "_Supported formats: `.safetensors`, `.pt`, `.bin`. "
            "LoRAs must be trained for SDXL (Illustrious-XL compatible)._"
        )

        with gr.Row():
            upload_file = gr.File(
                label="Select LoRA file (.safetensors, .pt, .bin)",
                file_types=[".safetensors", ".pt", ".bin"],
                scale=4,
            )
            upload_btn = gr.Button("⬆️ Upload", variant="secondary", scale=1)

        upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=1)

        gr.Markdown("---")
        gr.Markdown("### Delete LoRA")
        gr.Markdown("_Select a row in the table above, then click Delete._")

        with gr.Row():
            selected_row_state = gr.State(value=None)
            delete_btn = gr.Button("🗑️ Delete Selected", variant="stop")

        # ------------------------------------------------------------------
        # Event wiring
        # ------------------------------------------------------------------

        def on_refresh():
            rows, msg = _refresh_loras(client)
            return rows, msg

        refresh_btn.click(
            fn=on_refresh,
            inputs=[],
            outputs=[lora_table, status_text],
        )

        def on_upload(file_obj):
            return _upload_lora(file_obj, client)

        upload_btn.click(
            fn=on_upload,
            inputs=[upload_file],
            outputs=[upload_status],
        )

        # Track selected row via dataframe select event
        def on_row_select(evt: gr.SelectData):
            return [evt.index[0]] if evt.index else None

        lora_table.select(
            fn=on_row_select,
            inputs=None,
            outputs=[selected_row_state],
        )

        def on_delete(selected_rows, table_data: List[List]):
            msg, new_data = _delete_lora(selected_rows, table_data, client)
            return new_data, msg

        delete_btn.click(
            fn=on_delete,
            inputs=[selected_row_state, lora_table],
            outputs=[lora_table, status_text],
        )
