"""
Model management action handlers.

Provides list, upload, and delete operations for model files stored on the
RunPod Network Volume.

Directory layout on the volume
-------------------------------
/runpod-volume/models/
    checkpoints/   — SDXL base checkpoints
    loras/         — LoRA .safetensors files
    controlnets/   — ControlNet model dirs
"""

import base64
import logging
import os
from typing import Any

from model_manager import CHECKPOINTS_DIR, LORAS_DIR, CONTROLNETS_DIR

logger = logging.getLogger(__name__)

# Map model_type strings to filesystem directories
_TYPE_TO_DIR: dict[str, str] = {
    "checkpoint": CHECKPOINTS_DIR,
    "lora": LORAS_DIR,
    "controlnet": CONTROLNETS_DIR,
}


def _resolve_dir(model_type: str) -> str:
    """Return the directory for the given model_type, raising on unknown types."""
    directory = _TYPE_TO_DIR.get(model_type.lower())
    if directory is None:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Valid types: {list(_TYPE_TO_DIR.keys())}"
        )
    return directory


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

def handle_list_models(job_input: dict[str, Any]) -> dict[str, Any]:
    """
    List model files available on the network volume.

    Request fields
    --------------
    model_type : str
        One of ``"checkpoint"``, ``"lora"``, ``"controlnet"``.
    base_model_filter : str, optional
        If provided, only entries whose name contains this string are returned.

    Returns
    -------
    dict
        ``{"models": [{"name": ..., "path": ..., "size_mb": ..., "base_model": ...}]}``
    """
    model_type: str = job_input.get("model_type", "lora")
    base_model_filter: str = job_input.get("base_model_filter", "")

    directory = _resolve_dir(model_type)
    os.makedirs(directory, exist_ok=True)

    logger.info("Listing models in '%s' (filter='%s')", directory, base_model_filter)

    models: list[dict[str, Any]] = []
    for entry in sorted(os.scandir(directory), key=lambda e: e.name):
        if base_model_filter and base_model_filter.lower() not in entry.name.lower():
            continue

        size_bytes = entry.stat().st_size if entry.is_file() else _dir_size(entry.path)
        size_mb = round(size_bytes / (1024 * 1024), 2)
        rel_path = os.path.relpath(entry.path, "/runpod-volume")

        models.append(
            {
                "name": entry.name,
                "path": rel_path.replace("\\", "/"),
                "size_mb": size_mb,
                "base_model": base_model_filter or "unknown",
            }
        )

    logger.info("Found %d model(s)", len(models))
    return {"models": models}


# ---------------------------------------------------------------------------
# upload_model
# ---------------------------------------------------------------------------

def handle_upload_model(job_input: dict[str, Any]) -> dict[str, Any]:
    """
    Save a base64-encoded model file to the network volume.

    Request fields
    --------------
    filename : str
        Target filename (e.g. ``"my-lora.safetensors"``).
    model_type : str
        One of ``"checkpoint"``, ``"lora"``, ``"controlnet"``.
    base_model : str, optional
        Sub-directory hint (e.g. ``"sdxl"``).  Not used for path resolution
        in this implementation but returned in the response for reference.
    file_b64 : str
        Base64-encoded file content.
    chunk_index : int, optional
        Zero-based chunk index for chunked uploads.  Defaults to 0.
    total_chunks : int, optional
        Total number of chunks.  Defaults to 1.

    Returns
    -------
    dict
        ``{"status": "complete"|"partial", "path": ..., "size_mb": ...}``
    """
    filename: str = job_input.get("filename", "")
    model_type: str = job_input.get("model_type", "lora")
    base_model: str = job_input.get("base_model", "")
    file_b64: str = job_input.get("file_b64", "")
    chunk_index: int = int(job_input.get("chunk_index", 0))
    total_chunks: int = int(job_input.get("total_chunks", 1))

    if not filename:
        raise ValueError("'filename' is required for upload_model")
    if not file_b64:
        raise ValueError("'file_b64' is required for upload_model")

    directory = _resolve_dir(model_type)
    os.makedirs(directory, exist_ok=True)

    dest_path = os.path.join(directory, filename)

    # For chunked uploads, append to the file; for single-chunk, write directly
    write_mode = "ab" if chunk_index > 0 else "wb"
    file_bytes = base64.b64decode(file_b64)

    logger.info(
        "Uploading '%s' chunk %d/%d (%d bytes) → '%s'",
        filename,
        chunk_index + 1,
        total_chunks,
        len(file_bytes),
        dest_path,
    )

    with open(dest_path, write_mode) as fh:
        fh.write(file_bytes)

    is_complete = (chunk_index + 1) >= total_chunks
    size_mb = round(os.path.getsize(dest_path) / (1024 * 1024), 2)
    rel_path = os.path.relpath(dest_path, "/runpod-volume").replace("\\", "/")

    status = "complete" if is_complete else "partial"
    logger.info("Upload status: %s (%.2f MB written so far)", status, size_mb)

    return {
        "status": status,
        "path": rel_path,
        "size_mb": size_mb,
        "base_model": base_model,
    }


# ---------------------------------------------------------------------------
# delete_model
# ---------------------------------------------------------------------------

def handle_delete_model(job_input: dict[str, Any]) -> dict[str, Any]:
    """
    Delete a model file or directory from the network volume.

    Request fields
    --------------
    path : str
        Relative path under ``/runpod-volume/`` (e.g.
        ``"models/loras/my-lora.safetensors"``).  Alternatively, provide
        ``name`` + ``model_type`` to construct the path automatically.
    name : str, optional
        Filename / directory name.
    model_type : str, optional
        One of ``"checkpoint"``, ``"lora"``, ``"controlnet"``.

    Returns
    -------
    dict
        ``{"status": "deleted", "path": ...}``
    """
    rel_path: str = job_input.get("path", "")
    name: str = job_input.get("name", "")
    model_type: str = job_input.get("model_type", "")

    if rel_path:
        # Sanitise: prevent path traversal outside the volume
        abs_path = os.path.normpath(os.path.join("/runpod-volume", rel_path))
        if not abs_path.startswith("/runpod-volume"):
            raise ValueError(f"Path '{rel_path}' is outside the allowed volume root")
    elif name and model_type:
        directory = _resolve_dir(model_type)
        abs_path = os.path.join(directory, name)
    else:
        raise ValueError("Provide either 'path' or both 'name' and 'model_type'")

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Model not found: '{abs_path}'")

    logger.info("Deleting '%s'", abs_path)

    if os.path.isfile(abs_path):
        os.remove(abs_path)
    else:
        import shutil
        shutil.rmtree(abs_path)

    logger.info("Deleted '%s'", abs_path)
    return {
        "status": "deleted",
        "path": os.path.relpath(abs_path, "/runpod-volume").replace("\\", "/"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dir_size(path: str) -> int:
    """Return total size in bytes of all files under a directory."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for fname in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fname))
            except OSError:
                pass
    return total
