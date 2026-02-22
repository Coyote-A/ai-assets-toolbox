"""
Centralized model registry for the AI Assets Toolbox.

Defines all models required by the app, their HuggingFace sources, local
storage paths under the models volume, and helper utilities for manifest
tracking (which models have been downloaded to the volume).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Volume / mount constants
# ---------------------------------------------------------------------------

MODELS_VOLUME_NAME = "ai-toolbox-models"
MODELS_MOUNT_PATH = "/vol/models"

LORAS_VOLUME_NAME = "ai-toolbox-loras"
LORAS_MOUNT_PATH = "/vol/loras"

# ---------------------------------------------------------------------------
# Manifest / progress file names (stored at the root of the models volume)
# ---------------------------------------------------------------------------

MANIFEST_FILE = ".manifest.json"   # tracks which models are downloaded
PROGRESS_FILE = ".progress.json"   # tracks download progress


# ---------------------------------------------------------------------------
# ModelEntry dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    """Describes a single model that must be present on the models volume."""

    key: str
    """Unique identifier used throughout the codebase (e.g. "illustrious-xl")."""

    repo_id: str
    """HuggingFace repository id (e.g. "OnomaAIResearch/Illustrious-xl-early-release-v0")."""

    filename: str | None
    """
    Single file to download from the repo, or ``None`` to download the full
    repo (or a subfolder when *subfolder* is also set).
    """

    subfolder: str | None
    """
    Subfolder inside the HuggingFace repo to download, or ``None`` for the
    repo root.
    """

    local_dir: str
    """
    Relative path under the volume mount point where the model is stored
    (e.g. "illustrious-xl").  The absolute path is
    ``{MODELS_MOUNT_PATH}/{local_dir}``.
    """

    size_bytes: int
    """Approximate download size in bytes (used for progress estimation)."""

    description: str
    """Human-readable description shown in the setup wizard UI."""

    service: str
    """Which GPU service uses this model: ``"upscale"`` or ``"caption"``."""


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

ALL_MODELS: list[ModelEntry] = [
    ModelEntry(
        key="illustrious-xl",
        repo_id="OnomaAIResearch/Illustrious-xl-early-release-v0",
        filename="Illustrious-XL-v0.1.safetensors",
        subfolder=None,
        local_dir="illustrious-xl",
        size_bytes=6_938_000_000,
        description="Illustrious-XL v0.1 (base SDXL model)",
        service="upscale",
    ),
    ModelEntry(
        key="controlnet-tile",
        repo_id="xinsir/controlnet-tile-sdxl-1.0",
        filename=None,  # full repo
        subfolder=None,
        local_dir="controlnet-tile",
        size_bytes=2_502_000_000,
        description="ControlNet Tile SDXL 1.0",
        service="upscale",
    ),
    ModelEntry(
        key="sdxl-vae",
        repo_id="madebyollin/sdxl-vae-fp16-fix",
        filename=None,  # full repo
        subfolder=None,
        local_dir="sdxl-vae",
        size_bytes=335_000_000,
        description="SDXL VAE fp16-fix",
        service="upscale",
    ),
    ModelEntry(
        key="ip-adapter",
        repo_id="h94/IP-Adapter",
        filename="ip-adapter-plus_sdxl_vit-h.safetensors",
        subfolder="sdxl_models",
        local_dir="ip-adapter",
        size_bytes=100_000_000,
        description="IP-Adapter Plus SDXL ViT-H",
        service="upscale",
    ),
    ModelEntry(
        key="clip-vit-h",
        repo_id="h94/IP-Adapter",
        filename=None,  # full subfolder
        subfolder="models/image_encoder",
        local_dir="clip-vit-h",
        size_bytes=3_500_000_000,
        description="CLIP ViT-H (for IP-Adapter)",
        service="upscale",
    ),
    ModelEntry(
        key="qwen3-vl-2b",
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        filename=None,  # full repo
        subfolder=None,
        local_dir="qwen3-vl-2b",
        size_bytes=6_000_000_000,
        description="Qwen2.5-VL-3B-Instruct (captioning)",
        service="caption",
    ),
]

# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_models_for_service(service: str) -> list[ModelEntry]:
    """Return all models whose *service* field matches *service*."""
    return [m for m in ALL_MODELS if m.service == service]


def get_model(key: str) -> ModelEntry:
    """Return the :class:`ModelEntry` for *key*, raising :exc:`KeyError` if not found."""
    for m in ALL_MODELS:
        if m.key == key:
            return m
    raise KeyError(f"Unknown model key: {key!r}")


def get_model_path(key: str) -> str:
    """Return the absolute path to *key*'s directory on the models volume."""
    entry = get_model(key)
    return f"{MODELS_MOUNT_PATH}/{entry.local_dir}"


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def read_manifest(volume_path: str = MODELS_MOUNT_PATH) -> dict:
    """
    Read the download manifest from *volume_path*.

    Returns an empty dict if the manifest file does not exist yet.
    """
    manifest_path = os.path.join(volume_path, MANIFEST_FILE)
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


def write_manifest(manifest: dict, volume_path: str = MODELS_MOUNT_PATH) -> None:
    """Write *manifest* as JSON to *volume_path*."""
    manifest_path = os.path.join(volume_path, MANIFEST_FILE)
    os.makedirs(volume_path, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def is_model_downloaded(key: str, volume_path: str = MODELS_MOUNT_PATH) -> bool:
    """Return ``True`` if *key* is recorded as downloaded in the manifest
    **and** the manifest's ``repo_id`` matches the current registry entry.

    If the ``repo_id`` stored in the manifest differs from the registry (e.g.
    the model was switched from Qwen2.5-VL to Qwen3-VL), the cached weights
    are considered stale and the function returns ``False``.
    """
    manifest = read_manifest(volume_path)
    entry_data = manifest.get(key, {})
    if not entry_data.get("completed", False) and not entry_data.get("downloaded", False):
        return False
    # Verify repo_id matches the current registry entry.
    # If the manifest entry has no repo_id (written before repo_id tracking was
    # added) we treat it as stale so the weights are re-downloaded with the
    # correct repo.
    try:
        registry_entry = get_model(key)
    except KeyError:
        return False
    manifest_repo_id = entry_data.get("repo_id")
    if manifest_repo_id != registry_entry.repo_id:
        return False
    return True


def all_models_ready(volume_path: str = MODELS_MOUNT_PATH) -> bool:
    """Return ``True`` if every model in :data:`ALL_MODELS` is downloaded."""
    return all(is_model_downloaded(m.key, volume_path) for m in ALL_MODELS)
