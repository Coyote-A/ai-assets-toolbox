"""
Centralized model registry for the AI Assets Toolbox.

Defines all models required by the app, their HuggingFace sources, local
storage paths under the models volume, and helper utilities for manifest
tracking (which models have been downloaded to the volume).

Metadata is now stored in Modal Dict (via MetadataStore) instead of JSON files,
providing faster access and automatic synchronization across containers.
"""

from __future__ import annotations

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
# Manifest / progress file names (legacy - kept for migration reference)
# ---------------------------------------------------------------------------

MANIFEST_FILE = ".manifest.json"   # tracks which models are downloaded (legacy)
PROGRESS_FILE = ".progress.json"   # tracks download progress (legacy)


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
        repo_id="OnomaAIResearch/Illustrious-XL-v2.0",
        filename="Illustrious-XL-v2.0.safetensors",
        subfolder=None,
        local_dir="illustrious-xl",
        size_bytes=6_938_000_000,
        description="Illustrious-XL v2.0 (base SDXL model)",
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
        key="qwen2.5-vl-3b",
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        filename=None,  # full repo
        subfolder=None,
        local_dir="qwen2.5-vl-3b",
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


def check_model_files_exist(key: str) -> bool:
    """Check if the model's files actually exist on the volume.
    
    For single-file models (filename is set), checks if that specific file exists.
    For full-repo or subfolder models, checks if the directory exists and is non-empty.
    
    Returns:
        True if the model files appear to be present, False otherwise.
    """
    entry = get_model(key)
    model_dir = f"{MODELS_MOUNT_PATH}/{entry.local_dir}"
    
    if not os.path.isdir(model_dir):
        return False
    
    if entry.filename is not None:
        # Single-file model: check for the specific file
        # First check the flattened location (expected after download)
        file_path = os.path.join(model_dir, entry.filename)
        if os.path.isfile(file_path):
            return True
        # Also check in the subfolder if one was specified (pre-flattened location)
        if entry.subfolder:
            nested_path = os.path.join(model_dir, entry.subfolder, entry.filename)
            if os.path.isfile(nested_path):
                return True
        return False
    else:
        # Full repo or subfolder: check directory is non-empty
        return len(os.listdir(model_dir)) > 0


def get_model_file_path(key: str) -> str | None:
    """Return the absolute path to a single-file model's file, or None if not found.
    
    For models with a filename, searches both the flattened location and the
    original subfolder location. Returns None for full-repo models or if the
    file is not found.
    """
    entry = get_model(key)
    if entry.filename is None:
        return None
    
    model_dir = f"{MODELS_MOUNT_PATH}/{entry.local_dir}"
    
    # Check flattened location first
    flat_path = os.path.join(model_dir, entry.filename)
    if os.path.isfile(flat_path):
        return flat_path
    
    # Check in subfolder if specified
    if entry.subfolder:
        nested_path = os.path.join(model_dir, entry.subfolder, entry.filename)
        if os.path.isfile(nested_path):
            return nested_path
    
    return None


# ---------------------------------------------------------------------------
# Manifest helpers (using MetadataStore)
# ---------------------------------------------------------------------------

def read_manifest(volume_path: str = MODELS_MOUNT_PATH) -> dict:
    """
    Read the download manifest from Modal Dict.

    The *volume_path* parameter is ignored (kept for backward compatibility).
    Returns an empty dict if no manifest exists yet.
    """
    from src.services.metadata_store import MetadataStore
    return MetadataStore().get_manifest()


def write_manifest(manifest: dict, volume_path: str = MODELS_MOUNT_PATH) -> None:
    """Write *manifest* to Modal Dict.

    The *volume_path* parameter is ignored (kept for backward compatibility).
    """
    from src.services.metadata_store import MetadataStore
    MetadataStore().set_manifest(manifest)


def is_model_downloaded(key: str, volume_path: str = MODELS_MOUNT_PATH) -> bool:
    """Return ``True`` if *key* is recorded as downloaded in the manifest
    **and** the manifest's ``repo_id`` matches the current registry entry
    **and** the files actually exist on disk.

    This performs a two-stage validation:
    1. Check the manifest entry in Modal Dict has correct repo_id
    2. Verify files exist on the volume (defense in depth)

    If the ``repo_id`` stored in the manifest differs from the registry (e.g.
    the model was switched from Qwen2.5-VL to Qwen3-VL), the cached weights
    are considered stale and the function returns ``False``.
    """
    from src.services.metadata_store import MetadataStore
    
    try:
        registry_entry = get_model(key)
    except KeyError:
        return False
    
    # Check manifest entry in Modal Dict
    store = MetadataStore()
    if not store.is_model_downloaded(key, registry_entry.repo_id):
        return False
    
    # Also verify files exist on disk (defense in depth)
    # This handles edge cases like volume corruption or manual file deletion
    return check_model_files_exist(key)


def all_models_ready(volume_path: str = MODELS_MOUNT_PATH) -> bool:
    """Return ``True`` if every model in :data:`ALL_MODELS` is downloaded."""
    return all(is_model_downloaded(m.key, volume_path) for m in ALL_MODELS)
