"""
Download service for the AI Assets Toolbox setup wizard.

A CPU-only Modal class that downloads model weights from HuggingFace (and
LoRAs from CivitAI) to the shared Modal Volumes, tracking progress in
``.progress.json`` and completion in ``.manifest.json``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone

import modal

from src.app_config import app, download_image, lora_volume, models_volume
from src.services.model_registry import (
    ALL_MODELS,
    LORAS_MOUNT_PATH,
    MANIFEST_FILE,
    MODELS_MOUNT_PATH,
    PROGRESS_FILE,
    ModelEntry,
    get_model,
    read_manifest,
    write_manifest,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_progress(
    volume_path: str,
    model_key: str,
    status: str,
    percentage: float | None = None,
    error: str | None = None,
) -> None:
    """Persist a progress entry for *model_key* to ``.progress.json``."""
    progress_path = os.path.join(volume_path, PROGRESS_FILE)
    try:
        if os.path.exists(progress_path):
            with open(progress_path, encoding="utf-8") as fh:
                progress = json.load(fh)
        else:
            progress = {}
    except (OSError, json.JSONDecodeError):
        progress = {}

    entry: dict = {"status": status}
    if percentage is not None:
        entry["percentage"] = round(percentage, 1)
    if error is not None:
        entry["error"] = error

    progress[model_key] = entry

    os.makedirs(volume_path, exist_ok=True)
    with open(progress_path, "w", encoding="utf-8") as fh:
        json.dump(progress, fh, indent=2)


def _mark_complete(volume_path: str, model_key: str) -> None:
    """Record *model_key* as fully downloaded in ``.manifest.json``."""
    manifest = read_manifest(volume_path)
    manifest[model_key] = {
        "completed": True,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    write_manifest(manifest, volume_path)


def _download_hf_model(entry: ModelEntry, hf_token: str | None) -> None:
    """
    Download a single HuggingFace model described by *entry*.

    - If ``entry.filename`` is set: use :func:`hf_hub_download` to fetch that
      single file (optionally from ``entry.subfolder``).
    - If ``entry.filename`` is ``None`` and ``entry.subfolder`` is ``None``:
      use :func:`snapshot_download` to mirror the full repo.
    - If ``entry.filename`` is ``None`` and ``entry.subfolder`` is set: use
      :func:`snapshot_download` with ``allow_patterns`` restricted to the
      subfolder, then move the files up to *target_dir*.
    """
    from huggingface_hub import hf_hub_download, snapshot_download  # type: ignore[import]

    target_dir = os.path.join(MODELS_MOUNT_PATH, entry.local_dir)
    os.makedirs(target_dir, exist_ok=True)

    if entry.filename is not None:
        # Single-file download
        logger.info(
            "Downloading %s/%s → %s",
            entry.repo_id,
            entry.filename,
            target_dir,
        )
        hf_hub_download(
            repo_id=entry.repo_id,
            filename=entry.filename,
            subfolder=entry.subfolder,
            local_dir=target_dir,
            token=hf_token,
        )
    elif entry.subfolder is None:
        # Full repo snapshot
        logger.info("Snapshot-downloading %s → %s", entry.repo_id, target_dir)
        snapshot_download(
            repo_id=entry.repo_id,
            local_dir=target_dir,
            token=hf_token,
        )
    else:
        # Subfolder-only snapshot: download then flatten
        logger.info(
            "Snapshot-downloading %s/%s → %s",
            entry.repo_id,
            entry.subfolder,
            target_dir,
        )
        snapshot_download(
            repo_id=entry.repo_id,
            local_dir=target_dir,
            allow_patterns=[f"{entry.subfolder}/**"],
            token=hf_token,
        )
        # Move files from the subfolder up to target_dir
        subfolder_path = os.path.join(target_dir, entry.subfolder)
        if os.path.isdir(subfolder_path):
            for item in os.listdir(subfolder_path):
                src = os.path.join(subfolder_path, item)
                dst = os.path.join(target_dir, item)
                shutil.move(src, dst)
            # Remove the now-empty subfolder tree
            shutil.rmtree(os.path.join(target_dir, entry.subfolder.split("/")[0]), ignore_errors=True)


# ---------------------------------------------------------------------------
# Modal class
# ---------------------------------------------------------------------------

@app.cls(
    image=download_image,
    volumes={
        MODELS_MOUNT_PATH: models_volume,
        LORAS_MOUNT_PATH: lora_volume,
    },
    timeout=3600,
    scaledown_window=120,
)
class DownloadService:
    """CPU-only Modal service that downloads models to the shared volumes."""

    # ------------------------------------------------------------------
    # Status / progress
    # ------------------------------------------------------------------

    @modal.method()
    def check_status(self) -> dict:
        """
        Return download status for every model in the registry.

        Returns a dict keyed by model key::

            {
                "illustrious-xl": {
                    "downloaded": True,
                    "size_bytes": 6938000000,
                    "description": "Illustrious-XL v0.1 (base SDXL model)",
                },
                ...
            }
        """
        manifest = read_manifest(MODELS_MOUNT_PATH)
        result: dict = {}
        for entry in ALL_MODELS:
            result[entry.key] = {
                "downloaded": (
                    entry.key in manifest
                    and manifest[entry.key].get("completed", False)
                ),
                "size_bytes": entry.size_bytes,
                "description": entry.description,
            }
        return result

    @modal.method()
    def get_progress(self) -> dict:
        """
        Return the current contents of ``.progress.json``.

        Returns an empty dict if no download has been started yet.
        """
        progress_path = os.path.join(MODELS_MOUNT_PATH, PROGRESS_FILE)
        if os.path.exists(progress_path):
            with open(progress_path, encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    # ------------------------------------------------------------------
    # Single-model download
    # ------------------------------------------------------------------

    @modal.method()
    def download_model(self, key: str, hf_token: str | None = None) -> dict:
        """
        Download the model identified by *key* from HuggingFace.

        Progress is written to ``.progress.json`` (status: ``"downloading"``
        while in flight, ``"completed"`` on success, ``"error"`` on failure).
        On success the manifest is updated and the volume is committed.

        Returns a dict with ``{"status": "completed"|"already_complete"|"error"}``.
        """
        # Reload volume to see latest state from other containers
        models_volume.reload()

        manifest = read_manifest(MODELS_MOUNT_PATH)
        if key in manifest and manifest[key].get("completed", False):
            logger.info("Model %r already downloaded — skipping.", key)
            return {"status": "already_complete"}

        try:
            entry = get_model(key)
        except KeyError:
            logger.error("Unknown model key: %r", key)
            return {"status": "error", "error": f"Unknown model key: {key!r}"}

        logger.info("Starting download for model %r (%s)", key, entry.description)
        _write_progress(MODELS_MOUNT_PATH, key, "downloading", percentage=0.0)
        models_volume.commit()

        try:
            _download_hf_model(entry, hf_token)
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            logger.exception("Download failed for model %r: %s", key, error_msg)
            _write_progress(MODELS_MOUNT_PATH, key, "error", error=error_msg)
            models_volume.commit()
            return {"status": "error", "error": error_msg}

        # Mark complete
        _write_progress(MODELS_MOUNT_PATH, key, "completed", percentage=100.0)
        _mark_complete(MODELS_MOUNT_PATH, key)
        models_volume.commit()
        logger.info("Model %r downloaded successfully.", key)
        return {"status": "completed"}

    # ------------------------------------------------------------------
    # Bulk download
    # ------------------------------------------------------------------

    @modal.method()
    def download_all(self, hf_token: str | None = None) -> dict:
        """
        Download every model that is not yet marked as complete.

        Models are downloaded sequentially.  Returns the final status dict
        (same shape as :meth:`check_status`).
        """
        manifest = read_manifest(MODELS_MOUNT_PATH)
        for entry in ALL_MODELS:
            if not (
                entry.key in manifest
                and manifest[entry.key].get("completed", False)
            ):
                self.download_model.local(entry.key, hf_token)
        return self.check_status.local()

    # ------------------------------------------------------------------
    # LoRA download (CivitAI)
    # ------------------------------------------------------------------

    @modal.method()
    def download_lora(self, civitai_model_id: int, civitai_token: str) -> dict:
        """
        Download a LoRA from CivitAI and save it to the LoRA volume.

        The file is saved as ``{LORAS_MOUNT_PATH}/lora_{civitai_model_id}.safetensors``.

        Returns ``{"status": "completed"|"error", ...}``.
        """
        import requests  # type: ignore[import]

        lora_volume.reload()

        dest_filename = f"lora_{civitai_model_id}.safetensors"
        dest_path = os.path.join(LORAS_MOUNT_PATH, dest_filename)

        if os.path.exists(dest_path):
            logger.info("LoRA %d already present at %s — skipping.", civitai_model_id, dest_path)
            return {"status": "already_complete", "path": dest_path}

        url = f"https://civitai.com/api/v1/model-versions/{civitai_model_id}/download"
        headers = {"Authorization": f"Bearer {civitai_token}"}

        logger.info("Downloading LoRA %d from CivitAI → %s", civitai_model_id, dest_path)
        partial_path = dest_path + ".partial"

        try:
            os.makedirs(LORAS_MOUNT_PATH, exist_ok=True)
            with requests.get(url, headers=headers, stream=True, timeout=600) as resp:
                resp.raise_for_status()
                with open(partial_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        fh.write(chunk)
            # Atomic rename
            os.rename(partial_path, dest_path)
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            logger.exception("LoRA download failed for model_id=%d: %s", civitai_model_id, error_msg)
            # Clean up partial file if present
            if os.path.exists(partial_path):
                os.remove(partial_path)
            return {"status": "error", "error": error_msg}

        lora_volume.commit()
        logger.info("LoRA %d downloaded successfully to %s.", civitai_model_id, dest_path)
        return {"status": "completed", "path": dest_path}
