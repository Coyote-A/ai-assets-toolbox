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
from src.services.civitai import CivitAIModelInfo, fetch_model_info, parse_civitai_input
from src.services.model_metadata import ModelInfo, ModelMetadataManager
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


def _mark_complete(volume_path: str, model_key: str, repo_id: str) -> None:
    """Record *model_key* as fully downloaded in ``.manifest.json``."""
    manifest = read_manifest(volume_path)
    manifest[model_key] = {
        "completed": True,
        "repo_id": repo_id,
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
        # Full repo snapshot.
        # local_dir_use_symlinks=False ensures all files are written directly
        # to target_dir rather than being symlinked from a hidden .cache tree.
        logger.info("Snapshot-downloading %s → %s", entry.repo_id, target_dir)
        snapshot_download(
            repo_id=entry.repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
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
            local_dir_use_symlinks=False,
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

        try:
            entry = get_model(key)
        except KeyError:
            logger.error("Unknown model key: %r", key)
            return {"status": "error", "error": f"Unknown model key: {key!r}"}

        manifest = read_manifest(MODELS_MOUNT_PATH)
        if key in manifest and manifest[key].get("completed", False):
            # Check if the repo_id in the manifest matches the registry.
            # If they differ the cached weights are stale (e.g. model was
            # switched from Qwen2.5-VL to Qwen3-VL) and must be replaced.
            manifest_repo_id = manifest[key].get("repo_id")
            if manifest_repo_id == entry.repo_id:
                logger.info("Model %r already downloaded — skipping.", key)
                return {"status": "already_complete"}
            else:
                logger.warning(
                    "Model %r repo_id mismatch: manifest has %r, registry has %r — "
                    "deleting stale weights and re-downloading.",
                    key,
                    manifest_repo_id,
                    entry.repo_id,
                )
                # Delete old model directory
                old_model_dir = os.path.join(MODELS_MOUNT_PATH, entry.local_dir)
                if os.path.exists(old_model_dir):
                    shutil.rmtree(old_model_dir)
                    logger.info("Deleted stale model directory: %s", old_model_dir)
                # Remove stale manifest entry
                del manifest[key]
                write_manifest(manifest, MODELS_MOUNT_PATH)
                # Commit the deletion so other containers see the clean state,
                # then reload to confirm the volume is consistent before we
                # start writing new files.
                models_volume.commit()
                models_volume.reload()

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
        _mark_complete(MODELS_MOUNT_PATH, key, entry.repo_id)
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
    # Volume cleanup
    # ------------------------------------------------------------------

    @modal.method()
    def reset_model(self, key: str) -> dict:
        """Delete a model from the volume and remove it from the manifest.
        Used when a model needs to be re-downloaded (e.g., wrong version).
        """
        entry = get_model(key)
        model_dir = os.path.join(MODELS_MOUNT_PATH, entry.local_dir)

        # Delete the model directory
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            logger.info("Deleted model directory: %s", model_dir)

        # Remove from manifest
        manifest = read_manifest(MODELS_MOUNT_PATH)
        if key in manifest:
            del manifest[key]
            write_manifest(manifest, MODELS_MOUNT_PATH)
            logger.info("Removed '%s' from manifest", key)

        # Remove from progress
        progress_path = os.path.join(MODELS_MOUNT_PATH, PROGRESS_FILE)
        if os.path.exists(progress_path):
            with open(progress_path) as f:
                progress = json.load(f)
            if key in progress:
                del progress[key]
                with open(progress_path, "w") as f:
                    json.dump(progress, f, indent=2)

        from src.app_config import models_volume
        models_volume.commit()

        return self.check_status()

    # ------------------------------------------------------------------
    # LoRA download (CivitAI)
    # ------------------------------------------------------------------

    @modal.method()
    def download_lora(self, civitai_model_id: int, civitai_token: str) -> dict:
        """
        Download a LoRA from CivitAI and save it to the LoRA volume.

        The file is saved as ``{LORAS_MOUNT_PATH}/lora_{civitai_model_id}.safetensors``.

        Returns ``{"status": "completed"|"error", ...}``.

        .. deprecated::
            Use :meth:`download_civitai_model` instead, which supports full
            metadata integration, checkpoints, and flexible URL/ID input.
        """
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
            import requests  # lazy import — only available in download_image
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

    # ------------------------------------------------------------------
    # CivitAI model download (Phase 2)
    # ------------------------------------------------------------------

    @modal.method()
    def download_civitai_model(self, user_input: str, civitai_token: str | None = None) -> dict:
        """
        Download a model from CivitAI by URL, model ID, or version ID.

        Steps:

        1. Parse user input to get model_id/version_id
        2. Fetch metadata from CivitAI API
        3. Download the file to the appropriate subdirectory
        4. Save metadata to .metadata.json
        5. Return the model info

        Returns: ``{"success": bool, "model_info": dict | None, "error": str | None}``
        """
        from src.app_config import lora_volume  # noqa: F811 — re-import for clarity inside method

        # Step 1: Parse input
        model_id, version_id = parse_civitai_input(user_input)
        if model_id is None and version_id is None:
            return {"success": False, "model_info": None, "error": f"Could not parse input: {user_input}"}

        # Step 2: Fetch metadata
        logger.info("Fetching CivitAI metadata for model=%s version=%s", model_id, version_id)
        civitai_info = fetch_model_info(model_id, version_id, civitai_token)
        if civitai_info is None:
            return {"success": False, "model_info": None, "error": "Failed to fetch model info from CivitAI"}

        # Step 3: Determine target path
        # LoRAs go to /vol/loras/loras/, checkpoints to /vol/loras/checkpoints/
        if civitai_info.model_type == "checkpoint":
            subdir = os.path.join(LORAS_MOUNT_PATH, "checkpoints")
        else:
            subdir = os.path.join(LORAS_MOUNT_PATH, "loras")
        os.makedirs(subdir, exist_ok=True)

        target_path = os.path.join(subdir, civitai_info.filename)

        # Check if already exists
        if os.path.exists(target_path):
            logger.info("File already exists: %s", target_path)
            # Still update metadata in case it was missing
            self._save_civitai_metadata(civitai_info)
            lora_volume.commit()
            return {
                "success": True,
                "model_info": self._civitai_to_dict(civitai_info),
                "error": None,
                "already_existed": True,
            }

        # Step 4: Download
        logger.info("Downloading %s (%s) to %s", civitai_info.name, civitai_info.filename, target_path)
        partial_path = target_path + ".partial"

        try:
            import requests  # lazy import — only available in download_image
            headers: dict[str, str] = {}
            if civitai_token:
                headers["Authorization"] = f"Bearer {civitai_token}"

            resp = requests.get(civitai_info.download_url, headers=headers, stream=True, timeout=600)
            resp.raise_for_status()

            with open(partial_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)

            # Atomic rename
            os.replace(partial_path, target_path)
            logger.info("Download complete: %s", target_path)

        except Exception as e:  # noqa: BLE001
            # Cleanup partial
            if os.path.exists(partial_path):
                os.remove(partial_path)
            logger.error("Download failed: %s", e)
            return {"success": False, "model_info": None, "error": str(e)}

        # Step 5: Save metadata
        self._save_civitai_metadata(civitai_info)

        lora_volume.commit()

        return {
            "success": True,
            "model_info": self._civitai_to_dict(civitai_info),
            "error": None,
            "already_existed": False,
        }

    def _save_civitai_metadata(self, civitai_info: CivitAIModelInfo) -> None:
        """Save CivitAI model info to the metadata manager."""
        mgr = ModelMetadataManager(LORAS_MOUNT_PATH)
        model_info = ModelInfo(
            filename=civitai_info.filename,
            model_type=civitai_info.model_type,
            name=civitai_info.name,
            civitai_model_id=civitai_info.model_id,
            civitai_version_id=civitai_info.version_id,
            trigger_words=civitai_info.trigger_words,
            description=civitai_info.description,
            base_model=civitai_info.base_model,
            default_weight=civitai_info.recommended_weight or 1.0,
            tags=civitai_info.tags,
            source_url=f"https://civitai.com/models/{civitai_info.model_id}",
            size_bytes=civitai_info.size_bytes,
        )
        mgr.add_model(model_info)

    @staticmethod
    def _civitai_to_dict(info: CivitAIModelInfo) -> dict:
        """Convert :class:`~src.services.civitai.CivitAIModelInfo` to a serializable dict."""
        return {
            "model_id": info.model_id,
            "version_id": info.version_id,
            "name": info.name,
            "version_name": info.version_name,
            "model_type": info.model_type,
            "base_model": info.base_model,
            "trigger_words": info.trigger_words,
            "filename": info.filename,
            "size_bytes": info.size_bytes,
            "recommended_weight": info.recommended_weight,
        }

    @modal.method()
    def fetch_civitai_info(self, user_input: str, civitai_token: str | None = None) -> dict:
        """
        Fetch CivitAI model info without downloading.

        Useful for previewing model details in the UI before committing to a
        download.

        Returns: ``{"success": bool, "model_info": dict | None, "error": str | None}``
        """
        model_id, version_id = parse_civitai_input(user_input)
        if model_id is None and version_id is None:
            return {"success": False, "model_info": None, "error": f"Could not parse: {user_input}"}

        info = fetch_model_info(model_id, version_id, civitai_token)
        if info is None:
            return {"success": False, "model_info": None, "error": "Failed to fetch from CivitAI API"}

        return {"success": True, "model_info": self._civitai_to_dict(info), "error": None}

    @modal.method()
    def delete_user_model(self, filename: str, model_type: str = "lora") -> dict:
        """
        Delete a user model (LoRA or checkpoint) from the volume.

        Returns: ``{"success": bool, "error": str | None}``
        """
        from src.app_config import lora_volume  # noqa: F811

        if model_type == "checkpoint":
            file_path = os.path.join(LORAS_MOUNT_PATH, "checkpoints", filename)
        else:
            file_path = os.path.join(LORAS_MOUNT_PATH, "loras", filename)

        # Also check flat path (legacy)
        flat_path = os.path.join(LORAS_MOUNT_PATH, filename)

        deleted = False
        for path in [file_path, flat_path]:
            if os.path.exists(path):
                os.remove(path)
                logger.info("Deleted model file: %s", path)
                deleted = True
                break

        # Remove metadata
        mgr = ModelMetadataManager(LORAS_MOUNT_PATH)
        mgr.remove_model(filename)

        lora_volume.commit()

        if not deleted:
            return {"success": False, "error": f"File not found: {filename}"}
        return {"success": True, "error": None}

    @modal.method()
    def list_user_models(self, model_type: str | None = None) -> list[dict]:
        """
        List all user models with metadata.

        Scans both the metadata store and the filesystem so that files without
        metadata entries (orphans) are also returned.

        Returns a list of model info dicts.
        """
        from src.app_config import lora_volume  # noqa: F811
        lora_volume.reload()

        mgr = ModelMetadataManager(LORAS_MOUNT_PATH)
        models = mgr.list_models(model_type)

        # Also scan for files without metadata
        result: list[dict] = []
        known_files = {m.filename for m in models}

        # Add models with metadata
        for m in models:
            d = {
                "filename": m.filename,
                "name": m.name,
                "model_type": m.model_type,
                "trigger_words": m.trigger_words,
                "base_model": m.base_model,
                "default_weight": m.default_weight,
                "civitai_model_id": m.civitai_model_id,
                "tags": m.tags,
                "size_bytes": m.size_bytes,
            }
            result.append(d)

        # Scan for orphan files (no metadata)
        for subdir_name, mtype in [("loras", "lora"), ("checkpoints", "checkpoint"), ("", "lora")]:
            scan_dir = os.path.join(LORAS_MOUNT_PATH, subdir_name) if subdir_name else LORAS_MOUNT_PATH
            if not os.path.isdir(scan_dir):
                continue
            for fname in os.listdir(scan_dir):
                if fname.startswith(".") or fname in known_files:
                    continue
                if not fname.endswith((".safetensors", ".ckpt", ".pt")):
                    continue
                if model_type and mtype != model_type:
                    continue
                # Orphan file — add with minimal metadata
                fpath = os.path.join(scan_dir, fname)
                result.append({
                    "filename": fname,
                    "name": fname.rsplit(".", 1)[0],
                    "model_type": mtype,
                    "trigger_words": [],
                    "base_model": "",
                    "default_weight": 1.0,
                    "civitai_model_id": None,
                    "tags": [],
                    "size_bytes": os.path.getsize(fpath) if os.path.isfile(fpath) else 0,
                })

        return result

    @modal.method()
    def migrate_metadata(self) -> dict:
        """
        Run metadata migration from hardcoded LoRAs.

        Also moves flat LoRA files (stored directly under ``LORAS_MOUNT_PATH``)
        into the ``loras/`` subdirectory for the new layout.

        Returns: ``{"migrated": int, "moved": int}``
        """
        from src.app_config import lora_volume  # noqa: F811
        lora_volume.reload()

        mgr = ModelMetadataManager(LORAS_MOUNT_PATH)

        # Migrate hardcoded LoRAs (from upscale.py's HARDCODED_LORAS format)
        hardcoded = {
            "Aesthetic Quality": {
                "filename": "lora_929497.safetensors",
                "civitai_id": 929497,
                "trigger": "",
                "weight": 1.0,
                "base_model": "Illustrious",
            },
            "Detailer IL": {
                "filename": "lora_1231943.safetensors",
                "civitai_id": 1231943,
                "trigger": "",
                "weight": 1.0,
                "base_model": "Illustrious",
            },
        }
        migrated = mgr.migrate_from_hardcoded(hardcoded)

        # Move flat files into loras/ subdirectory
        loras_subdir = os.path.join(LORAS_MOUNT_PATH, "loras")
        os.makedirs(loras_subdir, exist_ok=True)

        moved = 0
        for fname in os.listdir(LORAS_MOUNT_PATH):
            if fname.startswith(".") or not fname.endswith((".safetensors", ".ckpt", ".pt")):
                continue
            src_path = os.path.join(LORAS_MOUNT_PATH, fname)
            if not os.path.isfile(src_path):
                continue
            dst_path = os.path.join(loras_subdir, fname)
            if not os.path.exists(dst_path):
                shutil.move(src_path, dst_path)
                logger.info("Moved %s -> loras/%s", fname, fname)
                moved += 1

        lora_volume.commit()
        return {"migrated": migrated, "moved": moved}
