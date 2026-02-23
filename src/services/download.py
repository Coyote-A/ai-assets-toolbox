"""
Download service for the AI Assets Toolbox setup wizard.

A CPU-only Modal class that downloads model weights from HuggingFace (and
LoRAs from CivitAI) to the shared Modal Volumes, tracking progress and
completion in Modal Dict via MetadataStore.
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
    is_model_downloaded,
    read_manifest,
    write_manifest,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers (using MetadataStore)
# ---------------------------------------------------------------------------

def _write_progress(
    volume_path: str,
    model_key: str,
    status: str,
    percentage: float | None = None,
    error: str | None = None,
) -> None:
    """Update progress for *model_key* in Modal Dict via MetadataStore.
    
    The *volume_path* parameter is ignored (kept for backward compatibility).
    """
    from src.services.metadata_store import MetadataStore
    MetadataStore().set_model_progress(model_key, status, percentage, error)


def _mark_complete(volume_path: str, model_key: str, repo_id: str) -> None:
    """Record *model_key* as fully downloaded in Modal Dict via MetadataStore.
    
    The *volume_path* parameter is ignored (kept for backward compatibility).
    """
    from src.services.metadata_store import MetadataStore
    MetadataStore().mark_complete(model_key, repo_id)


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
        # hf_hub_download places the file at {local_dir}/{subfolder}/{filename}
        # when subfolder is set.  Move it up to {local_dir}/{filename} so that
        # the rest of the codebase can find it at the expected flat path.
        if entry.subfolder:
            nested_path = os.path.join(target_dir, entry.subfolder, entry.filename)
            flat_path = os.path.join(target_dir, entry.filename)
            if os.path.exists(nested_path) and not os.path.exists(flat_path):
                shutil.move(nested_path, flat_path)
                logger.info(
                    "Moved %s → %s (flattened subfolder)",
                    nested_path,
                    flat_path,
                )
                # Remove the now-empty subfolder tree
                subfolder_root = os.path.join(target_dir, entry.subfolder.split("/")[0])
                shutil.rmtree(subfolder_root, ignore_errors=True)
    elif entry.subfolder is None:
        # Full repo snapshot.
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
        from src.services.metadata_store import MetadataStore
        store = MetadataStore()
        result: dict = {}
        for entry in ALL_MODELS:
            result[entry.key] = {
                "downloaded": is_model_downloaded(entry.key, MODELS_MOUNT_PATH),
                "size_bytes": entry.size_bytes,
                "description": entry.description,
            }
        return result

    @modal.method()
    def get_progress(self) -> dict:
        """
        Return the current download progress from Modal Dict.

        Returns an empty dict if no download has been started yet.
        """
        from src.services.metadata_store import MetadataStore
        return MetadataStore().get_progress()

    # ------------------------------------------------------------------
    # Single-model download
    # ------------------------------------------------------------------

    @modal.method()
    def download_model(self, key: str, hf_token: str | None = None) -> dict:
        """
        Download the model identified by *key* from HuggingFace.

        Progress is tracked in Modal Dict (status: ``"downloading"``
        while in flight, ``"completed"`` on success, ``"error"`` on failure).
        On success the manifest is updated and the volume is committed.

        Returns a dict with ``{"status": "completed"|"already_complete"|"error"}``.
        """
        from src.services.metadata_store import MetadataStore
        store = MetadataStore()
        
        try:
            entry = get_model(key)
        except KeyError:
            logger.error("Unknown model key: %r", key)
            return {"status": "error", "error": f"Unknown model key: {key!r}"}

        # Check if already downloaded using MetadataStore
        if store.is_model_downloaded(key, entry.repo_id):
            logger.info("Model %r already downloaded — skipping.", key)
            return {"status": "already_complete"}
        
        # Check for stale entry with wrong repo_id
        model_entry = store.get_model_entry(key)
        if model_entry and model_entry.get("completed", False):
            manifest_repo_id = model_entry.get("repo_id")
            if manifest_repo_id != entry.repo_id:
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
                store.remove_model_entry(key)
                # Commit the deletion so other containers see the clean state
                models_volume.commit()

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
    def download_all(self, hf_token: str | None = None, civitai_token: str | None = None) -> dict:
        """
        Download every model that is not yet marked as complete.

        Models are downloaded sequentially.  Returns the final status dict
        (same shape as :meth:`check_status`).

        Args:
            hf_token: HuggingFace API token for gated models.
            civitai_token: CivitAI API token for NSFW/early-access models.
        """
        from src.services.metadata_store import MetadataStore
        store = MetadataStore()
        
        for entry in ALL_MODELS:
            if not store.is_model_downloaded(entry.key, entry.repo_id):
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
        from src.services.metadata_store import MetadataStore
        
        entry = get_model(key)
        model_dir = os.path.join(MODELS_MOUNT_PATH, entry.local_dir)

        # Delete the model directory
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            logger.info("Deleted model directory: %s", model_dir)

        # Remove from manifest using MetadataStore
        store = MetadataStore()
        store.remove_model_entry(key)
        logger.info("Removed '%s' from manifest", key)

        # Clear progress using MetadataStore
        store.clear_progress(key)

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

        # CivitAI download endpoint: /api/download/models/{versionId}?token={api_key}
        # The token must be passed as a query parameter, not as a Bearer header.
        url = f"https://civitai.com/api/download/models/{civitai_model_id}"
        params: dict = {}
        if civitai_token:
            params["token"] = civitai_token

        logger.info("Downloading LoRA %d from CivitAI → %s", civitai_model_id, dest_path)
        partial_path = dest_path + ".partial"

        try:
            import requests  # lazy import — only available in download_image
            os.makedirs(LORAS_MOUNT_PATH, exist_ok=True)
            with requests.get(url, params=params, stream=True, timeout=600) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                if "json" in content_type or "html" in content_type:
                    logger.error(
                        "CivitAI returned non-binary response (Content-Type=%r) "
                        "for model_id=%d. Model may require auth or is gated.",
                        content_type, civitai_model_id,
                    )
                    return {
                        "status": "error",
                        "error": (
                            f"CivitAI returned {content_type} instead of a model file. "
                            "Check your API token and model access."
                        ),
                    }
                with open(partial_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        fh.write(chunk)
            actual_size = os.path.getsize(partial_path)
            if actual_size < 1024:
                with open(partial_path, "rb") as fh:
                    sample = fh.read(500)
                logger.error(
                    "Downloaded LoRA file is suspiciously small (%d bytes). Sample: %s",
                    actual_size, sample,
                )
                os.remove(partial_path)
                return {
                    "status": "error",
                    "error": f"Downloaded file is only {actual_size} bytes — likely not a model file.",
                }
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

            # CivitAI download endpoint: /api/download/models/{versionId}?token={api_key}
            # The token must be passed as a query parameter, not as a Bearer header.
            params: dict = {}
            if civitai_token:
                params["token"] = civitai_token

            resp = requests.get(civitai_info.download_url, params=params, stream=True, timeout=600)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "json" in content_type or "html" in content_type:
                logger.error(
                    "CivitAI returned non-binary response (Content-Type=%r). "
                    "The model may require a valid CivitAI API token or may be "
                    "gated (Early Access). Final URL: %s",
                    content_type,
                    resp.url,
                )
                return {
                    "success": False,
                    "model_info": None,
                    "error": (
                        f"CivitAI returned {content_type} instead of a model file. "
                        "The model may require a valid CivitAI API token or may be "
                        "gated (Early Access). Please check your token and model access."
                    ),
                }

            with open(partial_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)

            actual_size = os.path.getsize(partial_path)
            logger.info(
                "Downloaded %s: %d bytes (expected ~%d bytes)",
                civitai_info.filename,
                actual_size,
                civitai_info.size_bytes,
            )
            if actual_size < 1024:
                with open(partial_path, "rb") as f:
                    sample = f.read(500)
                logger.error(
                    "Downloaded file is suspiciously small (%d bytes). "
                    "Content sample: %s",
                    actual_size,
                    sample,
                )
                os.remove(partial_path)
                return {
                    "success": False,
                    "model_info": None,
                    "error": (
                        f"Downloaded file is only {actual_size} bytes — expected a "
                        f"model file of ~{civitai_info.size_bytes} bytes. "
                        "CivitAI may have returned an error page instead of the "
                        "model. Check your API token and model access permissions."
                    ),
                }

            # Atomic rename
            os.replace(partial_path, target_path)
            logger.info("Download complete: %s (%d bytes)", target_path, actual_size)

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
