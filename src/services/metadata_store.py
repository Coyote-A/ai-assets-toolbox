"""
Centralized metadata storage using Modal Dict.

This module provides a Dict-based storage solution for model metadata,
replacing the previous JSON file-based approach. Using Modal Dict provides:
- No reload needed: Dict is always current across all containers
- Atomic updates: Built-in concurrency control
- Faster access: In-memory reads
- Simpler code: No file I/O handling
"""
from __future__ import annotations

import modal
from datetime import datetime, timezone
from typing import Optional

# Get the shared Dict
metadata_dict = modal.Dict.from_name("ai-toolbox-model-metadata", create_if_missing=True)

# Keys within the Dict
MANIFEST_KEY = "manifest"
PROGRESS_KEY = "progress"


class MetadataStore:
    """Manages model metadata in Modal Dict."""
    
    def __init__(self):
        self._dict = metadata_dict
    
    # Manifest operations
    def get_manifest(self) -> dict:
        """Get the full manifest dict."""
        return self._dict.get(MANIFEST_KEY, {})
    
    def set_manifest(self, manifest: dict) -> None:
        """Set the full manifest dict."""
        self._dict[MANIFEST_KEY] = manifest
    
    def get_model_entry(self, model_key: str) -> dict | None:
        """Get a single model's entry from the manifest."""
        manifest = self.get_manifest()
        return manifest.get(model_key)
    
    def set_model_entry(self, model_key: str, entry: dict) -> None:
        """Set a single model's entry in the manifest."""
        manifest = self.get_manifest()
        manifest[model_key] = entry
        self.set_manifest(manifest)
    
    def mark_complete(self, model_key: str, repo_id: str, size_bytes: int = 0) -> None:
        """Mark a model as completely downloaded."""
        self.set_model_entry(model_key, {
            "completed": True,
            "repo_id": repo_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "size_bytes": size_bytes,
        })
    
    def remove_model_entry(self, model_key: str) -> None:
        """Remove a model's entry from the manifest."""
        manifest = self.get_manifest()
        manifest.pop(model_key, None)
        self.set_manifest(manifest)
    
    # Progress operations
    def get_progress(self) -> dict:
        """Get the full progress dict."""
        return self._dict.get(PROGRESS_KEY, {})
    
    def set_progress(self, progress: dict) -> None:
        """Set the full progress dict."""
        self._dict[PROGRESS_KEY] = progress
    
    def get_model_progress(self, model_key: str) -> dict | None:
        """Get a single model's progress entry."""
        progress = self.get_progress()
        return progress.get(model_key)
    
    def set_model_progress(
        self, 
        model_key: str, 
        status: str, 
        percentage: float | None = None,
        error: str | None = None,
    ) -> None:
        """Set a single model's progress entry."""
        progress = self.get_progress()
        progress[model_key] = {
            "status": status,
            "percentage": percentage,
            "error": error,
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self.set_progress(progress)
    
    def clear_progress(self, model_key: str) -> None:
        """Clear a model's progress entry."""
        progress = self.get_progress()
        progress.pop(model_key, None)
        self.set_progress(progress)
    
    # Validation
    def is_model_downloaded(self, model_key: str, expected_repo_id: str) -> bool:
        """Check if model is marked complete with matching repo_id."""
        entry = self.get_model_entry(model_key)
        if not entry:
            return False
        if not entry.get("completed", False):
            return False
        if entry.get("repo_id") != expected_repo_id:
            return False
        return True
