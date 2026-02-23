# Plan: Migrate Model Metadata from JSON to Modal Dict

## Current State

### Storage Locations
| Data Type | Current Storage | Location |
|-----------|----------------|----------|
| Model Manifest | `.manifest.json` | models_volume root |
| Download Progress | `.progress.json` | models_volume root |
| API Tokens | `.api_tokens.json` | lora_volume root |
| LoRA Metadata | Per-folder JSON | lora_volume subdirs |

### Current Issues
1. **Stale reads**: Warm containers cache volume state, requiring explicit `reload()` calls
2. **Race conditions**: Concurrent reads/writes to JSON files
3. **Slow access**: File I/O overhead for frequent status checks
4. **Inconsistent validation**: Manifest says "downloaded" but files may be missing/wrong

## Proposed Architecture

### New Modal Dicts
```python
# In src/app_config.py
model_metadata = modal.Dict.from_name("ai-toolbox-model-metadata", create_if_missing=True)
```

### Data Structure
```python
model_metadata = {
    "manifest": {
        "illustrious-xl": {
            "completed": True,
            "repo_id": "OnomaAIResearch/Illustrious-XL-v2.0",
            "timestamp": "2024-02-23T10:00:00Z",
            "size_bytes": 6938000000,
        },
        "controlnet-tile": {...},
        ...
    },
    "progress": {
        "illustrious-xl": {
            "status": "downloading",  # pending, downloading, complete, error
            "percentage": 45.2,
            "error": None,
            "started_at": "2024-02-23T10:00:00Z",
        },
        ...
    },
}
```

### Key Benefits
1. **No reload needed**: Dict is always current across all containers
2. **Atomic updates**: Built-in concurrency control
3. **Faster access**: In-memory reads
4. **Simpler code**: No file I/O handling

## Implementation Plan

### Phase 1: Create Metadata Store Module

Create `src/services/metadata_store.py`:

```python
"""Centralized metadata storage using Modal Dict."""
from __future__ import annotations

import modal
from datetime import datetime, timezone
from typing import Optional

# Get the shared Dict
metadata_dict = modal.Dict.from_name("ai-toolbox-model-metadata")

# Keys within the Dict
MANIFEST_KEY = "manifest"
PROGRESS_KEY = "progress"

class MetadataStore:
    """Manages model metadata in Modal Dict."""
    
    def __init__(self):
        self._dict = metadata_dict
    
    # Manifest operations
    def get_manifest(self) -> dict:
        return self._dict.get(MANIFEST_KEY, {})
    
    def set_manifest(self, manifest: dict) -> None:
        self._dict[MANIFEST_KEY] = manifest
    
    def get_model_entry(self, model_key: str) -> dict | None:
        manifest = self.get_manifest()
        return manifest.get(model_key)
    
    def set_model_entry(self, model_key: str, entry: dict) -> None:
        manifest = self.get_manifest()
        manifest[model_key] = entry
        self.set_manifest(manifest)
    
    def mark_complete(self, model_key: str, repo_id: str, size_bytes: int = 0) -> None:
        self.set_model_entry(model_key, {
            "completed": True,
            "repo_id": repo_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "size_bytes": size_bytes,
        })
    
    # Progress operations
    def get_progress(self) -> dict:
        return self._dict.get(PROGRESS_KEY, {})
    
    def set_progress(self, progress: dict) -> None:
        self._dict[PROGRESS_KEY] = progress
    
    def get_model_progress(self, model_key: str) -> dict | None:
        progress = self.get_progress()
        return progress.get(model_key)
    
    def set_model_progress(
        self, 
        model_key: str, 
        status: str, 
        percentage: float | None = None,
        error: str | None = None,
    ) -> None:
        progress = self.get_progress()
        progress[model_key] = {
            "status": status,
            "percentage": percentage,
            "error": error,
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self.set_progress(progress)
    
    def clear_progress(self, model_key: str) -> None:
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
```

### Phase 2: Update Model Registry

Modify `src/services/model_registry.py`:

```python
# Replace file-based functions with Dict-based

def read_manifest(volume_path: str = MODELS_MOUNT_PATH) -> dict:
    """Read manifest from Modal Dict (volume_path ignored for compatibility)."""
    from src.services.metadata_store import MetadataStore
    return MetadataStore().get_manifest()

def write_manifest(manifest: dict, volume_path: str = MODELS_MOUNT_PATH) -> None:
    """Write manifest to Modal Dict."""
    from src.services.metadata_store import MetadataStore
    MetadataStore().set_manifest(manifest)

def is_model_downloaded(key: str, volume_path: str = MODELS_MOUNT_PATH) -> bool:
    """Check if model is downloaded with correct repo_id AND files exist."""
    from src.services.metadata_store import MetadataStore
    try:
        entry = get_model(key)
    except KeyError:
        return False
    
    # Check manifest entry
    store = MetadataStore()
    if not store.is_model_downloaded(key, entry.repo_id):
        return False
    
    # ALSO verify files exist on disk (defense in depth)
    return check_model_files_exist(key)
```

### Phase 3: Update Download Service

Modify `src/services/download.py`:

```python
# Replace _write_progress and _mark_complete

def _write_progress(
    volume_path: str,
    model_key: str,
    status: str,
    percentage: float | None = None,
    error: str | None = None,
) -> None:
    """Update progress in Modal Dict."""
    from src.services.metadata_store import MetadataStore
    MetadataStore().set_model_progress(model_key, status, percentage, error)

def _mark_complete(volume_path: str, model_key: str, repo_id: str) -> None:
    """Mark model as complete in Modal Dict."""
    from src.services.metadata_store import MetadataStore
    MetadataStore().mark_complete(model_key, repo_id)

# In DownloadService class:
@modal.method()
def check_status(self) -> dict:
    """Check download status for all models."""
    from src.services.metadata_store import MetadataStore
    store = MetadataStore()
    manifest = store.get_manifest()
    
    result = {}
    for entry in ALL_MODELS:
        downloaded = store.is_model_downloaded(entry.key, entry.repo_id)
        result[entry.key] = {
            "description": entry.description,
            "service": entry.service,
            "downloaded": downloaded,
            "size_bytes": entry.size_bytes,
        }
    return result

@modal.method()
def get_progress(self) -> dict:
    """Get current download progress."""
    from src.services.metadata_store import MetadataStore
    return MetadataStore().get_progress()
```

### Phase 4: Migration Script

Create one-time migration to move existing JSON data to Dict:

```python
# scripts/migrate_metadata_to_dict.py
"""One-time migration: JSON files -> Modal Dict"""

import modal

app = modal.App("migrate-metadata")
models_volume = modal.Volume.from_name("ai-toolbox-models")
metadata_dict = modal.Dict.from_name("ai-toolbox-model-metadata", create_if_missing=True)

@app.function(volumes={"/vol": models_volume})
def migrate():
    import json
    import os
    
    # Read existing JSON files
    manifest_path = "/vol/.manifest.json"
    progress_path = "/vol/.progress.json"
    
    manifest = {}
    progress = {}
    
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Loaded manifest with {len(manifest)} entries")
    
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        print(f"Loaded progress with {len(progress)} entries")
    
    # Write to Dict
    metadata_dict["manifest"] = manifest
    metadata_dict["progress"] = progress
    
    print("Migration complete!")
    print(f"Manifest: {metadata_dict.get('manifest', {})}")

if __name__ == "__main__":
    migrate()
```

### Phase 5: Update Token Store

The token store already has a Modal Dict defined (`ai-toolbox-tokens`) but still uses JSON files. Update it:

```python
# src/services/token_store.py - simplified version using Dict

from src.app_config import token_store_dict

def get_tokens() -> TokenData:
    data = token_store_dict.get("tokens", {})
    return TokenData(
        hf_token=data.get("hf_token"),
        civitai_token=data.get("civitai_token"),
    )

def save_tokens(hf_token: str | None = None, civitai_token: str | None = None) -> None:
    data = token_store_dict.get("tokens", {})
    if hf_token is not None:
        data["hf_token"] = hf_token
    if civitai_token is not None:
        data["civitai_token"] = civitai_token
    token_store_dict["tokens"] = data
```

## File Existence Validation

Even with Dict storage, we must verify files exist on disk:

```python
def is_model_ready(model_key: str) -> bool:
    """Full validation: manifest entry + file existence."""
    entry = get_model(model_key)
    store = MetadataStore()
    
    # 1. Check manifest has correct entry
    if not store.is_model_downloaded(model_key, entry.repo_id):
        return False
    
    # 2. Check files actually exist on volume
    if not check_model_files_exist(model_key):
        return False
    
    return True
```

This handles edge cases:
- Model deleted from volume but manifest not updated
- Volume corrupted
- Manual file manipulation

## Testing Plan

1. **Unit tests**: Test MetadataStore operations in isolation
2. **Integration test**: Run migration script, verify data transferred
3. **E2E test**: Run setup wizard, verify downloads tracked correctly
4. **Edge case test**: Delete model file, verify status shows not downloaded

## Rollback Plan

If issues arise:
1. Dict data is independent of volume files
2. Can revert code changes
3. JSON files still work as backup (until deleted)

## Commits

1. `feat: add MetadataStore class using Modal Dict`
2. `refactor: update model_registry to use MetadataStore`
3. `refactor: update download service to use MetadataStore`
4. `refactor: update token_store to use Modal Dict`
5. `feat: add migration script for JSON -> Dict`
6. `docs: update AGENTS.md with new architecture`
