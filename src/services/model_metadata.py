"""Model metadata management for LoRAs and checkpoints."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

METADATA_FILE = ".metadata.json"


@dataclass
class ModelInfo:
    """Metadata for a single model (LoRA or checkpoint)."""
    # Identity
    filename: str                          # e.g. "lora_929497.safetensors"
    model_type: str                        # "lora" or "checkpoint"
    name: str                              # Display name, e.g. "Aesthetic Quality"

    # CivitAI metadata (optional)
    civitai_model_id: Optional[int] = None
    civitai_version_id: Optional[int] = None

    # Content metadata
    trigger_words: list[str] = field(default_factory=list)
    description: str = ""
    base_model: str = ""                   # e.g. "Illustrious", "SDXL 1.0", "Pony"

    # Recommended settings
    default_weight: float = 1.0
    recommended_weight_min: float = 0.0
    recommended_weight_max: float = 2.0

    # Tags for filtering
    tags: list[str] = field(default_factory=list)

    # Source URL
    source_url: str = ""

    # File info
    size_bytes: int = 0


class ModelMetadataManager:
    """CRUD operations on .metadata.json stored on the LoRA volume."""

    def __init__(self, volume_path: str):
        self._volume_path = volume_path
        self._metadata_path = os.path.join(volume_path, METADATA_FILE)

    def _read(self) -> dict[str, dict]:
        """Read the metadata file. Returns {filename: {metadata_dict}}."""
        if not os.path.exists(self._metadata_path):
            return {}
        try:
            with open(self._metadata_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read metadata: %s", e)
            return {}

    def _write(self, data: dict[str, dict]) -> None:
        """Write the metadata file."""
        os.makedirs(os.path.dirname(self._metadata_path), exist_ok=True)
        with open(self._metadata_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def list_models(self, model_type: str | None = None) -> list[ModelInfo]:
        """List all models, optionally filtered by type."""
        data = self._read()
        result = []
        for filename, info in data.items():
            # Skip internal sentinel keys (e.g. __active_checkpoint__)
            if filename.startswith("__"):
                continue
            if model_type and info.get("model_type") != model_type:
                continue
            result.append(ModelInfo(**{k: v for k, v in info.items() if k in ModelInfo.__dataclass_fields__}))
        return result

    def get_model(self, filename: str) -> ModelInfo | None:
        """Get metadata for a specific model by filename."""
        data = self._read()
        if filename not in data:
            return None
        info = data[filename]
        return ModelInfo(**{k: v for k, v in info.items() if k in ModelInfo.__dataclass_fields__})

    def add_model(self, info: ModelInfo) -> None:
        """Add or update model metadata."""
        data = self._read()
        data[info.filename] = asdict(info)
        self._write(data)

    def remove_model(self, filename: str) -> bool:
        """Remove model metadata. Returns True if found and removed."""
        data = self._read()
        if filename not in data:
            return False
        del data[filename]
        self._write(data)
        return True

    def get_trigger_words(self, filename: str) -> list[str]:
        """Get trigger words for a model."""
        info = self.get_model(filename)
        return info.trigger_words if info else []

    def get_active_checkpoint(self) -> str | None:
        """Get the currently active checkpoint filename."""
        data = self._read()
        return data.get("__active_checkpoint__", None)

    def set_active_checkpoint(self, filename: str | None) -> None:
        """Set the active checkpoint."""
        data = self._read()
        if filename is None:
            data.pop("__active_checkpoint__", None)
        else:
            data["__active_checkpoint__"] = filename
        self._write(data)
