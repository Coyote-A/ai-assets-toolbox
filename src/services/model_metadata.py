"""Model metadata management for LoRAs and checkpoints."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List

logger = logging.getLogger(__name__)

METADATA_FILE = ".metadata.json"


class ModelType(str, Enum):
    """Model type classification for SDXL components."""
    CHECKPOINT = "checkpoint"  # Full base models (SDXL 1.0, DreamShaper, etc.)
    LORA = "lora"              # LoRA adapters
    EMBEDDING = "embedding"    # Textual Inversion embeddings
    VAE = "vae"                # Variational Autoencoders


@dataclass
class ModelInfo:
    """Metadata for a single model (LoRA, checkpoint, embedding, or VAE)."""
    # Identity
    filename: str                          # e.g. "lora_929497.safetensors"
    name: str                              # Display name, e.g. "Aesthetic Quality"

    # Model type (defaults to LORA for backward compatibility)
    model_type: ModelType = ModelType.LORA  # checkpoint, lora, embedding, vae

    # CivitAI metadata (optional)
    civitai_model_id: Optional[int] = None
    civitai_version_id: Optional[int] = None

    # Content metadata
    trigger_words: List[str] = field(default_factory=list)
    description: str = ""
    base_model: str = ""                   # e.g. "Illustrious", "SDXL 1.0", "Pony"

    # Recommended settings (primarily for LoRAs)
    default_weight: float = 1.0
    recommended_weight_min: float = 0.0
    recommended_weight_max: float = 2.0

    # Tags for filtering
    tags: List[str] = field(default_factory=list)

    # Source URL
    source_url: str = ""

    # File info
    size_bytes: int = 0

    # Generation settings
    clip_skip: int = 0                     # 0 = disabled, 1-12 = layers to skip

    # Embedded components (checkpoints may include these)
    embedded_vae: Optional[str] = None     # Filename of embedded VAE
    embedded_loras: List[dict] = field(default_factory=list)  # Embedded LoRAs with weights

    # VAE-specific settings
    is_fp16_fixed: bool = False            # Whether VAE has fp16 fix

    # Preview/thumbnail
    preview_url: str = ""                  # Thumbnail/preview image URL


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

    def _dict_to_model_info(self, data: dict) -> ModelInfo:
        """Convert a dict to ModelInfo, handling ModelType conversion."""
        # Filter to only valid fields
        filtered = {k: v for k, v in data.items() if k in ModelInfo.__dataclass_fields__}
        
        # Convert model_type string to ModelType enum if needed
        if "model_type" in filtered and isinstance(filtered["model_type"], str):
            try:
                filtered["model_type"] = ModelType(filtered["model_type"])
            except ValueError:
                # Unknown type, default to LORA for backward compatibility
                filtered["model_type"] = ModelType.LORA
        elif "model_type" not in filtered:
            # Default to LORA for backward compatibility
            filtered["model_type"] = ModelType.LORA
        
        return ModelInfo(**filtered)

    def list_models(self, model_type: ModelType | str | None = None) -> list[ModelInfo]:
        """List all models, optionally filtered by type.
        
        Args:
            model_type: Filter by model type. Can be ModelType enum or string value.
                       None returns all models.
        
        Returns:
            List of ModelInfo objects matching the filter.
        """
        data = self._read()
        result = []
        
        # Normalize model_type to string for comparison
        filter_type = None
        if model_type is not None:
            filter_type = model_type.value if isinstance(model_type, ModelType) else model_type
        
        for filename, info in data.items():
            # Skip internal sentinel keys (e.g. __active_checkpoint__)
            if filename.startswith("__"):
                continue
            if filter_type and info.get("model_type") != filter_type:
                continue
            result.append(self._dict_to_model_info(info))
        return result

    def get_model(self, filename: str) -> ModelInfo | None:
        """Get metadata for a specific model by filename."""
        data = self._read()
        if filename not in data:
            return None
        return self._dict_to_model_info(data[filename])
    
    def get_models_by_type(self, model_type: ModelType) -> List[ModelInfo]:
        """Get all models of a specific type.
        
        Args:
            model_type: The ModelType to filter by.
            
        Returns:
            List of ModelInfo objects of the specified type.
        """
        return self.list_models(model_type)
    
    def get_checkpoints(self) -> List[ModelInfo]:
        """Get all downloaded checkpoints."""
        return self.get_models_by_type(ModelType.CHECKPOINT)
    
    def get_loras(self) -> List[ModelInfo]:
        """Get all downloaded LoRAs."""
        return self.get_models_by_type(ModelType.LORA)
    
    def get_embeddings(self) -> List[ModelInfo]:
        """Get all downloaded embeddings."""
        return self.get_models_by_type(ModelType.EMBEDDING)
    
    def get_vaes(self) -> List[ModelInfo]:
        """Get all downloaded VAEs."""
        return self.get_models_by_type(ModelType.VAE)

    def add_model(self, info: ModelInfo) -> None:
        """Add or update model metadata.
        
        Serializes ModelType enum to string for JSON compatibility.
        """
        data = self._read()
        # Convert to dict and serialize ModelType enum to string
        model_dict = asdict(info)
        if "model_type" in model_dict and isinstance(model_dict["model_type"], ModelType):
            model_dict["model_type"] = model_dict["model_type"].value
        data[info.filename] = model_dict
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
    
    def get_active_vae(self) -> str | None:
        """Get the currently active VAE filename (if custom VAE is set).
        
        Returns None if using the checkpoint's embedded VAE or default VAE.
        """
        data = self._read()
        return data.get("__active_vae__", None)
    
    def set_active_vae(self, filename: str | None) -> None:
        """Set the active VAE.
        
        Args:
            filename: VAE filename to use, or None to use checkpoint's VAE.
        """
        data = self._read()
        if filename is None:
            data.pop("__active_vae__", None)
        else:
            data["__active_vae__"] = filename
        self._write(data)
