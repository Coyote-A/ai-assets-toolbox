"""
Settings storage service using Modal Dict for server-side persistence.

This module provides a server-side settings storage solution that persists
across server restarts. Settings are stored in a Modal Dict, which provides:
- No reload needed: Dict is always current across all containers
- Atomic updates: Built-in concurrency control
- Faster access: In-memory reads
- Simpler code: No file I/O handling

Architecture
------------
- Settings are stored in a Modal Dict (ai-toolbox-settings)
- Single user deployment - no session ID needed
- On page load, settings are read from the Dict
- When settings are saved, they're written to the Dict

Excluded from Persistence
-------------------------
- Seed: Intentionally random each run (-1)
- IP-Adapter Style Image: Binary image data, user-specific per session
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

from src.app_config import settings_store

logger = logging.getLogger(__name__)

# Key within the Dict for storing upscale settings
UPSCALE_SETTINGS_KEY = "upscale_settings"


@dataclass
class UpscaleSettings:
    """Container for upscale tab generation settings."""

    # Grid Settings
    tile_size: int = 1024
    overlap: int = 128
    gen_resolution: str = "1536×1536 ✨ Recommended"

    # Generation Settings
    checkpoint_name: Optional[str] = None
    global_prompt: str = ""
    negative_prompt: str = ""
    denoise_strength: float = 0.35
    steps: int = 30
    cfg_scale: float = 7.0

    # LoRA Selections (JSON-serializable list)
    lora_selections: List[Dict[str, Any]] = field(default_factory=list)

    # ControlNet Settings
    controlnet_enabled: bool = True
    conditioning_scale: float = 0.7

    # IP-Adapter Settings
    ip_adapter_enabled: bool = False
    ip_adapter_scale: float = 0.6

    # Seam Fix Settings
    seam_fix_enabled: bool = False
    seam_fix_strength: float = 0.35
    seam_fix_feather: int = 32


def get_upscale_settings() -> UpscaleSettings:
    """
    Retrieve upscale settings from the Modal Dict.

    Returns:
        UpscaleSettings with all values (defaults if not set)
    """
    try:
        data = settings_store.get(UPSCALE_SETTINGS_KEY, {})

        if not data:
            logger.debug("get_upscale_settings: no saved settings, using defaults")
            return UpscaleSettings()

        settings = UpscaleSettings(
            tile_size=data.get("tile_size", 1024),
            overlap=data.get("overlap", 128),
            gen_resolution=data.get("gen_resolution", "1536×1536 ✨ Recommended"),
            checkpoint_name=data.get("checkpoint_name"),
            global_prompt=data.get("global_prompt", ""),
            negative_prompt=data.get("negative_prompt", ""),
            denoise_strength=data.get("denoise_strength", 0.35),
            steps=data.get("steps", 30),
            cfg_scale=data.get("cfg_scale", 7.0),
            lora_selections=data.get("lora_selections", []),
            controlnet_enabled=data.get("controlnet_enabled", True),
            conditioning_scale=data.get("conditioning_scale", 0.7),
            ip_adapter_enabled=data.get("ip_adapter_enabled", False),
            ip_adapter_scale=data.get("ip_adapter_scale", 0.6),
            seam_fix_enabled=data.get("seam_fix_enabled", False),
            seam_fix_strength=data.get("seam_fix_strength", 0.35),
            seam_fix_feather=data.get("seam_fix_feather", 32),
        )

        logger.debug("get_upscale_settings: loaded settings from Dict")
        return settings

    except Exception:
        logger.exception("get_upscale_settings: error retrieving settings")
        return UpscaleSettings()


def save_upscale_settings(settings: UpscaleSettings) -> None:
    """
    Save upscale settings to the Modal Dict.

    Args:
        settings: UpscaleSettings dataclass instance
    """
    try:
        data = asdict(settings)
        settings_store[UPSCALE_SETTINGS_KEY] = data
        logger.info("save_upscale_settings: saved settings to Dict")
    except Exception:
        logger.exception("save_upscale_settings: error saving settings")


def save_upscale_setting(key: str, value: Any) -> None:
    """
    Save a single setting to the Modal Dict.

    This is useful for incremental updates without loading/saving all settings.

    Args:
        key: Setting name (must match UpscaleSettings field)
        value: Setting value
    """
    try:
        data = settings_store.get(UPSCALE_SETTINGS_KEY, {})
        data[key] = value
        settings_store[UPSCALE_SETTINGS_KEY] = data
        logger.debug("save_upscale_setting: saved %s", key)
    except Exception:
        logger.exception("save_upscale_setting: error saving setting")
