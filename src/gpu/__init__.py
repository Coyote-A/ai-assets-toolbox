"""
GPU service classes for the unified Modal app.

Both classes are decorated with ``@app.cls()`` (from ``src.app_config``) so
importing them here is enough to register them against the shared
``modal.App("ai-toolbox")`` instance.
"""

from src.gpu.caption import CaptionService
from src.gpu.upscale import UpscaleService

__all__ = ["CaptionService", "UpscaleService"]
