"""
Configuration for the AI Assets Toolbox frontend.
Reads settings from environment variables or a .env file.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on actual env vars

RUNPOD_API_KEY: str = os.getenv("RUNPOD_API_KEY", "")

# Multi-endpoint IDs — each worker has its own RunPod serverless endpoint.
# Backward compatibility: if the new vars are absent but the legacy
# RUNPOD_ENDPOINT_ID is set, fall back to it for both workers.
RUNPOD_UPSCALE_ENDPOINT_ID: str = os.getenv(
    "RUNPOD_UPSCALE_ENDPOINT_ID",
    os.getenv("RUNPOD_ENDPOINT_ID", ""),
)
RUNPOD_CAPTION_ENDPOINT_ID: str = os.getenv(
    "RUNPOD_CAPTION_ENDPOINT_ID",
    os.getenv("RUNPOD_ENDPOINT_ID", ""),
)

# RunPod public endpoint for Qwen-Image-Edit (used by the spritesheet tab).
RUNPOD_IMAGE_EDIT_ENDPOINT: str = os.getenv(
    "RUNPOD_IMAGE_EDIT_ENDPOINT",
    "https://api.runpod.ai/v2/qwen-image-edit-2511-lora",
)

# Legacy alias kept for any code that still references RUNPOD_ENDPOINT_ID.
RUNPOD_ENDPOINT_ID: str = os.getenv("RUNPOD_ENDPOINT_ID", RUNPOD_UPSCALE_ENDPOINT_ID)

# Tile processing defaults
TILE_SIZE: int = 1024
DEFAULT_OVERLAP: int = 128
MAX_RESOLUTION: int = 8192

# Request timeouts (seconds)
RUNSYNC_TIMEOUT: int = 300   # 5 minutes for synchronous jobs
POLL_INTERVAL: float = 2.0   # seconds between status polls
POLL_TIMEOUT: int = 600      # 10 minutes max for async jobs
