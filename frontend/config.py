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
RUNPOD_ENDPOINT_ID: str = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_BASE_URL: str = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

# Tile processing defaults
TILE_SIZE: int = 1024
DEFAULT_OVERLAP: int = 128
MAX_RESOLUTION: int = 8192

# Request timeouts (seconds)
RUNSYNC_TIMEOUT: int = 300   # 5 minutes for synchronous jobs
POLL_INTERVAL: float = 2.0   # seconds between status polls
POLL_TIMEOUT: int = 600      # 10 minutes max for async jobs
