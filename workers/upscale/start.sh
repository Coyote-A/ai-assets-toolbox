#!/usr/bin/env bash
# ============================================================
# AI Assets Toolbox — Upscale Worker Startup Script
# ============================================================
# This script is called by the Dockerfile CMD.
#
# All core models (Illustrious-XL, ControlNet Tile, SDXL VAE,
# IP-Adapter, CLIP ViT-H) are pre-downloaded into the Docker
# image at /app/models/ during build time — no model downloads
# happen at startup.
#
# The network volume is used only for:
#   - LoRA .safetensors files (dynamic, user-uploaded)
#   - Output images
# ============================================================

set -euo pipefail

echo "[start.sh] Starting AI Assets Toolbox — Upscale Worker"

# ---------------------------------------------------------------------------
# Ensure network volume directories exist (LoRAs + outputs only)
# ---------------------------------------------------------------------------
VOLUME_ROOT="${RUNPOD_VOLUME_PATH:-/runpod-volume}"

echo "[start.sh] Ensuring runtime directories under ${VOLUME_ROOT}/"
mkdir -p \
    "${VOLUME_ROOT}/models/loras" \
    "${VOLUME_ROOT}/outputs"

echo "[start.sh] Directory structure ready"

# ---------------------------------------------------------------------------
# Download hardcoded CivitAI LoRAs (skips files that already exist)
# ---------------------------------------------------------------------------
echo "[start.sh] Downloading hardcoded CivitAI LoRAs (if not already present)..."
python -c "
import logging
logging.basicConfig(level=logging.INFO, format='[lora-download] %(message)s')
from model_manager import ensure_loras_downloaded
ensure_loras_downloaded()
" || echo "[start.sh] WARNING: LoRA download step encountered errors (continuing)"

# ---------------------------------------------------------------------------
# Log GPU info (informational)
# ---------------------------------------------------------------------------
if command -v nvidia-smi &>/dev/null; then
    echo "[start.sh] GPU info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

# ---------------------------------------------------------------------------
# Start the RunPod serverless handler
# ---------------------------------------------------------------------------
echo "[start.sh] Launching handler.py"
exec python -u /app/handler.py
