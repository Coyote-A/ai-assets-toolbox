#!/usr/bin/env bash
# ============================================================
# AI Assets Toolbox — RunPod Worker Startup Script
# ============================================================
# This script is called by the Dockerfile CMD.
#
# All core models (Illustrious-XL, ControlNet Tile, SDXL VAE,
# Qwen2.5-VL-7B, IP-Adapter) are pre-downloaded into the Docker
# image at /app/hf_cache during build time — no model downloads
# happen at startup.
#
# The network volume is used only for:
#   - LoRA .safetensors files (dynamic, user-uploaded)
#   - Output images
# ============================================================

set -euo pipefail

echo "[start.sh] Starting AI Assets Toolbox RunPod worker"

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
# Log GPU info (informational)
# ---------------------------------------------------------------------------
if command -v nvidia-smi &>/dev/null; then
    echo "[start.sh] GPU info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

# --- RunPod Model Cache Diagnostics ---
echo "=== RunPod Model Cache Diagnostics ==="
RUNPOD_CACHE="/runpod-volume/huggingface-cache/hub"
if [ -d "$RUNPOD_CACHE" ]; then
    echo "RunPod cache directory found at $RUNPOD_CACHE"
    echo "Cached models:"
    ls -d "$RUNPOD_CACHE"/models--* 2>/dev/null | while read dir; do
        model_name=$(basename "$dir" | sed 's/models--//' | sed 's/--/\//g')
        echo "  - $model_name"
    done
    if [ -z "$(ls -d "$RUNPOD_CACHE"/models--* 2>/dev/null)" ]; then
        echo "  (none found)"
    fi
else
    echo "RunPod cache directory not found at $RUNPOD_CACHE"
    echo "Models will be loaded from baked-in Docker image cache or downloaded live."
fi
echo "======================================"

# ---------------------------------------------------------------------------
# Start the RunPod serverless handler
# ---------------------------------------------------------------------------
echo "[start.sh] Launching handler.py"
exec python -u /app/handler.py
