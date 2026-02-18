#!/usr/bin/env bash
# ============================================================
# AI Assets Toolbox — RunPod Worker Startup Script
# ============================================================
# This script is called by the Dockerfile CMD.
# It ensures required directories exist on the network volume
# and then starts the RunPod serverless handler.
# ============================================================

set -euo pipefail

echo "[start.sh] Starting AI Assets Toolbox RunPod worker"

# ---------------------------------------------------------------------------
# Ensure network volume directories exist
# ---------------------------------------------------------------------------
VOLUME_ROOT="${RUNPOD_VOLUME_PATH:-/runpod-volume}"

echo "[start.sh] Ensuring model directories under ${VOLUME_ROOT}/models/"
mkdir -p \
    "${VOLUME_ROOT}/models/checkpoints" \
    "${VOLUME_ROOT}/models/loras" \
    "${VOLUME_ROOT}/models/controlnets" \
    "${VOLUME_ROOT}/hf_cache" \
    "${VOLUME_ROOT}/outputs"

echo "[start.sh] Directory structure ready"

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
