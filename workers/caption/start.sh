#!/usr/bin/env bash
# ============================================================
# AI Assets Toolbox — Caption Worker Startup Script
# ============================================================
# This script is called by the Dockerfile CMD.
#
# Qwen3-VL-2B-Instruct is pre-downloaded into the Docker image
# at /app/models/qwen3-vl-2b/ during build time — no model
# downloads happen at startup.
# ============================================================

set -euo pipefail

echo "[start.sh] Starting AI Assets Toolbox — Caption Worker"

# ---------------------------------------------------------------------------
# Log GPU info (informational)
# ---------------------------------------------------------------------------
if command -v nvidia-smi &>/dev/null; then
    echo "[start.sh] GPU info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
else
    echo "[start.sh] No GPU detected (nvidia-smi not found)"
fi

# ---------------------------------------------------------------------------
# Verify model directory
# ---------------------------------------------------------------------------
MODEL_DIR="/app/models/qwen3-vl-2b"
echo ""
echo "[start.sh] Model directory contents:"
ls -lh "${MODEL_DIR}/" 2>/dev/null || echo "[start.sh] WARNING: Model directory not found at ${MODEL_DIR}"

# ---------------------------------------------------------------------------
# Start the RunPod serverless handler
# ---------------------------------------------------------------------------
echo ""
echo "[start.sh] Starting caption handler..."
exec python3 -u /app/handler.py
