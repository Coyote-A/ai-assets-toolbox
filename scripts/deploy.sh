#!/bin/bash
# Build and push Docker image for RunPod serverless
# Usage: ./scripts/deploy.sh <dockerhub_username>
#
# Prerequisites:
#   - Docker installed and running
#   - Logged in to Docker Hub: docker login
#   - Run from the repository root

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dockerhub_username> [tag]"
  echo "  dockerhub_username  Your Docker Hub username"
  echo "  tag                 Image tag (default: latest)"
  exit 1
fi

DOCKERHUB_USERNAME="$1"
TAG="${2:-latest}"
IMAGE_NAME="ai-assets-toolbox-backend"
FULL_IMAGE="${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "============================================================"
echo "  AI Assets Toolbox — Backend Deployment"
echo "============================================================"
echo "  Image : ${FULL_IMAGE}"
echo "  Source: ${REPO_ROOT}/backend"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "[1/3] Building Docker image..."
docker build \
  --platform linux/amd64 \
  -t "${FULL_IMAGE}" \
  -f "${REPO_ROOT}/backend/Dockerfile" \
  "${REPO_ROOT}/backend"

echo ""
echo "[2/3] Tagging image as ${FULL_IMAGE}..."
docker tag "${FULL_IMAGE}" "${FULL_IMAGE}"

# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Pushing image to Docker Hub..."
docker push "${FULL_IMAGE}"

# ---------------------------------------------------------------------------
# Next steps
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Build and push complete!"
echo "============================================================"
echo ""
echo "Next steps — create a RunPod Serverless Endpoint:"
echo ""
echo "  1. Go to https://www.runpod.io/console/serverless"
echo "  2. Click 'New Endpoint'"
echo "  3. Set Docker image to: ${FULL_IMAGE}"
echo "  4. Attach your Network Volume (mount path: /runpod-volume/)"
echo "  5. Select GPU: A100 80GB SXM (recommended)"
echo "  6. Set environment variables if needed (see .env.example)"
echo "  7. Save and copy the Endpoint ID into your frontend/.env"
echo ""
echo "  RUNPOD_ENDPOINT_ID=<your_endpoint_id>"
echo ""
