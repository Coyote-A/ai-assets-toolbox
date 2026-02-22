#!/bin/bash
# Build and push Docker images for RunPod serverless workers
# Usage: ./scripts/deploy.sh <dockerhub_username> [upscale|caption|all] [tag]
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
  echo "Usage: $0 <dockerhub_username> [upscale|caption|all] [tag]"
  echo "  dockerhub_username  Your Docker Hub username"
  echo "  worker              Worker to deploy: upscale, caption, or all (default: all)"
  echo "  tag                 Image tag (default: latest)"
  exit 1
fi

DOCKERHUB_USERNAME="$1"
WORKER="${2:-all}"
TAG="${3:-latest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Helper: build and push a single worker
# ---------------------------------------------------------------------------
build_and_push() {
  local worker_name="$1"
  local image_name="ai-assets-${worker_name}"
  local full_image="${DOCKERHUB_USERNAME}/${image_name}:${TAG}"
  local context="${REPO_ROOT}/workers/${worker_name}"

  echo "============================================================"
  echo "  AI Assets Toolbox — ${worker_name} Worker Deployment"
  echo "============================================================"
  echo "  Image : ${full_image}"
  echo "  Source: ${context}"
  echo "============================================================"
  echo ""

  echo "[1/3] Building Docker image..."
  docker build \
    --platform linux/amd64 \
    -t "${full_image}" \
    -f "${context}/Dockerfile" \
    "${context}"

  echo ""
  echo "[2/3] Tagging image as ${full_image}..."
  docker tag "${full_image}" "${full_image}"

  echo ""
  echo "[3/3] Pushing image to Docker Hub..."
  docker push "${full_image}"

  echo ""
  echo "============================================================"
  echo "  ${worker_name} worker build and push complete!"
  echo "============================================================"
  echo ""
  echo "Next steps — create a RunPod Serverless Endpoint:"
  echo ""
  echo "  1. Go to https://www.runpod.io/console/serverless"
  echo "  2. Click 'New Endpoint'"
  echo "  3. Set Docker image to: ${full_image}"
  if [[ "${worker_name}" == "upscale" ]]; then
    echo "  4. Attach your Network Volume (mount path: /runpod-volume/)"
    echo "  5. Select GPU: A100 80GB SXM (recommended)"
    echo "  6. Save and copy the Endpoint ID into your frontend/.env"
    echo ""
    echo "  RUNPOD_UPSCALE_ENDPOINT_ID=<your_endpoint_id>"
  else
    echo "  4. Select GPU: A100 80GB SXM (recommended)"
    echo "  5. Save and copy the Endpoint ID into your frontend/.env"
    echo ""
    echo "  RUNPOD_CAPTION_ENDPOINT_ID=<your_endpoint_id>"
  fi
  echo ""
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "${WORKER}" in
  upscale)
    build_and_push "upscale"
    ;;
  caption)
    build_and_push "caption"
    ;;
  all)
    build_and_push "upscale"
    build_and_push "caption"
    ;;
  *)
    echo "Error: unknown worker '${WORKER}'. Use: upscale, caption, or all"
    exit 1
    ;;
esac
