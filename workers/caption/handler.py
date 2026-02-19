"""
RunPod Serverless Handler — Caption Worker

Entry point for all caption job requests. Accepts image tiles (base64-encoded),
runs each through the Qwen3-VL-2B-Instruct pipeline, and returns per-tile captions.
"""

import base64
import io
import logging
import traceback
from typing import Any

import runpod
from PIL import Image

from qwen_pipeline import QwenPipeline, DEFAULT_SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = "/app/models/qwen3-vl-2b"

# Global pipeline instance — loaded once at startup, stays in VRAM
_pipeline: QwenPipeline | None = None


def _get_pipeline() -> QwenPipeline:
    """Return the global QwenPipeline, initialising it on first call."""
    global _pipeline
    if _pipeline is None:
        logger.info("Initialising QwenPipeline from '%s'", MODEL_PATH)
        _pipeline = QwenPipeline(model_path=MODEL_PATH)
    return _pipeline


def _b64_to_pil(image_b64: str) -> Image.Image:
    """Decode a base64-encoded image string to a PIL Image."""
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def handle_caption(job_input: dict[str, Any]) -> dict[str, Any]:
    """
    Handle the ``caption`` action.

    Expected input
    --------------
    {
        "action": "caption",
        "tiles": [
            {"tile_id": "0_0", "image_b64": "<base64>"},
            ...
        ],
        "system_prompt": "optional custom prompt",   # optional
        "max_tokens": 300                             # optional
    }

    Returns
    -------
    {
        "captions": {
            "0_0": "A detailed forest scene...",
            "0_1": "A stone castle wall..."
        }
    }
    """
    tiles: list[dict[str, Any]] = job_input.get("tiles", [])
    if not tiles:
        raise ValueError("No 'tiles' field found in caption request")

    system_prompt: str = job_input.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    max_tokens: int = int(job_input.get("max_tokens", 300))

    logger.info("Caption action: processing %d tile(s)", len(tiles))

    pipeline = _get_pipeline()
    captions: dict[str, str] = {}

    for tile_entry in tiles:
        tile_id: str = tile_entry.get("tile_id", "unknown")
        image_b64: str = tile_entry.get("image_b64", "")

        if not image_b64:
            logger.warning("Tile '%s' has no image data — skipping", tile_id)
            captions[tile_id] = ""
            continue

        pil_image = _b64_to_pil(image_b64)
        logger.info("Captioning tile '%s' (%dx%d)", tile_id, pil_image.width, pil_image.height)

        caption = pipeline.caption(
            pil_image,
            system_prompt=system_prompt,
            max_new_tokens=max_tokens,
        )
        logger.info("Tile '%s' caption: %s", tile_id, caption[:80])
        captions[tile_id] = caption

    return {"captions": captions}


def handle_health(job_input: dict[str, Any]) -> dict[str, Any]:
    """Return GPU and model health information."""
    import torch
    import os

    gpu_name = "unknown"
    vram_total_gb = 0.0
    vram_used_gb = 0.0

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_total_gb = round(props.total_memory / (1024 ** 3), 2)
        vram_used_gb = round(
            (props.total_memory - torch.cuda.mem_get_info(0)[0]) / (1024 ** 3), 2
        )

    model_loaded = _pipeline is not None
    model_path_exists = os.path.isdir(MODEL_PATH)

    return {
        "status": "ok",
        "gpu": gpu_name,
        "vram_total_gb": vram_total_gb,
        "vram_used_gb": vram_used_gb,
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
        "model_path_exists": model_path_exists,
    }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """
    Main RunPod serverless handler.

    Routes incoming jobs to the appropriate action handler based on the
    ``action`` field in ``job["input"]``.
    """
    job_input: dict[str, Any] = job.get("input", {})
    action: str = job_input.get("action", "")

    logger.info("Received job id=%s action=%s", job.get("id"), action)

    try:
        if action == "caption":
            result = handle_caption(job_input)
        elif action == "health":
            result = handle_health(job_input)
        else:
            logger.warning("Unknown action: %s", action)
            return {"error": f"Unknown action: '{action}'"}

        logger.info("Job id=%s completed successfully", job.get("id"))
        return result

    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            "Job id=%s failed with exception: %s\n%s",
            job.get("id"),
            exc,
            traceback.format_exc(),
        )
        return {"error": str(exc)}


if __name__ == "__main__":
    logger.info("Starting RunPod caption worker")

    # Preload the Qwen model at startup so the first request is fast.
    # Wrapped in try/except so startup failures are logged clearly before exit.
    logger.info("Preloading Qwen3-VL-2B model...")
    try:
        _get_pipeline()
        logger.info("Model preloaded successfully, ready for jobs")
    except Exception as exc:  # pylint: disable=broad-except
        logger.critical(
            "FATAL: Model preload failed — worker cannot start.\n"
            "Exception: %s\n%s",
            exc,
            traceback.format_exc(),
        )
        raise SystemExit(1) from exc

    runpod.serverless.start({"handler": handler})
