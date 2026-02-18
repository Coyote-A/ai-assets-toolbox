"""
RunPod Serverless Handler — AI Assets Toolbox
Entry point for all serverless job requests.
"""

import logging
import traceback
from typing import Any

import runpod

from actions.caption import handle_caption
from actions.upscale import handle_upscale
from actions.upscale_regions import handle_upscale_regions
from actions.models import handle_list_models, handle_upload_model, handle_delete_model
from model_manager import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def handle_health(job_input: dict[str, Any]) -> dict[str, Any]:
    """Return GPU and storage health information."""
    import torch
    import os
    import shutil

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

    network_storage_available_gb = 0.0
    network_storage_path = "/runpod-volume"
    if os.path.exists(network_storage_path):
        stat = shutil.disk_usage(network_storage_path)
        network_storage_available_gb = round(stat.free / (1024 ** 3), 2)

    manager = ModelManager.get_instance()
    loaded_model = manager.get_current_model_info()

    return {
        "status": "ok",
        "gpu": gpu_name,
        "vram_total_gb": vram_total_gb,
        "vram_used_gb": vram_used_gb,
        "loaded_model": loaded_model,
        "network_storage_available_gb": network_storage_available_gb,
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
        elif action == "upscale":
            result = handle_upscale(job_input)
        elif action == "upscale_regions":
            result = handle_upscale_regions(job_input)
        elif action == "list_models":
            result = handle_list_models(job_input)
        elif action == "upload_model":
            result = handle_upload_model(job_input)
        elif action == "delete_model":
            result = handle_delete_model(job_input)
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
    logger.info("Starting RunPod serverless handler")
    runpod.serverless.start({"handler": handler})
