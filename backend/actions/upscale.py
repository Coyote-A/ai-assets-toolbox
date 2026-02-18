"""
Upscale action handler.

Processes image tiles through a diffusion img2img pipeline (SDXL/Illustrious)
with optional LoRA, ControlNet, and IP-Adapter conditioning.

Request schema
--------------
{
    "action": "upscale",
    "model_config": {
        "base_model": "illustrious-xl",   # short model name
        "loras": [
            {"name": "detail-enhancer", "weight": 0.7}
        ],
        "controlnet": {
            "enabled": true,
            "model": "sdxl-tile",
            "conditioning_scale": 0.6
        }
    },
    "generation_params": {
        "steps": 30,
        "cfg_scale": 7.0,
        "denoising_strength": 0.35,
        "seed": 42
    },
    "global_prompt": "masterpiece, best quality",
    "negative_prompt": "blurry, low quality",
    "ip_adapter_enabled": false,
    "ip_adapter_image": "<base64>",
    "ip_adapter_scale": 0.6,
    "tiles": [
        {
            "tile_id": "0_0",
            "image_b64": "<base64>",
            "prompt_override": null
        }
    ]
}

Response schema
---------------
{
    "tiles": [
        {"tile_id": "0_0", "image_b64": "<base64>", "seed_used": 42},
        ...
    ]
}
"""

import logging
import random
from typing import Any, Optional

from model_manager import ModelManager, CHECKPOINT_MODELS, CONTROLNET_MODELS
from utils.image_utils import b64_to_pil, pil_to_b64

logger = logging.getLogger(__name__)


def handle_upscale(job_input: dict[str, Any]) -> dict[str, Any]:
    """
    Handle the ``upscale`` action.

    Parameters
    ----------
    job_input:
        The ``input`` dict from the RunPod job payload.

    Returns
    -------
    dict
        ``{"tiles": [{"tile_id": ..., "image_b64": ..., "seed_used": ...}, ...]}``
    """
    model_config: dict[str, Any] = job_input.get("model_config", {})
    gen_params: dict[str, Any] = job_input.get("generation_params", {})
    global_prompt: str = job_input.get("global_prompt", "")
    negative_prompt: str = job_input.get("negative_prompt", "")
    tiles: list[dict[str, Any]] = job_input.get("tiles", [])

    # Optional generation resolution override.
    # When set, the pipeline generates at this size regardless of the input tile size.
    # The frontend is responsible for upscaling the input tile to this size before
    # sending it, and for downscaling the result back to the grid tile size after.
    target_width: Optional[int] = job_input.get("target_width")
    target_height: Optional[int] = job_input.get("target_height")
    if target_width is not None:
        target_width = int(target_width)
    if target_height is not None:
        target_height = int(target_height)

    # IP-Adapter params (all optional)
    ip_adapter_enabled: bool = bool(job_input.get("ip_adapter_enabled", False))
    ip_adapter_image_b64: Optional[str] = job_input.get("ip_adapter_image")
    ip_adapter_scale: float = float(job_input.get("ip_adapter_scale", 0.6))

    if not tiles:
        raise ValueError("No 'tiles' provided in upscale request")

    # ------------------------------------------------------------------
    # Parse model config
    # ------------------------------------------------------------------
    base_model: str = model_config.get("base_model", "illustrious-xl")
    loras: list[dict[str, Any]] = model_config.get("loras", [])
    controlnet_cfg: dict[str, Any] = model_config.get("controlnet", {})
    controlnet_enabled: bool = controlnet_cfg.get("enabled", False)
    controlnet_model_name: str = controlnet_cfg.get("model", "sdxl-tile")
    controlnet_conditioning_scale: float = float(controlnet_cfg.get("conditioning_scale", 0.6))

    # ------------------------------------------------------------------
    # Parse generation params
    # ------------------------------------------------------------------
    steps: int = int(gen_params.get("steps", 30))
    cfg_scale: float = float(gen_params.get("cfg_scale", 7.0))
    strength: float = float(gen_params.get("denoising_strength", 0.35))
    base_seed: Optional[int] = gen_params.get("seed")

    logger.info(
        "Upscale action: model='%s' tiles=%d steps=%d strength=%.2f controlnet=%s ip_adapter=%s target_size=%s",
        base_model,
        len(tiles),
        steps,
        strength,
        controlnet_enabled,
        ip_adapter_enabled,
        f"{target_width}x{target_height}" if target_width and target_height else "from_image",
    )

    # ------------------------------------------------------------------
    # Load model via ModelManager
    # ------------------------------------------------------------------
    manager = ModelManager.get_instance()
    manager.load_diffusion_model(base_model)

    # Apply LoRAs — all provided LoRAs are applied in order
    if loras:
        for lora_entry in loras:
            lora_name_val: str = lora_entry.get("name", "")
            lora_weight_val: float = float(lora_entry.get("weight", 1.0))
            if lora_name_val:
                logger.info("Applying LoRA '%s' with weight %.2f", lora_name_val, lora_weight_val)
                manager.apply_lora(lora_name_val, weight=lora_weight_val)

    # ------------------------------------------------------------------
    # IP-Adapter: load or unload as needed
    # ------------------------------------------------------------------
    ip_adapter_pil: Optional[Any] = None
    if ip_adapter_enabled and ip_adapter_image_b64:
        manager.load_ip_adapter(scale=ip_adapter_scale)
        if manager._ip_adapter_loaded:
            raw_ip_img = b64_to_pil(ip_adapter_image_b64)
            # Resize to 224×224 as expected by the ViT-H CLIP image encoder
            ip_adapter_pil = raw_ip_img.resize((224, 224))
            logger.info("IP-Adapter style image prepared (224×224)")
        else:
            logger.warning("IP-Adapter requested but failed to load — proceeding without it")
    else:
        # Ensure IP-Adapter is unloaded when not needed
        manager.unload_ip_adapter()

    # ------------------------------------------------------------------
    # Build the pipeline wrapper
    # ------------------------------------------------------------------
    pipe = manager.diffusion_pipe  # access the raw pipeline via public property

    # ------------------------------------------------------------------
    # Process tiles
    # ------------------------------------------------------------------
    results: list[dict[str, Any]] = []

    for tile_entry in tiles:
        tile_id: str = tile_entry.get("tile_id", "unknown")
        image_b64: str = tile_entry.get("image_b64", "")
        prompt_override: Optional[str] = tile_entry.get("prompt_override")

        if not image_b64:
            logger.warning("Tile '%s' has no image data — skipping", tile_id)
            results.append({"tile_id": tile_id, "image_b64": "", "seed_used": None})
            continue

        # Compose prompt: global + tile-specific override
        if prompt_override:
            prompt = f"{global_prompt}, {prompt_override}" if global_prompt else prompt_override
        else:
            prompt = global_prompt

        # Determine seed for this tile
        tile_seed: int = base_seed if base_seed is not None else random.randint(0, 2**32 - 1)

        pil_image = b64_to_pil(image_b64)
        logger.info(
            "Processing tile '%s' (%dx%d) seed=%d",
            tile_id,
            pil_image.width,
            pil_image.height,
            tile_seed,
        )

        output_image = _run_sdxl(
            pipe=pipe,
            image=pil_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=tile_seed,
            controlnet_enabled=controlnet_enabled,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ip_adapter_image=ip_adapter_pil,
            target_width=target_width,
            target_height=target_height,
        )

        result_b64 = pil_to_b64(output_image)
        results.append({"tile_id": tile_id, "image_b64": result_b64, "seed_used": tile_seed})
        logger.info("Tile '%s' processed successfully", tile_id)

    return {"tiles": results}


# ---------------------------------------------------------------------------
# Internal generation helpers
# ---------------------------------------------------------------------------

def _run_sdxl(
    pipe: Any,
    image: Any,
    prompt: str,
    negative_prompt: str,
    strength: float,
    steps: int,
    cfg_scale: float,
    seed: int,
    controlnet_enabled: bool,
    controlnet_conditioning_scale: float,
    ip_adapter_image: Optional[Any] = None,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
) -> Any:
    """
    Run the SDXL img2img pipeline.

    Parameters
    ----------
    target_width, target_height:
        Optional explicit output dimensions for the pipeline.  When provided
        (and different from the input image size), the pipeline is instructed
        to generate at this resolution.  The frontend is responsible for
        upscaling the input tile to this size before calling this function and
        for downscaling the result back to the grid tile size afterwards.
        When ``None``, the pipeline derives the output size from the input image.
    """
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": image,
        "strength": strength,
        "num_inference_steps": steps,
        "guidance_scale": cfg_scale,
        "generator": generator,
    }

    # Pass explicit width/height to the pipeline when provided.
    # This ensures the diffusion model generates at the requested resolution
    # (e.g. 1536×1536 for Illustrious-XL native resolution) rather than
    # inferring it from the input image dimensions.
    if target_width is not None:
        kwargs["width"] = target_width
    if target_height is not None:
        kwargs["height"] = target_height

    if controlnet_enabled:
        kwargs["control_image"] = image
        kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale

    if ip_adapter_image is not None:
        kwargs["ip_adapter_image"] = ip_adapter_image

    result = pipe(**kwargs)
    return result.images[0]



