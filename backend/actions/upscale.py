"""
Upscale action handler.

Processes image tiles through a diffusion img2img pipeline (SDXL or Flux)
with optional LoRA and ControlNet conditioning.

Request schema
--------------
{
    "action": "upscale",
    "model_config": {
        "base_model": "z-image-xl",   # short model name
        "model_type": "sdxl",         # "sdxl" | "flux"
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

from model_manager import ModelManager, CHECKPOINT_MODELS, CONTROLNET_MODELS, _is_flux_model
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

    if not tiles:
        raise ValueError("No 'tiles' provided in upscale request")

    # ------------------------------------------------------------------
    # Parse model config
    # ------------------------------------------------------------------
    base_model: str = model_config.get("base_model", "z-image-xl")
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
        "Upscale action: model='%s' tiles=%d steps=%d strength=%.2f controlnet=%s",
        base_model,
        len(tiles),
        steps,
        strength,
        controlnet_enabled,
    )

    # ------------------------------------------------------------------
    # Load model via ModelManager
    # ------------------------------------------------------------------
    manager = ModelManager.get_instance()
    manager.load_diffusion_model(base_model)

    # Apply LoRAs (only the first LoRA is applied; stacking is a future TODO)
    if loras:
        first_lora = loras[0]
        manager.apply_lora(first_lora["name"], weight=float(first_lora.get("weight", 1.0)))
        if len(loras) > 1:
            logger.warning(
                "Multiple LoRAs requested but only the first ('%s') will be applied. "
                "LoRA stacking is not yet implemented.",
                first_lora["name"],
            )

    # ------------------------------------------------------------------
    # Build the pipeline wrapper
    # ------------------------------------------------------------------
    is_flux = _is_flux_model(base_model)
    pipe = manager._diffusion_pipe  # access the raw pipeline

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

        if is_flux:
            output_image = _run_flux(
                pipe=pipe,
                image=pil_image,
                prompt=prompt,
                strength=strength,
                steps=steps,
                seed=tile_seed,
            )
        else:
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
) -> Any:
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
    if controlnet_enabled:
        kwargs["control_image"] = image
        kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale

    result = pipe(**kwargs)
    return result.images[0]


def _run_flux(
    pipe: Any,
    image: Any,
    prompt: str,
    strength: float,
    steps: int,
    seed: int,
) -> Any:
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        num_inference_steps=steps,
        generator=generator,
    )
    return result.images[0]
