"""
Region-based upscale action handler.

Processes image regions through a diffusion img2img pipeline (SDXL/Illustrious)
with optional LoRA, ControlNet, and IP-Adapter conditioning. Each region is
extracted from the source image with padding applied, processed individually
with its own prompt, and returned as an upscaled region.

Request schema
--------------
{
    "action": "upscale_regions",
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
    "source_image_b64": "<base64>",  # Original full image
    "target_resolution": {"width": 8192, "height": 8192},
    "regions": [
        {
            "region_id": "region_1",
            "x": 100, "y": 100, "w": 512, "h": 512,
            "padding": 64,
            "prompt": "detailed face",
            "negative_prompt": "blurry face"
        }
    ]
}

Response schema
---------------
{
    "regions": [
        {
            "region_id": "region_1",
            "image_b64": "<base64>",
            "seed_used": 42,
            "original_bbox": {"x": 100, "y": 100, "w": 512, "h": 512},
            "padded_bbox": {"x": 36, "y": 36, "w": 640, "h": 640}
        },
        ...
    ]
}
"""

import logging
import random
from typing import Any, Optional

from model_manager import ModelManager, HARDCODED_LORAS
from utils.image_utils import b64_to_pil, pil_to_b64
from pipelines.sdxl_pipeline import _build_compel, _encode_prompt_with_compel

logger = logging.getLogger(__name__)

# Module-level compel instance — built lazily on first use and reused across calls.
_compel_instance = None


def _get_compel(pipe: Any) -> Any:
    """Return (or lazily build) the module-level Compel instance for *pipe*."""
    global _compel_instance  # noqa: PLW0603
    if _compel_instance is None:
        _compel_instance = _build_compel(pipe)
    return _compel_instance


def handle_upscale_regions(job_input: dict[str, Any]) -> dict[str, Any]:
    """
    Handle the ``upscale_regions`` action.

    Parameters
    ----------
    job_input:
        The ``input`` dict from the RunPod job payload.

    Returns
    -------
    dict
        ``{"regions": [{"region_id": ..., "image_b64": ..., "seed_used": ..., 
        "original_bbox": {...}, "padded_bbox": {...}}, ...]}``
    """
    model_config: dict[str, Any] = job_input.get("model_config", {})
    gen_params: dict[str, Any] = job_input.get("generation_params", {})
    global_prompt: str = job_input.get("global_prompt", "")
    negative_prompt: str = job_input.get("negative_prompt", "")
    source_image_b64: str = job_input.get("source_image_b64", "")
    target_resolution: dict[str, int] = job_input.get("target_resolution", {})
    regions: list[dict[str, Any]] = job_input.get("regions", [])

    # Optional generation resolution override (mirrors upscale.py behaviour).
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

    if not source_image_b64:
        raise ValueError("No 'source_image_b64' provided in upscale_regions request")

    if not regions:
        raise ValueError("No 'regions' provided in upscale_regions request")

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
        "Upscale regions action: model='%s' regions=%d steps=%d strength=%.2f controlnet=%s ip_adapter=%s target_size=%s",
        base_model,
        len(regions),
        steps,
        strength,
        controlnet_enabled,
        ip_adapter_enabled,
        f"{target_width}x{target_height}" if target_width and target_height else "from_image",
    )

    # ------------------------------------------------------------------
    # Inject trigger words for active LoRAs into the global prompt
    # (backend-side injection as a safety net; frontend also injects them)
    # ------------------------------------------------------------------
    injected_triggers: list[str] = []
    for lora in loras:
        lora_name = lora.get("name", "")
        lora_info = HARDCODED_LORAS.get(lora_name, {})
        for trigger in lora_info.get("trigger_words", []):
            if trigger and trigger not in global_prompt and trigger not in injected_triggers:
                injected_triggers.append(trigger)
    if injected_triggers:
        trigger_text = ", ".join(injected_triggers)
        global_prompt = f"{trigger_text}, {global_prompt}" if global_prompt else trigger_text
        logger.info("Injected trigger words into prompt: %s", injected_triggers)

    # ------------------------------------------------------------------
    # Load source image
    # ------------------------------------------------------------------
    source_image = b64_to_pil(source_image_b64)
    logger.info(
        "Source image loaded: %dx%d",
        source_image.width,
        source_image.height,
    )

    # ------------------------------------------------------------------
    # Load model via ModelManager
    # ------------------------------------------------------------------
    manager = ModelManager.get_instance()
    manager.load_diffusion_model(base_model)

    # Apply all LoRAs simultaneously using multi-adapter support
    applied_lora_names: list[str] = []
    if loras:
        logger.info("Applying %d LoRA(s) simultaneously: %s", len(loras), [l.get("name") for l in loras])
        applied_lora_names = manager.apply_loras(loras)

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
    # Process regions
    # ------------------------------------------------------------------
    results: list[dict[str, Any]] = []

    for region_entry in regions:
        region_id: str = region_entry.get("region_id", "unknown")
        x: int = int(region_entry.get("x", 0))
        y: int = int(region_entry.get("y", 0))
        w: int = int(region_entry.get("w", 0))
        h: int = int(region_entry.get("h", 0))
        padding: int = int(region_entry.get("padding", 0))
        region_prompt: str = region_entry.get("prompt", "")
        region_negative_prompt: str = region_entry.get("negative_prompt", "")

        # Calculate padded bounding box
        padded_x = max(0, x - padding)
        padded_y = max(0, y - padding)
        padded_w = w + 2 * padding
        padded_h = h + 2 * padding

        # Clamp to image bounds
        padded_x = min(padded_x, source_image.width - 1)
        padded_y = min(padded_y, source_image.height - 1)
        padded_w = min(padded_w, source_image.width - padded_x)
        padded_h = min(padded_h, source_image.height - padded_y)

        padded_bbox = {
            "x": padded_x,
            "y": padded_y,
            "w": padded_w,
            "h": padded_h,
        }

        original_bbox = {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        }

        # Extract region from source image
        region_image = source_image.crop((padded_x, padded_y, padded_x + padded_w, padded_y + padded_h))
        logger.info(
            "Extracted region '%s': original=(%d,%d,%d,%d) padded=(%d,%d,%d,%d) -> %dx%d",
            region_id,
            x, y, w, h,
            padded_x, padded_y, padded_w, padded_h,
            region_image.width, region_image.height,
        )

        # Compose prompt: global + region-specific
        if region_prompt:
            prompt = f"{global_prompt}, {region_prompt}" if global_prompt else region_prompt
        else:
            prompt = global_prompt

        # Compose negative prompt: global + region-specific
        if region_negative_prompt:
            neg_prompt = f"{negative_prompt}, {region_negative_prompt}" if negative_prompt else region_negative_prompt
        else:
            neg_prompt = negative_prompt

        # Determine seed for this region
        region_seed: int = base_seed if base_seed is not None else random.randint(0, 2**32 - 1)

        logger.info(
            "Processing region '%s' (%dx%d) seed=%d prompt='%s'",
            region_id,
            region_image.width,
            region_image.height,
            region_seed,
            prompt[:50] + "..." if len(prompt) > 50 else prompt,
        )

        output_image = _run_sdxl(
            pipe=pipe,
            image=region_image,
            prompt=prompt,
            negative_prompt=neg_prompt,
            strength=strength,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=region_seed,
            controlnet_enabled=controlnet_enabled,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ip_adapter_image=ip_adapter_pil,
            target_width=target_width,
            target_height=target_height,
        )

        result_b64 = pil_to_b64(output_image)
        results.append({
            "region_id": region_id,
            "image_b64": result_b64,
            "seed_used": region_seed,
            "original_bbox": original_bbox,
            "padded_bbox": padded_bbox,
        })
        logger.info("Region '%s' processed successfully", region_id)

    # Clean up LoRAs after all regions are processed
    if applied_lora_names:
        manager.remove_loras(applied_lora_names)

    return {"regions": results}


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
    Run the SDXL img2img pipeline for a region.

    Parameters
    ----------
    target_width, target_height:
        Optional explicit output dimensions for the pipeline.  When provided,
        the pipeline generates at this resolution rather than deriving it from
        the input image dimensions.
    """
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    kwargs: dict[str, Any] = {
        "image": image,
        "strength": strength,
        "num_inference_steps": steps,
        "guidance_scale": cfg_scale,
        "generator": generator,
    }

    # ------------------------------------------------------------------
    # Prompt encoding — use compel for long-prompt support when available
    # ------------------------------------------------------------------
    compel = _get_compel(pipe)
    use_compel = compel is not None
    if use_compel:
        conditioning, pooled, neg_conditioning, neg_pooled = _encode_prompt_with_compel(
            compel, prompt, negative_prompt
        )
        if conditioning is not None:
            kwargs["prompt_embeds"] = conditioning
            kwargs["pooled_prompt_embeds"] = pooled
            kwargs["negative_prompt_embeds"] = neg_conditioning
            kwargs["negative_pooled_prompt_embeds"] = neg_pooled
        else:
            use_compel = False

    if not use_compel:
        kwargs["prompt"] = prompt
        kwargs["negative_prompt"] = negative_prompt

    # Pass explicit width/height to the pipeline when provided.
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
