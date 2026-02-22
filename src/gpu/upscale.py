"""
Upscale service — Illustrious-XL + ControlNet Tile on a Modal A10G GPU container.

This module is part of the unified Modal app (``modal.App("ai-toolbox")``).
It does NOT define its own ``modal.App``; instead it imports ``app``,
``upscale_image``, and ``lora_volume`` from ``src.app_config`` and registers
``UpscaleService`` against that shared app via ``@app.cls()``.

Design decisions vs. the old ``modal/upscale_app.py``
------------------------------------------------------
* **No web endpoint** — each action is a separate ``@modal.method()`` called
  via ``.remote()`` from the Gradio UI.  No action-routing dispatcher needed.
* **No auth** — Modal CLI token handles authentication automatically.
* **No base64** — images are passed as raw ``bytes`` (PNG/JPEG).  Modal's
  cloudpickle serialisation handles bytes efficiently.
* **No RunPod-compatible response wrapper** — exceptions propagate directly to
  the caller; no ``{"status": "FAILED"}`` dicts.
* **Chunked upload replaced by single call** — ``upload_model`` accepts raw
  ``bytes`` directly instead of base64-encoded chunks.  For very large files
  the caller can still split and call multiple times with ``append=True``.

Public interface
----------------
``upscale_tiles(tiles, model_config, gen_params, ...) -> list[dict]``
    Input tiles:  ``[{"tile_id": "0_0", "image_bytes": <bytes>}, ...]``
    Output tiles: ``[{"tile_id": "0_0", "image_bytes": <bytes>, "seed_used": 42}, ...]``

``upscale_regions(source_image_bytes, regions, model_config, gen_params, ...) -> list[dict]``
    Input:  raw bytes of the full source image + list of region dicts.
    Output: ``[{"region_id": ..., "image_bytes": <bytes>, "seed_used": ...,
                "original_bbox": {...}, "padded_bbox": {...}}, ...]``

``list_models() -> list[dict]``
    Returns ``[{"name": ..., "path": ..., "size_mb": ...}, ...]``

``upload_model(filename, file_data, append) -> dict``
    Saves raw bytes to the LoRA volume.

``delete_model(filename) -> dict``
    Removes a LoRA file from the volume.

``health() -> dict``
    GPU and storage health information.
"""

import io
import logging
import os
import random
import shutil
from typing import Any, Optional

import modal

from src.app_config import app, upscale_image, lora_volume

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — model paths baked into the container image at build time
# ---------------------------------------------------------------------------

MODELS_DIR = "/app/models"

ILLUSTRIOUS_XL_PATH = f"{MODELS_DIR}/illustrious-xl/Illustrious-XL-v2.0.safetensors"
CONTROLNET_TILE_PATH = f"{MODELS_DIR}/controlnet-tile"
SDXL_VAE_PATH = f"{MODELS_DIR}/sdxl-vae"
IP_ADAPTER_PATH = f"{MODELS_DIR}/ip-adapter/ip-adapter-plus_sdxl_vit-h.safetensors"
CLIP_VIT_H_PATH = f"{MODELS_DIR}/clip-vit-h"

# LoRA volume mount point (must match the volumes= dict in @app.cls)
LORAS_DIR = "/vol/loras"

# ---------------------------------------------------------------------------
# Hardcoded CivitAI LoRAs — downloaded to the volume on first container start
# ---------------------------------------------------------------------------

HARDCODED_LORAS: dict[str, dict[str, Any]] = {
    "Aesthetic Quality": {
        "civitai_model_id": 929497,
        "filename": "lora_929497.safetensors",
        "trigger_words": ["masterpiece", "best quality", "very aesthetic"],
    },
    "Detailer IL": {
        "civitai_model_id": 1231943,
        "filename": "lora_1231943.safetensors",
        "trigger_words": ["Jeddtl02"],
    },
}

# ---------------------------------------------------------------------------
# UpscaleService
# ---------------------------------------------------------------------------


@app.cls(
    # A10G has 24 GB VRAM — enough for SDXL + ControlNet + IP-Adapter with CPU offload.
    gpu="A10G",
    image=upscale_image,
    # Pull CivitAI token from the named Modal secret store.
    secrets=[modal.Secret.from_name("ai-toolbox-secrets")],
    # Keep the container alive for 5 minutes after the last request.
    scaledown_window=300,
    # Hard timeout per request (10 minutes for large tile batches).
    timeout=600,
    # Mount the LoRA volume at /vol/loras/
    volumes={LORAS_DIR: lora_volume},
)
class UpscaleService:
    """
    SDXL/Illustrious-XL upscale service with ControlNet Tile, IP-Adapter, and LoRA support.

    Lifecycle
    ---------
    1. Modal starts a container and calls ``load_models()`` via ``@modal.enter()``.
    2. The pipeline stays resident in VRAM for the lifetime of the container.
    3. Each ``.remote()`` call dispatches to the appropriate ``@modal.method()``.
    4. After ``container_idle_timeout`` seconds of inactivity the container is
       torn down; the next call triggers a new cold start.

    Calling from the Gradio UI (or locally)
    ----------------------------------------
    .. code-block:: python

        from src.gpu.upscale import UpscaleService
        import io
        from PIL import Image

        buf = io.BytesIO()
        pil_tile.save(buf, format="PNG")

        results = UpscaleService().upscale_tiles.remote(
            tiles=[{"tile_id": "0_0", "image_bytes": buf.getvalue()}],
            model_config={"loras": [], "controlnet": {"enabled": True, "conditioning_scale": 0.6}},
            gen_params={"steps": 30, "cfg_scale": 7.0, "denoising_strength": 0.35, "seed": None},
            global_prompt="masterpiece, best quality",
            negative_prompt="blurry, low quality",
        )
        # results == [{"tile_id": "0_0", "image_bytes": <bytes>, "seed_used": 12345}]
    """

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    @modal.enter()
    def load_models(self) -> None:
        """
        Load all models once when the container starts.

        Called exactly once per container instance by Modal before any method
        receives a call.
        """
        import torch
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            DPMSolverMultistepScheduler,
            StableDiffusionXLControlNetImg2ImgPipeline,
            StableDiffusionXLImg2ImgPipeline,
        )

        logger.info("=== UpscaleService: loading models ===")

        # ------------------------------------------------------------------
        # 1. Load VAE fp16-fix
        # ------------------------------------------------------------------
        logger.info("Loading SDXL VAE fp16-fix from '%s'", SDXL_VAE_PATH)
        vae = AutoencoderKL.from_pretrained(
            SDXL_VAE_PATH,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        logger.info("VAE loaded")

        # ------------------------------------------------------------------
        # 2. Load ControlNet Tile
        # ------------------------------------------------------------------
        logger.info("Loading ControlNet Tile from '%s'", CONTROLNET_TILE_PATH)
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_TILE_PATH,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        logger.info("ControlNet loaded")

        # ------------------------------------------------------------------
        # 3. Load SDXL pipeline from single-file checkpoint, then wrap with
        #    ControlNet using from_pipe() (avoids double-loading UNet weights)
        # ------------------------------------------------------------------
        logger.info("Loading SDXL base pipeline from '%s'", ILLUSTRIOUS_XL_PATH)
        base_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            ILLUSTRIOUS_XL_PATH,
            torch_dtype=torch.float16,
            use_safetensors=True,
            vae=vae,
        )
        logger.info("Wrapping with ControlNet via from_pipe()")
        self._pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pipe(
            base_pipe,
            controlnet=controlnet,
        )

        # ------------------------------------------------------------------
        # 4. Set scheduler
        # ------------------------------------------------------------------
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )

        # ------------------------------------------------------------------
        # 5. Enable CPU offload (keeps inactive components in CPU RAM)
        # ------------------------------------------------------------------
        self._pipe.enable_model_cpu_offload()
        logger.info("Pipeline ready with CPU offload enabled")

        # ------------------------------------------------------------------
        # 6. Build Compel for long-prompt encoding
        # ------------------------------------------------------------------
        self._compel = _build_compel(self._pipe)

        # ------------------------------------------------------------------
        # 7. Load IP-Adapter at startup (avoids lazy-load on first inference)
        # ------------------------------------------------------------------
        self._ip_adapter_loaded: bool = False
        logger.info(
            "Loading IP-Adapter from '%s' with CLIP encoder '%s'",
            IP_ADAPTER_PATH,
            CLIP_VIT_H_PATH,
        )
        try:
            self._pipe.load_ip_adapter(
                os.path.dirname(IP_ADAPTER_PATH),
                subfolder="",
                weight_name=os.path.basename(IP_ADAPTER_PATH),
                image_encoder_folder=CLIP_VIT_H_PATH,
            )
            # Move the image encoder to CUDA so it matches the pipeline device.
            # Without this the encoder stays on CPU while the pipeline runs on
            # CUDA, causing a device-mismatch RuntimeError during inference.
            if hasattr(self._pipe, "image_encoder") and self._pipe.image_encoder is not None:
                self._pipe.image_encoder.to("cuda")
                logger.info("IP-Adapter image_encoder moved to CUDA")
            # Start with scale 0 — will be set to the requested value when
            # IP-Adapter is actually used, or kept at 0 to disable it.
            self._pipe.set_ip_adapter_scale(0.0)
            self._ip_adapter_loaded = True
            logger.info("IP-Adapter loaded successfully at startup (scale=0.0)")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Failed to load IP-Adapter at startup — will proceed without it: %s", exc
            )

        # ------------------------------------------------------------------
        # 8. Download hardcoded CivitAI LoRAs to the volume (if missing)
        # ------------------------------------------------------------------
        _ensure_loras_downloaded(LORAS_DIR, lora_volume)

        logger.info("=== UpscaleService: all models loaded ===")

    # ------------------------------------------------------------------
    # Public Modal methods — upscale
    # ------------------------------------------------------------------

    @modal.method()
    def upscale_tiles(
        self,
        tiles: list[dict],
        model_config: dict,
        gen_params: dict,
        global_prompt: str = "",
        negative_prompt: str = "",
        ip_adapter_enabled: bool = False,
        ip_adapter_image_bytes: Optional[bytes] = None,
        ip_adapter_scale: float = 0.6,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> list[dict]:
        """
        Process image tiles through the SDXL img2img pipeline.

        Parameters
        ----------
        tiles:
            List of tile dicts, each with:
            - ``"tile_id"`` (str): unique identifier, e.g. ``"0_0"``.
            - ``"image_bytes"`` (bytes): raw PNG or JPEG bytes of the tile.
            - ``"prompt_override"`` (str, optional): per-tile prompt suffix.
        model_config:
            Dict with keys:
            - ``"loras"`` (list[dict]): each with ``"name"`` and ``"weight"``.
            - ``"controlnet"`` (dict): ``"enabled"`` (bool) and
              ``"conditioning_scale"`` (float).
        gen_params:
            Dict with keys:
            - ``"steps"`` (int, default 30)
            - ``"cfg_scale"`` (float, default 7.0)
            - ``"denoising_strength"`` (float, default 0.35)
            - ``"seed"`` (int or None — None means random per tile)
        global_prompt:
            Positive prompt applied to all tiles (trigger words are injected
            automatically for active LoRAs).
        negative_prompt:
            Negative prompt applied to all tiles.
        ip_adapter_enabled:
            Whether to use IP-Adapter style conditioning.
        ip_adapter_image_bytes:
            Raw bytes of the style reference image (PNG/JPEG).
        ip_adapter_scale:
            IP-Adapter influence scale (0.0–1.0).
        target_width, target_height:
            Optional explicit output dimensions for each tile.

        Returns
        -------
        list[dict]
            One dict per input tile:
            - ``"tile_id"`` (str)
            - ``"image_bytes"`` (bytes): PNG bytes of the processed tile.
            - ``"seed_used"`` (int): the seed that was used.
            Tiles with missing ``image_bytes`` have ``"image_bytes": b""`` and
            ``"seed_used": None``.

        Raises
        ------
        ValueError
            If ``tiles`` is empty.
        """
        if not tiles:
            raise ValueError("'tiles' list must not be empty")

        # Parse model config
        loras: list[dict] = model_config.get("loras", [])
        controlnet_cfg: dict = model_config.get("controlnet", {})
        controlnet_enabled: bool = bool(controlnet_cfg.get("enabled", False))
        controlnet_conditioning_scale: float = float(
            controlnet_cfg.get("conditioning_scale", 0.6)
        )

        # Parse generation params
        steps: int = int(gen_params.get("steps", 30))
        cfg_scale: float = float(gen_params.get("cfg_scale", 7.0))
        strength: float = float(gen_params.get("denoising_strength", 0.35))
        base_seed: Optional[int] = gen_params.get("seed")

        if target_width is not None:
            target_width = int(target_width)
        if target_height is not None:
            target_height = int(target_height)

        logger.info(
            "upscale_tiles: tiles=%d steps=%d strength=%.2f controlnet=%s "
            "ip_adapter=%s target=%s",
            len(tiles),
            steps,
            strength,
            controlnet_enabled,
            ip_adapter_enabled,
            f"{target_width}x{target_height}" if target_width and target_height else "from_image",
        )

        # Inject trigger words for active LoRAs into the global prompt
        global_prompt = _inject_trigger_words(global_prompt, loras)

        # Apply LoRAs
        lora_volume.reload()
        applied_lora_names: list[str] = []
        if loras:
            applied_lora_names = _apply_loras(self._pipe, loras, LORAS_DIR)

        # IP-Adapter
        ip_adapter_pil: Optional[Any] = None
        if ip_adapter_enabled and ip_adapter_image_bytes:
            self._load_ip_adapter(scale=ip_adapter_scale)
            if self._ip_adapter_loaded:
                raw_ip_img = _bytes_to_pil(ip_adapter_image_bytes)
                ip_adapter_pil = raw_ip_img.resize((224, 224))
                logger.info("IP-Adapter style image prepared (224×224)")
            else:
                logger.warning(
                    "IP-Adapter requested but failed to load — proceeding without it"
                )
        else:
            self._unload_ip_adapter()

        # Process tiles
        results: list[dict] = []

        for tile_entry in tiles:
            tile_id: str = tile_entry.get("tile_id", "unknown")
            image_bytes: bytes = tile_entry.get("image_bytes", b"")
            prompt_override: Optional[str] = tile_entry.get("prompt_override")

            if not image_bytes:
                logger.warning("Tile '%s' has no image data — skipping", tile_id)
                results.append({"tile_id": tile_id, "image_bytes": b"", "seed_used": None})
                continue

            # Compose prompt
            if prompt_override:
                prompt = f"{global_prompt}, {prompt_override}" if global_prompt else prompt_override
            else:
                prompt = global_prompt

            tile_seed: int = base_seed if base_seed is not None else random.randint(0, 2**32 - 1)

            pil_image = _bytes_to_pil(image_bytes)
            logger.info(
                "Processing tile '%s' (%dx%d) seed=%d",
                tile_id,
                pil_image.width,
                pil_image.height,
                tile_seed,
            )

            output_image = _run_sdxl(
                pipe=self._pipe,
                compel=self._compel,
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

            result_bytes = _pil_to_bytes(output_image)
            results.append({
                "tile_id": tile_id,
                "image_bytes": result_bytes,
                "seed_used": tile_seed,
            })
            logger.info("Tile '%s' processed successfully", tile_id)

        # Clean up LoRAs after all tiles are processed
        if applied_lora_names:
            _remove_loras(self._pipe, applied_lora_names)

        return results

    @modal.method()
    def upscale_regions(
        self,
        source_image_bytes: bytes,
        regions: list[dict],
        model_config: dict,
        gen_params: dict,
        global_prompt: str = "",
        negative_prompt: str = "",
        ip_adapter_enabled: bool = False,
        ip_adapter_image_bytes: Optional[bytes] = None,
        ip_adapter_scale: float = 0.6,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> list[dict]:
        """
        Process image regions through the SDXL img2img pipeline.

        Each region is extracted from the source image with optional padding,
        processed individually, and returned with its bounding box info.

        Parameters
        ----------
        source_image_bytes:
            Raw bytes (PNG/JPEG) of the full source image.
        regions:
            List of region dicts, each with:
            - ``"region_id"`` (str)
            - ``"x"``, ``"y"``, ``"w"``, ``"h"`` (int): bounding box in pixels.
            - ``"padding"`` (int): extra pixels to include around the region.
            - ``"prompt"`` (str, optional): per-region prompt suffix.
            - ``"negative_prompt"`` (str, optional): per-region negative prompt suffix.
        model_config:
            Same structure as in ``upscale_tiles``.
        gen_params:
            Same structure as in ``upscale_tiles``.
        global_prompt:
            Positive prompt applied to all regions.
        negative_prompt:
            Negative prompt applied to all regions.
        ip_adapter_enabled:
            Whether to use IP-Adapter style conditioning.
        ip_adapter_image_bytes:
            Raw bytes of the style reference image.
        ip_adapter_scale:
            IP-Adapter influence scale.
        target_width, target_height:
            Optional explicit output dimensions for each region.

        Returns
        -------
        list[dict]
            One dict per input region:
            - ``"region_id"`` (str)
            - ``"image_bytes"`` (bytes): PNG bytes of the processed region.
            - ``"seed_used"`` (int)
            - ``"original_bbox"`` (dict): ``{x, y, w, h}``
            - ``"padded_bbox"`` (dict): ``{x, y, w, h}``

        Raises
        ------
        ValueError
            If ``source_image_bytes`` or ``regions`` is empty.
        """
        if not source_image_bytes:
            raise ValueError("'source_image_bytes' must not be empty")
        if not regions:
            raise ValueError("'regions' list must not be empty")

        # Parse model config
        loras: list[dict] = model_config.get("loras", [])
        controlnet_cfg: dict = model_config.get("controlnet", {})
        controlnet_enabled: bool = bool(controlnet_cfg.get("enabled", False))
        controlnet_conditioning_scale: float = float(
            controlnet_cfg.get("conditioning_scale", 0.6)
        )

        # Parse generation params
        steps: int = int(gen_params.get("steps", 30))
        cfg_scale: float = float(gen_params.get("cfg_scale", 7.0))
        strength: float = float(gen_params.get("denoising_strength", 0.35))
        base_seed: Optional[int] = gen_params.get("seed")

        if target_width is not None:
            target_width = int(target_width)
        if target_height is not None:
            target_height = int(target_height)

        logger.info(
            "upscale_regions: regions=%d steps=%d strength=%.2f controlnet=%s "
            "ip_adapter=%s target=%s",
            len(regions),
            steps,
            strength,
            controlnet_enabled,
            ip_adapter_enabled,
            f"{target_width}x{target_height}" if target_width and target_height else "from_image",
        )

        # Inject trigger words for active LoRAs into the global prompt
        global_prompt = _inject_trigger_words(global_prompt, loras)

        # Load source image
        source_image = _bytes_to_pil(source_image_bytes)
        logger.info("Source image loaded: %dx%d", source_image.width, source_image.height)

        # Apply LoRAs
        lora_volume.reload()
        applied_lora_names: list[str] = []
        if loras:
            applied_lora_names = _apply_loras(self._pipe, loras, LORAS_DIR)

        # IP-Adapter
        ip_adapter_pil: Optional[Any] = None
        if ip_adapter_enabled and ip_adapter_image_bytes:
            self._load_ip_adapter(scale=ip_adapter_scale)
            if self._ip_adapter_loaded:
                raw_ip_img = _bytes_to_pil(ip_adapter_image_bytes)
                ip_adapter_pil = raw_ip_img.resize((224, 224))
                logger.info("IP-Adapter style image prepared (224×224)")
            else:
                logger.warning(
                    "IP-Adapter requested but failed to load — proceeding without it"
                )
        else:
            self._unload_ip_adapter()

        # Process regions
        results: list[dict] = []

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

            padded_bbox = {"x": padded_x, "y": padded_y, "w": padded_w, "h": padded_h}
            original_bbox = {"x": x, "y": y, "w": w, "h": h}

            # Extract region from source image
            region_image = source_image.crop(
                (padded_x, padded_y, padded_x + padded_w, padded_y + padded_h)
            )
            logger.info(
                "Extracted region '%s': original=(%d,%d,%d,%d) padded=(%d,%d,%d,%d) → %dx%d",
                region_id,
                x, y, w, h,
                padded_x, padded_y, padded_w, padded_h,
                region_image.width, region_image.height,
            )

            # Compose prompt
            if region_prompt:
                prompt = f"{global_prompt}, {region_prompt}" if global_prompt else region_prompt
            else:
                prompt = global_prompt

            # Compose negative prompt
            if region_negative_prompt:
                neg_prompt = (
                    f"{negative_prompt}, {region_negative_prompt}"
                    if negative_prompt
                    else region_negative_prompt
                )
            else:
                neg_prompt = negative_prompt

            region_seed: int = base_seed if base_seed is not None else random.randint(0, 2**32 - 1)

            logger.info(
                "Processing region '%s' (%dx%d) seed=%d prompt='%s'",
                region_id,
                region_image.width,
                region_image.height,
                region_seed,
                (prompt[:50] + "...") if len(prompt) > 50 else prompt,
            )

            output_image = _run_sdxl(
                pipe=self._pipe,
                compel=self._compel,
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

            result_bytes = _pil_to_bytes(output_image)
            results.append({
                "region_id": region_id,
                "image_bytes": result_bytes,
                "seed_used": region_seed,
                "original_bbox": original_bbox,
                "padded_bbox": padded_bbox,
            })
            logger.info("Region '%s' processed successfully", region_id)

        # Clean up LoRAs after all regions are processed
        if applied_lora_names:
            _remove_loras(self._pipe, applied_lora_names)

        return results

    # ------------------------------------------------------------------
    # Public Modal methods — model management
    # ------------------------------------------------------------------

    @modal.method()
    def list_models(self) -> list[dict]:
        """
        List LoRA files available on the volume.

        Returns
        -------
        list[dict]
            Each dict has keys ``"name"``, ``"path"``, ``"size_mb"``.
        """
        lora_volume.reload()
        os.makedirs(LORAS_DIR, exist_ok=True)

        logger.info("Listing LoRAs in '%s'", LORAS_DIR)

        models: list[dict] = []
        try:
            for entry in sorted(os.scandir(LORAS_DIR), key=lambda e: e.name):
                if not entry.is_file():
                    continue
                size_bytes = entry.stat().st_size
                size_mb = round(size_bytes / (1024 * 1024), 2)
                models.append({
                    "name": entry.name,
                    "path": entry.path,
                    "size_mb": size_mb,
                })
        except FileNotFoundError:
            logger.warning("LoRA directory '%s' does not exist yet", LORAS_DIR)

        logger.info("Found %d LoRA(s)", len(models))
        return models

    @modal.method()
    def upload_model(
        self,
        filename: str,
        file_data: bytes,
        append: bool = False,
    ) -> dict:
        """
        Save a LoRA file to the volume.

        Parameters
        ----------
        filename:
            Target filename (e.g. ``"my-lora.safetensors"``).
            Must not contain path separators.
        file_data:
            Raw bytes of the file (or a chunk of it).
        append:
            If ``True``, append to an existing file (for chunked uploads).
            If ``False`` (default), overwrite any existing file.

        Returns
        -------
        dict
            Keys: ``"path"`` (str), ``"size_mb"`` (float).

        Raises
        ------
        ValueError
            If ``filename`` is invalid or the extension is not allowed.
        """
        if not filename:
            raise ValueError("'filename' is required for upload_model")
        if not file_data:
            raise ValueError("'file_data' must not be empty")

        # Sanitise filename — no path separators allowed
        safe_filename = os.path.basename(filename)
        if not safe_filename or safe_filename != filename:
            raise ValueError(
                f"Invalid filename '{filename}' — must not contain path separators"
            )

        # Whitelist allowed file extensions
        ALLOWED_EXTENSIONS = {".safetensors", ".pt", ".ckpt", ".bin"}
        ext = os.path.splitext(safe_filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"File extension '{ext}' not allowed. Allowed: {ALLOWED_EXTENSIONS}"
            )

        os.makedirs(LORAS_DIR, exist_ok=True)
        dest_path = os.path.join(LORAS_DIR, safe_filename)

        write_mode = "ab" if append else "wb"
        logger.info(
            "Uploading '%s' (%d bytes, append=%s) → '%s'",
            safe_filename,
            len(file_data),
            append,
            dest_path,
        )

        with open(dest_path, write_mode) as fh:
            fh.write(file_data)

        size_mb = round(os.path.getsize(dest_path) / (1024 * 1024), 2)
        lora_volume.commit()
        logger.info("Upload complete: '%s' (%.2f MB)", safe_filename, size_mb)

        return {"path": dest_path, "size_mb": size_mb}

    @modal.method()
    def delete_model(self, filename: str) -> dict:
        """
        Delete a LoRA file from the volume.

        Parameters
        ----------
        filename:
            Filename of the LoRA to delete (e.g. ``"my-lora.safetensors"``).

        Returns
        -------
        dict
            Keys: ``"status"`` (``"deleted"``), ``"path"`` (str).

        Raises
        ------
        ValueError
            If ``filename`` is invalid or outside the LoRA directory.
        FileNotFoundError
            If the file does not exist.
        """
        if not filename:
            raise ValueError("'filename' is required for delete_model")

        safe_filename = os.path.basename(filename)
        if not safe_filename:
            raise ValueError(f"Invalid filename '{filename}'")

        abs_path = os.path.join(LORAS_DIR, safe_filename)

        # Prevent path traversal
        if not os.path.normpath(abs_path).startswith(
            os.path.normpath(LORAS_DIR) + os.sep
        ):
            raise ValueError(
                f"Path '{abs_path}' is outside the allowed LoRA directory '{LORAS_DIR}'"
            )

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"LoRA not found: '{abs_path}'")

        logger.info("Deleting '%s'", abs_path)

        if os.path.isfile(abs_path):
            os.remove(abs_path)
        else:
            shutil.rmtree(abs_path)

        lora_volume.commit()
        logger.info("Deleted '%s'", abs_path)

        return {"status": "deleted", "path": abs_path}

    @modal.method()
    def health(self) -> dict:
        """
        Return GPU and storage health information.

        Returns
        -------
        dict
            Keys: ``status``, ``gpu``, ``vram_total_gb``, ``vram_used_gb``,
            ``model_loaded``, ``ip_adapter_loaded``, ``loras_dir``,
            ``loras_storage_free_gb``.
        """
        import torch

        gpu_name = "unknown"
        vram_total_gb = 0.0
        vram_used_gb = 0.0

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_total_gb = round(props.total_memory / (1024 ** 3), 2)
            free_bytes, _ = torch.cuda.mem_get_info(0)
            vram_used_gb = round((props.total_memory - free_bytes) / (1024 ** 3), 2)

        loras_available_gb = 0.0
        if os.path.exists(LORAS_DIR):
            stat = shutil.disk_usage(LORAS_DIR)
            loras_available_gb = round(stat.free / (1024 ** 3), 2)

        return {
            "status": "ok",
            "gpu": gpu_name,
            "vram_total_gb": vram_total_gb,
            "vram_used_gb": vram_used_gb,
            "model_loaded": hasattr(self, "_pipe") and self._pipe is not None,
            "ip_adapter_loaded": getattr(self, "_ip_adapter_loaded", False),
            "loras_dir": LORAS_DIR,
            "loras_storage_free_gb": loras_available_gb,
        }

    # ------------------------------------------------------------------
    # IP-Adapter helpers (private)
    # ------------------------------------------------------------------

    def _load_ip_adapter(self, scale: float = 0.6) -> None:
        """Activate IP-Adapter at the requested scale.

        The IP-Adapter weights are loaded once at container startup in
        ``load_models()``.  This method only updates the scale (and, as a
        safety net, loads the adapter if the startup load failed).
        """
        if self._ip_adapter_loaded:
            logger.info("IP-Adapter already loaded — updating scale to %.2f", scale)
            try:
                self._pipe.set_ip_adapter_scale(scale)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Could not update IP-Adapter scale: %s", exc)
            return

        # Fallback: startup load failed — attempt to load now.
        logger.info(
            "IP-Adapter not loaded at startup — attempting lazy load "
            "(scale=%.2f) from '%s' with CLIP encoder '%s'",
            scale,
            IP_ADAPTER_PATH,
            CLIP_VIT_H_PATH,
        )
        try:
            self._pipe.load_ip_adapter(
                os.path.dirname(IP_ADAPTER_PATH),
                subfolder="",
                weight_name=os.path.basename(IP_ADAPTER_PATH),
                image_encoder_folder=CLIP_VIT_H_PATH,
            )
            # Ensure the image encoder is on CUDA to avoid device-mismatch errors.
            if hasattr(self._pipe, "image_encoder") and self._pipe.image_encoder is not None:
                self._pipe.image_encoder.to("cuda")
                logger.info("IP-Adapter image_encoder moved to CUDA (lazy load)")
            self._pipe.set_ip_adapter_scale(scale)
            self._ip_adapter_loaded = True
            logger.info("IP-Adapter loaded successfully (scale=%.2f)", scale)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to load IP-Adapter — continuing without it: %s", exc)

    def _unload_ip_adapter(self) -> None:
        """Disable IP-Adapter by setting its scale to 0.

        The weights remain loaded in memory so they are instantly available for
        the next request that needs them.  This avoids the overhead of
        unloading/reloading on every call and keeps ``_ip_adapter_loaded``
        consistent with the startup-load contract.
        """
        if not self._ip_adapter_loaded:
            return

        logger.info("Disabling IP-Adapter (setting scale to 0.0)")
        try:
            self._pipe.set_ip_adapter_scale(0.0)
            logger.info("IP-Adapter scale set to 0.0 (weights remain loaded)")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Could not set IP-Adapter scale to 0: %s", exc)


# ---------------------------------------------------------------------------
# Module-level helpers (no class dependency)
# ---------------------------------------------------------------------------


def _bytes_to_pil(image_bytes: bytes):
    """Decode raw image bytes (PNG/JPEG) to a PIL Image (RGB)."""
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    logger.debug("Decoded image bytes: %dx%d", image.width, image.height)
    return image


def _pil_to_bytes(image, fmt: str = "PNG") -> bytes:
    """Encode a PIL Image to raw bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    result = buffer.getvalue()
    logger.debug("Encoded PIL image to bytes (%d bytes)", len(result))
    return result


def _inject_trigger_words(global_prompt: str, loras: list[dict]) -> str:
    """Inject LoRA trigger words into the global prompt if not already present."""
    injected: list[str] = []
    for lora in loras:
        lora_name = lora.get("name", "")
        lora_info = HARDCODED_LORAS.get(lora_name, {})
        for trigger in lora_info.get("trigger_words", []):
            if trigger and trigger not in global_prompt and trigger not in injected:
                injected.append(trigger)
    if injected:
        trigger_text = ", ".join(injected)
        global_prompt = f"{trigger_text}, {global_prompt}" if global_prompt else trigger_text
        logger.info("Injected trigger words into prompt: %s", injected)
    return global_prompt


def _apply_loras(pipe: Any, loras: list[dict], loras_dir: str) -> list[str]:
    """Apply multiple LoRAs simultaneously using diffusers multi-adapter support.

    Parameters
    ----------
    pipe:
        The loaded diffusers pipeline.
    loras:
        List of dicts with keys ``"name"`` (str) and ``"weight"`` (float).
    loras_dir:
        Directory where LoRA ``.safetensors`` files are stored.

    Returns
    -------
    list[str]
        Adapter names that were successfully loaded (pass to :func:`_remove_loras`).
    """
    adapter_names: list[str] = []
    adapter_weights: list[float] = []

    for lora in loras:
        name: str = lora.get("name", "")
        weight: float = float(lora.get("weight", 1.0))
        if not name:
            continue

        # Resolve path — check HARDCODED_LORAS first, then raw filename
        if name in HARDCODED_LORAS:
            info = HARDCODED_LORAS[name]
            lora_path = os.path.join(loras_dir, info["filename"])
        else:
            safe_name = os.path.basename(name)
            lora_path = os.path.join(loras_dir, safe_name)
            if not os.path.isfile(lora_path):
                lora_path = os.path.join(loras_dir, f"{safe_name}.safetensors")

        if not os.path.isfile(lora_path):
            logger.warning("Skipping LoRA '%s' — file not found at '%s'", name, lora_path)
            continue

        logger.info("Loading LoRA '%s' (weight=%.2f) from '%s'", name, weight, lora_path)
        try:
            pipe.load_lora_weights(lora_path, adapter_name=name)
            adapter_names.append(name)
            adapter_weights.append(weight)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to load LoRA '%s': %s", name, exc)

    if adapter_names:
        logger.info("Activating adapters: %s with weights %s", adapter_names, adapter_weights)
        pipe.set_adapters(adapter_names, adapter_weights)

    return adapter_names


def _remove_loras(pipe: Any, adapter_names: list[str]) -> None:
    """Remove all applied LoRAs from the pipeline."""
    if not adapter_names:
        return
    logger.info("Unloading LoRA adapters: %s", adapter_names)
    try:
        pipe.unload_lora_weights()
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Could not unload LoRA weights: %s", exc)


def _build_compel(pipe: Any) -> Optional[Any]:
    """Build a Compel instance for long-prompt encoding with SDXL dual encoders.

    Returns ``None`` if compel is not installed or the pipeline lacks the
    required tokenizer/encoder attributes.
    """
    try:
        from compel import Compel, ReturnedEmbeddingsType

        if not (
            hasattr(pipe, "tokenizer")
            and hasattr(pipe, "tokenizer_2")
            and hasattr(pipe, "text_encoder")
            and hasattr(pipe, "text_encoder_2")
        ):
            logger.warning(
                "Pipeline is missing tokenizer/text_encoder attributes — compel disabled"
            )
            return None

        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )
        logger.info("Compel long-prompt encoder initialised")
        return compel
    except ImportError:
        logger.warning(
            "compel is not installed — prompts will be truncated at 77 tokens"
        )
        return None
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to initialise compel: %s — falling back to raw prompts", exc)
        return None


def _encode_prompt_with_compel(
    compel: Any,
    prompt: str,
    negative_prompt: str,
) -> tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    """Encode *prompt* and *negative_prompt* using compel.

    Returns
    -------
    tuple of (conditioning, pooled, neg_conditioning, neg_pooled)
        All four tensors ready to pass to the pipeline.
        Returns ``(None, None, None, None)`` on failure.
    """
    try:
        conditioning, pooled = compel(prompt)
        neg_conditioning, neg_pooled = compel(negative_prompt)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length(
            [conditioning, neg_conditioning]
        )
        return conditioning, pooled, neg_conditioning, neg_pooled
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "compel encoding failed: %s — falling back to raw prompt strings", exc
        )
        return None, None, None, None


def _run_sdxl(
    pipe: Any,
    compel: Optional[Any],
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
    Run the SDXL ControlNet img2img pipeline for a single image.

    Parameters
    ----------
    pipe:
        The loaded ``StableDiffusionXLControlNetImg2ImgPipeline``.
    compel:
        Optional Compel instance for long-prompt encoding.
    image:
        Input PIL image (tile or region to upscale).
    prompt / negative_prompt:
        Text prompts.
    strength:
        Denoising strength (0.0–1.0).
    steps:
        Number of inference steps.
    cfg_scale:
        Classifier-free guidance scale.
    seed:
        Random seed for reproducibility.
    controlnet_enabled:
        Whether to pass ControlNet conditioning.
    controlnet_conditioning_scale:
        ControlNet influence scale.
    ip_adapter_image:
        Optional PIL Image for IP-Adapter style conditioning.
    target_width, target_height:
        Optional explicit output dimensions.

    Returns
    -------
    PIL.Image.Image
        The generated output image.
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

    # Pass explicit width/height when provided
    if target_width is not None:
        kwargs["width"] = target_width
    if target_height is not None:
        kwargs["height"] = target_height

    # ControlNet conditioning — always pass control_image (required by the
    # ControlNet pipeline); set scale to 0 when ControlNet is disabled so
    # the pipeline still runs without errors.
    kwargs["control_image"] = image
    kwargs["controlnet_conditioning_scale"] = (
        controlnet_conditioning_scale if controlnet_enabled else 0.0
    )

    if ip_adapter_image is not None:
        kwargs["ip_adapter_image"] = ip_adapter_image

    logger.info(
        "SDXL generate: steps=%d strength=%.2f cfg=%.1f seed=%d controlnet=%s "
        "(scale=%.2f) ip_adapter=%s target=%s compel=%s prompt_len=%d",
        steps,
        strength,
        cfg_scale,
        seed,
        controlnet_enabled,
        controlnet_conditioning_scale if controlnet_enabled else 0.0,
        ip_adapter_image is not None,
        f"{target_width}x{target_height}" if target_width and target_height else "from_image",
        use_compel,
        len(prompt),
    )

    result = pipe(**kwargs)
    output_image = result.images[0]
    logger.info("SDXL generation complete")
    return output_image


def _ensure_loras_downloaded(loras_dir: str, volume: Any) -> None:
    """
    Download the hardcoded CivitAI LoRAs into *loras_dir* if not already present.

    Reads ``CIVITAI_API_TOKEN`` from the environment.  If the token is missing
    or a download fails the function logs a warning and continues rather than
    raising an exception.

    Uses the CivitAI v1 models API to resolve the latest download URL for each
    model, then streams the file to disk.

    Parameters
    ----------
    loras_dir:
        Local path where LoRA files should be stored (e.g. ``/vol/loras``).
    volume:
        The ``modal.Volume`` instance to commit after downloading new files.
    """
    import requests

    token = os.environ.get("CIVITAI_API_TOKEN", "")
    if not token:
        logger.warning(
            "CIVITAI_API_TOKEN is not set — hardcoded LoRAs will not be downloaded"
        )

    os.makedirs(loras_dir, exist_ok=True)

    for ui_name, info in HARDCODED_LORAS.items():
        dest_path = os.path.join(loras_dir, info["filename"])
        if os.path.isfile(dest_path):
            logger.info(
                "LoRA '%s' already exists at '%s' — skipping download",
                ui_name,
                dest_path,
            )
            continue

        if not token:
            logger.warning(
                "Skipping download of LoRA '%s' (model_id=%s) — no CivitAI API token",
                ui_name,
                info["civitai_model_id"],
            )
            continue

        # Resolve download URL via CivitAI API
        model_id = info["civitai_model_id"]
        api_url = f"https://civitai.com/api/v1/models/{model_id}"
        logger.info(
            "Fetching CivitAI model info for '%s' (model_id=%s) from %s",
            ui_name,
            model_id,
            api_url,
        )

        try:
            api_resp = requests.get(
                api_url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
            api_resp.raise_for_status()
            model_data = api_resp.json()

            # Get the first model version's first file download URL
            model_versions = model_data.get("modelVersions", [])
            if not model_versions:
                logger.warning(
                    "No model versions found for LoRA '%s' (model_id=%s)",
                    ui_name,
                    model_id,
                )
                continue

            latest_version = model_versions[0]
            files = latest_version.get("files", [])
            if not files:
                logger.warning(
                    "No files found for LoRA '%s' (model_id=%s, version_id=%s)",
                    ui_name,
                    model_id,
                    latest_version.get("id"),
                )
                continue

            # Prefer the primary file (type == "Model") if available
            primary_file = next(
                (f for f in files if f.get("type") == "Model"),
                files[0],
            )
            download_url = primary_file.get("downloadUrl", "")
            if not download_url:
                logger.warning(
                    "No downloadUrl for LoRA '%s' (model_id=%s)",
                    ui_name,
                    model_id,
                )
                continue

            # Append token to download URL
            sep = "&" if "?" in download_url else "?"
            download_url_with_token = f"{download_url}{sep}token={token}"

            logger.info(
                "Downloading LoRA '%s' from CivitAI → '%s'",
                ui_name,
                dest_path,
            )
            dl_resp = requests.get(
                download_url_with_token,
                stream=True,
                allow_redirects=True,
                timeout=300,
            )
            dl_resp.raise_for_status()

            with open(dest_path, "wb") as fh:
                for chunk in dl_resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)

            file_size = os.path.getsize(dest_path)
            logger.info(
                "LoRA '%s' downloaded successfully (%d bytes)",
                ui_name,
                file_size,
            )

        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Failed to download LoRA '%s' (model_id=%s): %s",
                ui_name,
                model_id,
                exc,
            )
            # Remove partial file if it exists
            if os.path.isfile(dest_path):
                try:
                    os.remove(dest_path)
                except OSError:
                    pass

    # Commit any newly downloaded files to the volume
    try:
        volume.commit()
        logger.info("LoRA volume committed after startup downloads")
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Could not commit LoRA volume: %s", exc)
