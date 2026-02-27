"""
Upscale service — Illustrious-XL + ControlNet Tile on a Modal A10G GPU container.

This module is part of the unified Modal app (``modal.App("ai-toolbox")``).
It does NOT define its own ``modal.App``; instead it imports ``app``,
``upscale_image``, ``models_volume``, and ``lora_volume`` from
``src.app_config`` and registers ``UpscaleService`` against that shared app
via ``@app.cls()``.

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
from __future__ import annotations

import io
import logging
import os
import random
import shutil
from typing import Any, Optional

import modal

from src.app_config import app, upscale_image, lora_volume, models_volume
from src.services.model_metadata import ModelMetadataManager, ModelInfo
from src.services.model_registry import (
    LORAS_MOUNT_PATH,
    MODELS_MOUNT_PATH,
    check_model_files_exist,
    get_model_file_path,
    get_model_path,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — model paths resolved from the models volume at runtime
# ---------------------------------------------------------------------------

# LoRA volume mount point (must match the volumes= dict in @app.cls)
LORAS_DIR = LORAS_MOUNT_PATH          # /vol/loras  — base volume path
LORAS_SUBDIR = os.path.join(LORAS_MOUNT_PATH, "loras")  # /vol/loras/loras — new subdirectory

# Model type subdirectories within the lora volume
CHECKPOINTS_DIR = os.path.join(LORAS_MOUNT_PATH, "checkpoints")  # /vol/loras/checkpoints
EMBEDDINGS_DIR = os.path.join(LORAS_MOUNT_PATH, "embeddings")    # /vol/loras/embeddings
VAES_DIR = os.path.join(LORAS_MOUNT_PATH, "vae")                 # /vol/loras/vae

# ---------------------------------------------------------------------------
# UpscaleService
# ---------------------------------------------------------------------------


@app.cls(
    # A10G has 24 GB VRAM — enough for SDXL + ControlNet + IP-Adapter with CPU offload.
    gpu="A10G",
    image=upscale_image,
    # Keep the container alive for 5 minutes after the last request.
    scaledown_window=300,
    # 40 minutes — handles up to ~35 tiles with safety margin
    timeout=2400,
    # Mount the models volume at /vol/models/ and the LoRA volume at /vol/loras/
    volumes={
        MODELS_MOUNT_PATH: models_volume,
        LORAS_DIR: lora_volume,
    },
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

        # Reload the volume to pick up any newly downloaded model weights.
        models_volume.reload()

        # ------------------------------------------------------------------
        # Guard: abort gracefully if required models are not yet downloaded.
        # ------------------------------------------------------------------
        required_models = [
            "illustrious-xl",
            "controlnet-tile",
            "sdxl-vae",
            "ip-adapter",
            "clip-vit-h",
        ]
        missing = [k for k in required_models if not check_model_files_exist(k)]
        if missing:
            logger.warning(
                "Missing models: %s — run setup wizard first", missing
            )
            self._models_ready = False
            return

        self._models_ready = True

        # Resolve model paths from the volume.
        illustrious_xl_dir = get_model_path("illustrious-xl")
        controlnet_tile_path = get_model_path("controlnet-tile")
        sdxl_vae_path = get_model_path("sdxl-vae")
        ip_adapter_dir = get_model_path("ip-adapter")
        clip_vit_h_path = get_model_path("clip-vit-h")

        # The Illustrious-XL model is a single safetensors file downloaded via
        # hf_hub_download into the local_dir.  The registry records the filename
        # as "Illustrious-XL-v2.0.safetensors".
        illustrious_xl_path = get_model_file_path("illustrious-xl")
        if illustrious_xl_path is None:
            illustrious_xl_path = os.path.join(
                illustrious_xl_dir, "Illustrious-XL-v2.0.safetensors"
            )

        # The IP-Adapter weight file may be in the flattened location or still
        # in the sdxl_models subfolder (depending on download version).
        ip_adapter_path = get_model_file_path("ip-adapter")
        if ip_adapter_path is None:
            ip_adapter_path = os.path.join(
                ip_adapter_dir, "ip-adapter-plus_sdxl_vit-h.safetensors"
            )

        # ------------------------------------------------------------------
        # 1. Load VAE fp16-fix
        # ------------------------------------------------------------------
        logger.info("Loading SDXL VAE fp16-fix from '%s'", sdxl_vae_path)
        vae = AutoencoderKL.from_pretrained(
            sdxl_vae_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        logger.info("VAE loaded")

        # ------------------------------------------------------------------
        # 2. Load ControlNet Tile
        # ------------------------------------------------------------------
        logger.info("Loading ControlNet Tile from '%s'", controlnet_tile_path)
        controlnet = ControlNetModel.from_pretrained(
            controlnet_tile_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        logger.info("ControlNet loaded")

        # ------------------------------------------------------------------
        # 3. Load SDXL pipeline from single-file checkpoint, then wrap with
        #    ControlNet using from_pipe() (avoids double-loading UNet weights)
        # ------------------------------------------------------------------
        logger.info("Loading SDXL base pipeline from '%s'", illustrious_xl_path)
        base_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            illustrious_xl_path,
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

        # Move text encoders to CUDA for Compel (CPU offload keeps them on CPU)
        self._pipe.text_encoder.to("cuda")
        self._pipe.text_encoder_2.to("cuda")

        # ------------------------------------------------------------------
        # 6. Build Compel for long-prompt encoding
        # ------------------------------------------------------------------
        self._compel = _build_compel(self._pipe)

        # ------------------------------------------------------------------
        # 7. IP-Adapter will be lazy-loaded when first needed
        # ------------------------------------------------------------------
        # IP-Adapter is only loaded when ip_adapter_enabled=True is passed to
        # the generation method. This saves memory and startup time when
        # IP-Adapter is not used.
        self._ip_adapter_loaded: bool = False
        logger.info("IP-Adapter will be lazy-loaded when first needed (not loaded at startup)")

        logger.info("=== UpscaleService: all models loaded ===")

    # ------------------------------------------------------------------
    # Checkpoint / VAE / Embedding management (private)
    # ------------------------------------------------------------------

    # State tracking for currently loaded checkpoint
    _current_checkpoint: Optional[str] = None      # Name of loaded checkpoint (None = default)
    _current_checkpoint_path: Optional[str] = None  # Path to checkpoint file
    _current_vae: Optional[str] = None              # Name of loaded VAE (None = default)
    _has_embedded_vae: bool = False                 # Whether current checkpoint has embedded VAE
    _loaded_embeddings: list[str] = []              # List of loaded embedding names

    def _ensure_checkpoint(self, checkpoint_name: Optional[str] = None) -> None:
        """
        Ensure the correct checkpoint is loaded.

        Parameters
        ----------
        checkpoint_name: Optional[str]
            Name of checkpoint to load, or None for default Illustrious-XL.

        Notes
        -----
        Checkpoint switching takes ~15-20 seconds. The loaded checkpoint is
        cached to avoid reloading on subsequent calls with the same checkpoint.
        """
        # If no checkpoint specified, use default
        if checkpoint_name is None:
            if self._current_checkpoint is not None:
                # Already loaded a custom checkpoint, need to reload default
                self._load_default_checkpoint()
            return

        # Check if already loaded
        if self._current_checkpoint == checkpoint_name:
            logger.info("Checkpoint '%s' already loaded", checkpoint_name)
            return

        # Load the custom checkpoint
        self._load_custom_checkpoint(checkpoint_name)

    def _load_default_checkpoint(self) -> None:
        """Load the default SDXL checkpoint (Illustrious-XL)."""
        import torch
        from diffusers import (
            StableDiffusionXLImg2ImgPipeline,
            StableDiffusionXLControlNetImg2ImgPipeline,
            DPMSolverMultistepScheduler,
        )

        logger.info("Reloading default checkpoint (Illustrious-XL)")

        # Get the default checkpoint path
        illustrious_xl_path = get_model_file_path("illustrious-xl")
        if illustrious_xl_path is None:
            illustrious_xl_dir = get_model_path("illustrious-xl")
            illustrious_xl_path = os.path.join(
                illustrious_xl_dir, "Illustrious-XL-v2.0.safetensors"
            )

        # Get VAE path
        sdxl_vae_path = get_model_path("sdxl-vae")

        # Load VAE
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            sdxl_vae_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        )

        # Load base pipeline
        base_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            illustrious_xl_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            vae=vae,
        )

        # Get ControlNet from current pipeline
        controlnet = self._pipe.controlnet

        # Wrap with ControlNet
        self._pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pipe(
            base_pipe,
            controlnet=controlnet,
        )

        # Restore scheduler
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )

        # Re-enable CPU offload
        self._pipe.enable_model_cpu_offload()

        # Move text encoders to CUDA for Compel
        self._pipe.text_encoder.to("cuda")
        self._pipe.text_encoder_2.to("cuda")

        # Rebuild Compel
        self._compel = _build_compel(self._pipe)

        # Reload IP-Adapter if it was loaded
        if self._ip_adapter_loaded:
            self._reload_ip_adapter()

        # Update tracking
        self._current_checkpoint = None
        self._current_checkpoint_path = None
        self._current_vae = None
        self._has_embedded_vae = False
        self._loaded_embeddings = []

        logger.info("Default checkpoint loaded successfully")

    def _load_custom_checkpoint(self, checkpoint_name: str) -> None:
        """
        Load a custom checkpoint from the checkpoints directory.

        Parameters
        ----------
        checkpoint_name: str
            Filename of the checkpoint to load (with or without .safetensors extension).

        Raises
        ------
        FileNotFoundError
            If the checkpoint file is not found.
        """
        import torch
        from diffusers import (
            StableDiffusionXLImg2ImgPipeline,
            StableDiffusionXLControlNetImg2ImgPipeline,
            DPMSolverMultistepScheduler,
            AutoencoderKL,
        )

        logger.info("Loading custom checkpoint: %s", checkpoint_name)

        # Find checkpoint file
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint_name)

        if not os.path.exists(checkpoint_path):
            # Try with .safetensors extension
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"{checkpoint_name}.safetensors")

        if not os.path.exists(checkpoint_path):
            # Try without any extension (in case user provided extension)
            base_name = os.path.splitext(checkpoint_name)[0]
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"{base_name}.safetensors")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")

        # Check if checkpoint has embedded VAE
        has_embedded_vae = False
        mgr = ModelMetadataManager(LORAS_MOUNT_PATH)
        model_info = mgr.get_model(checkpoint_name)
        if model_info and model_info.embedded_vae:
            has_embedded_vae = True
            logger.info(
                "Checkpoint '%s' has embedded VAE: %s",
                checkpoint_name,
                model_info.embedded_vae,
            )

        # Load VAE (use default fp16-fixed VAE for custom checkpoints without embedded VAE)
        vae = None
        if not has_embedded_vae:
            sdxl_vae_path = get_model_path("sdxl-vae")
            vae = AutoencoderKL.from_pretrained(
                sdxl_vae_path,
                torch_dtype=torch.float16,
                local_files_only=True,
            )
        else:
            logger.info("Using checkpoint's embedded VAE instead of default VAE")

        # Load base pipeline from checkpoint
        base_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            checkpoint_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            vae=vae,  # None means use embedded VAE if present
        )

        # Get ControlNet from current pipeline
        controlnet = self._pipe.controlnet

        # Wrap with ControlNet
        self._pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pipe(
            base_pipe,
            controlnet=controlnet,
        )

        # Ensure vae is not None in the new pipeline (prevents AttributeError in prepare_latents)
        if self._pipe.vae is None:
            if base_pipe.vae is not None:
                self._pipe.vae = base_pipe.vae
            else:
                # If both pipelines have None vae, load the default VAE
                logger.warning("Both pipelines have None VAE, loading default SDXL VAE")
                sdxl_vae_path = get_model_path("sdxl-vae")
                # Get device and dtype from existing pipeline components
                device = next(base_pipe.text_encoder.parameters()).device
                self._pipe.vae = AutoencoderKL.from_pretrained(
                    sdxl_vae_path,
                    torch_dtype=torch.float16,
                    local_files_only=True,
                ).to(device=device, dtype=torch.float16)

        # Set scheduler
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )

        # Ensure entire pipeline is in float16
        self._pipe.to(dtype=torch.float16)
        if hasattr(self._pipe, 'controlnet') and self._pipe.controlnet is not None:
            self._pipe.controlnet.to(dtype=torch.float16)
        
        # Disable VAE force_upcast to prevent dtype mismatches
        if hasattr(self._pipe.vae.config, "force_upcast"):
            self._pipe.vae.config.force_upcast = False

        # Enable CPU offload
        self._pipe.enable_model_cpu_offload()

        # Move text encoders to CUDA for Compel
        self._pipe.text_encoder.to("cuda")
        self._pipe.text_encoder_2.to("cuda")

        # Rebuild Compel
        self._compel = _build_compel(self._pipe)

        # Reload IP-Adapter if it was loaded
        if self._ip_adapter_loaded:
            self._reload_ip_adapter()

        # Update tracking
        self._current_checkpoint = checkpoint_name
        self._current_checkpoint_path = checkpoint_path
        self._current_vae = None
        self._has_embedded_vae = has_embedded_vae
        self._loaded_embeddings = []

        logger.info("Custom checkpoint loaded: %s (embedded_vae=%s)", checkpoint_name, has_embedded_vae)

    def _reload_ip_adapter(self) -> None:
        """Reload IP-Adapter after checkpoint switch."""
        ip_adapter_path = get_model_file_path("ip-adapter")
        if ip_adapter_path is None:
            ip_adapter_dir = get_model_path("ip-adapter")
            ip_adapter_path = os.path.join(
                ip_adapter_dir, "ip-adapter-plus_sdxl_vit-h.safetensors"
            )
        clip_vit_h_path = get_model_path("clip-vit-h")

        try:
            self._pipe.load_ip_adapter(
                os.path.dirname(ip_adapter_path),
                subfolder="",
                weight_name=os.path.basename(ip_adapter_path),
                image_encoder_folder=clip_vit_h_path,
            )
            if hasattr(self._pipe, "image_encoder") and self._pipe.image_encoder is not None:
                self._pipe.image_encoder.to("cuda")
            self._pipe.set_ip_adapter_scale(0.0)
            logger.info("IP-Adapter reloaded after checkpoint switch")
        except Exception as exc:
            logger.warning("Failed to reload IP-Adapter after checkpoint switch: %s", exc)
            self._ip_adapter_loaded = False

    def _load_vae(self, vae_name: Optional[str] = None) -> None:
        """
        Load a custom VAE or use checkpoint's embedded VAE.

        Parameters
        ----------
        vae_name: Optional[str]
            Name of VAE to load, or None for default/embedded VAE.
        """
        from diffusers import AutoencoderKL
        import torch

        if vae_name is None:
            # If checkpoint has embedded VAE, use it (already loaded with checkpoint)
            if self._has_embedded_vae:
                logger.info("Using checkpoint's embedded VAE")
                self._current_vae = None
                return
            # Use default VAE (already in pipeline)
            if self._current_vae is not None:
                logger.info("Switching back to default VAE")
                self._current_vae = None
            return

        # Check if already loaded
        if self._current_vae == vae_name:
            logger.info("VAE '%s' already loaded", vae_name)
            return

        # Warn if overriding embedded VAE
        if self._has_embedded_vae:
            logger.info("Overriding checkpoint's embedded VAE with custom VAE: %s", vae_name)

        vae_path = os.path.join(VAES_DIR, vae_name)
        if not os.path.exists(vae_path):
            vae_path = os.path.join(VAES_DIR, f"{vae_name}.safetensors")

        if not os.path.exists(vae_path):
            # Try without extension
            base_name = os.path.splitext(vae_name)[0]
            vae_path = os.path.join(VAES_DIR, f"{base_name}.safetensors")

        if not os.path.exists(vae_path):
            logger.warning("VAE not found: %s, using default", vae_name)
            return

        logger.info("Loading VAE: %s", vae_name)

        try:
            vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float16)
            self._pipe.vae = vae
            self._current_vae = vae_name
            logger.info("VAE loaded: %s", vae_name)
        except Exception as exc:
            logger.warning("Failed to load VAE '%s': %s", vae_name, exc)

    def _load_embeddings(self, embedding_names: Optional[list[str]] = None) -> None:
        """
        Load textual inversion embeddings.

        Parameters
        ----------
        embedding_names: Optional[list[str]]
            List of embedding filenames to load.
        """
        if not embedding_names:
            return

        for emb_name in embedding_names:
            if emb_name in self._loaded_embeddings:
                logger.info("Embedding '%s' already loaded", emb_name)
                continue

            emb_path = os.path.join(EMBEDDINGS_DIR, emb_name)
            if not os.path.exists(emb_path):
                # Try with common extensions
                for ext in [".pt", ".safetensors", ".bin"]:
                    test_path = os.path.join(EMBEDDINGS_DIR, f"{emb_name}{ext}")
                    if os.path.exists(test_path):
                        emb_path = test_path
                        break

            if not os.path.exists(emb_path):
                # Try without extension
                base_name = os.path.splitext(emb_name)[0]
                for ext in [".pt", ".safetensors", ".bin"]:
                    test_path = os.path.join(EMBEDDINGS_DIR, f"{base_name}{ext}")
                    if os.path.exists(test_path):
                        emb_path = test_path
                        break

            if not os.path.exists(emb_path):
                logger.warning("Embedding not found: %s", emb_name)
                continue

            logger.info("Loading embedding: %s", emb_name)

            try:
                # Load embedding - diffusers handles this automatically
                self._pipe.load_textual_inversion(emb_path)
                self._loaded_embeddings.append(emb_name)
                logger.info("Embedding loaded: %s", emb_name)
            except Exception as exc:
                logger.warning("Failed to load embedding '%s': %s", emb_name, exc)

    def _clear_embeddings(self) -> None:
        """Clear all loaded embeddings."""
        if not self._loaded_embeddings:
            return

        logger.info("Unloading embeddings: %s", self._loaded_embeddings)
        try:
            # Unload all embeddings by name
            for emb_name in self._loaded_embeddings:
                try:
                    # Get the token name used by the embedding
                    # Textual inversion embeddings are stored with their trigger word
                    self._pipe.unload_textual_inversion()
                except Exception as exc:
                    logger.debug("Could not unload embedding '%s': %s", emb_name, exc)

            self._loaded_embeddings = []
            logger.info("All embeddings unloaded")
        except Exception as exc:
            logger.warning("Error unloading embeddings: %s", exc)
            self._loaded_embeddings = []

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
        checkpoint_name: Optional[str] = None,
        vae_name: Optional[str] = None,
        embedding_names: Optional[list[str]] = None,
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
        checkpoint_name:
            Name of custom checkpoint to load, or None for default Illustrious-XL.
        vae_name:
            Name of custom VAE to load, or None for default.
        embedding_names:
            List of textual inversion embedding filenames to load.

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
        RuntimeError
            If models have not been downloaded yet.
        ValueError
            If ``tiles`` is empty.
        """
        if not getattr(self, "_models_ready", False):
            raise RuntimeError(
                "Models not downloaded. Please run the setup wizard first."
            )

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
        clip_skip: int = int(gen_params.get("clip_skip", 0))

        if target_width is not None:
            target_width = int(target_width)
        if target_height is not None:
            target_height = int(target_height)

        logger.info(
            "upscale_tiles: tiles=%d steps=%d strength=%.2f controlnet=%s "
            "ip_adapter=%s target=%s checkpoint=%s vae=%s embeddings=%s",
            len(tiles),
            steps,
            strength,
            controlnet_enabled,
            ip_adapter_enabled,
            f"{target_width}x{target_height}" if target_width and target_height else "from_image",
            checkpoint_name or "default",
            vae_name or "default",
            embedding_names or [],
        )

        # Ensure correct checkpoint is loaded
        self._ensure_checkpoint(checkpoint_name)

        # Load custom VAE if specified
        if vae_name:
            self._load_vae(vae_name)

        # Load embeddings if specified
        if embedding_names:
            self._load_embeddings(embedding_names)

        # Inject trigger words for active LoRAs into the global prompt
        global_prompt = _inject_trigger_words(global_prompt, loras, LORAS_DIR)

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
                ip_adapter_loaded=self._ip_adapter_loaded,
                target_width=target_width,
                target_height=target_height,
                clip_skip=clip_skip,
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
    def process_tile_batch(
        self,
        batch_id: int,
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
        checkpoint_name: Optional[str] = None,
        vae_name: Optional[str] = None,
        embedding_names: Optional[list[str]] = None,
    ) -> dict:
        """
        Process a batch of tiles through the SDXL img2img pipeline.

        This method is designed to be called via .map() for parallel processing
        across multiple GPU containers. Each container processes one batch.

        Parameters
        ----------
        batch_id: int
            Unique identifier for this batch (e.g., 0, 1, 2).
        tiles: list[dict]
            List of tile dicts for this batch only. Each with:
            - ``"tile_id"`` (str): unique identifier, e.g. ``"0_0"``.
            - ``"image_bytes"`` (bytes): raw PNG or JPEG bytes of the tile.
            - ``"prompt_override"`` (str, optional): per-tile prompt suffix.
        model_config: dict
            Same structure as in ``upscale_tiles``.
        gen_params: dict
            Same structure as in ``upscale_tiles``.
        global_prompt: str
            Positive prompt applied to all tiles.
        negative_prompt: str
            Negative prompt applied to all tiles.
        ip_adapter_enabled: bool
            Whether to use IP-Adapter style conditioning.
        ip_adapter_image_bytes: Optional[bytes]
            Raw bytes of the style reference image (PNG/JPEG).
        ip_adapter_scale: float
            IP-Adapter influence scale (0.0–1.0).
        target_width, target_height: Optional[int]
            Optional explicit output dimensions for each tile.
        checkpoint_name: Optional[str]
            Name of custom checkpoint to load, or None for default.
        vae_name: Optional[str]
            Name of custom VAE to load, or None for default.
        embedding_names: Optional[list[str]]
            List of textual inversion embedding filenames to load.

        Returns
        -------
        dict
            {
                "batch_id": int,
                "results": list[dict],  # Same format as upscale_tiles
                "success": bool,
                "error": Optional[str],
                "tiles_processed": int,
            }
        """
        if not getattr(self, "_models_ready", False):
            return {
                "batch_id": batch_id,
                "results": [],
                "success": False,
                "error": "Models not downloaded. Please run the setup wizard first.",
                "tiles_processed": 0,
            }

        if not tiles:
            return {
                "batch_id": batch_id,
                "results": [],
                "success": True,
                "error": None,
                "tiles_processed": 0,
            }

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
        clip_skip: int = int(gen_params.get("clip_skip", 0))

        if target_width is not None:
            target_width = int(target_width)
        if target_height is not None:
            target_height = int(target_height)

        logger.info(
            "process_tile_batch: batch_id=%d tiles=%d steps=%d strength=%.2f controlnet=%s "
            "ip_adapter=%s target=%s clip_skip=%d checkpoint=%s vae=%s embeddings=%s",
            batch_id,
            len(tiles),
            steps,
            strength,
            controlnet_enabled,
            ip_adapter_enabled,
            f"{target_width}x{target_height}" if target_width and target_height else "from_image",
            clip_skip,
            checkpoint_name or "default",
            vae_name or "default",
            embedding_names or [],
        )

        # Ensure correct checkpoint is loaded
        self._ensure_checkpoint(checkpoint_name)

        # Load custom VAE if specified
        if vae_name:
            self._load_vae(vae_name)

        # Load embeddings if specified
        if embedding_names:
            self._load_embeddings(embedding_names)

        # Inject trigger words for active LoRAs into the global prompt
        global_prompt = _inject_trigger_words(global_prompt, loras, LORAS_DIR)

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
        errors: list[str] = []

        for tile_entry in tiles:
            tile_id: str = tile_entry.get("tile_id", "unknown")
            image_bytes: bytes = tile_entry.get("image_bytes", b"")
            prompt_override: Optional[str] = tile_entry.get("prompt_override")

            if not image_bytes:
                logger.warning("Tile '%s' has no image data — skipping", tile_id)
                results.append({"tile_id": tile_id, "image_bytes": b"", "seed_used": None, "error": "No image data"})
                errors.append(f"{tile_id}: no image data")
                continue

            # Compose prompt
            if prompt_override:
                prompt = f"{global_prompt}, {prompt_override}" if global_prompt else prompt_override
            else:
                prompt = global_prompt

            tile_seed: int = base_seed if base_seed is not None else random.randint(0, 2**32 - 1)

            try:
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
                    ip_adapter_loaded=self._ip_adapter_loaded,
                    target_width=target_width,
                    target_height=target_height,
                    clip_skip=clip_skip,
                )

                result_bytes = _pil_to_bytes(output_image)
                results.append({
                    "tile_id": tile_id,
                    "image_bytes": result_bytes,
                    "seed_used": tile_seed,
                })
                logger.info("Tile '%s' processed successfully", tile_id)
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Tile '%s' failed: %s", tile_id, e)
                results.append({
                    "tile_id": tile_id,
                    "image_bytes": b"",
                    "seed_used": None,
                    "error": str(e),
                })
                errors.append(f"{tile_id}: {e}")

        # Clean up LoRAs after all tiles are processed
        if applied_lora_names:
            _remove_loras(self._pipe, applied_lora_names)

        return {
            "batch_id": batch_id,
            "results": results,
            "success": len(errors) == 0,
            "error": "; ".join(errors) if errors else None,
            "tiles_processed": sum(1 for r in results if "error" not in r),
        }

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
        checkpoint_name: Optional[str] = None,
        vae_name: Optional[str] = None,
        embedding_names: Optional[list[str]] = None,
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
        checkpoint_name:
            Name of custom checkpoint to load, or None for default.
        vae_name:
            Name of custom VAE to load, or None for default.
        embedding_names:
            List of textual inversion embedding filenames to load.

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
        RuntimeError
            If models have not been downloaded yet.
        ValueError
            If ``source_image_bytes`` or ``regions`` is empty.
        """
        if not getattr(self, "_models_ready", False):
            raise RuntimeError(
                "Models not downloaded. Please run the setup wizard first."
            )

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
            "ip_adapter=%s target=%s checkpoint=%s vae=%s embeddings=%s",
            len(regions),
            steps,
            strength,
            controlnet_enabled,
            ip_adapter_enabled,
            f"{target_width}x{target_height}" if target_width and target_height else "from_image",
            checkpoint_name or "default",
            vae_name or "default",
            embedding_names or [],
        )

        # Ensure correct checkpoint is loaded
        self._ensure_checkpoint(checkpoint_name)

        # Load custom VAE if specified
        if vae_name:
            self._load_vae(vae_name)

        # Load embeddings if specified
        if embedding_names:
            self._load_embeddings(embedding_names)

        # Inject trigger words for active LoRAs into the global prompt
        global_prompt = _inject_trigger_words(global_prompt, loras, LORAS_DIR)

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
                ip_adapter_loaded=self._ip_adapter_loaded,
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
    def list_models(self) -> dict:
        """List available LoRAs and checkpoints with metadata.

        Returns
        -------
        dict
            ``{"models": list[dict]}`` where each dict has keys
            ``"filename"``, ``"name"``, ``"model_type"``, ``"trigger_words"``,
            ``"base_model"``, ``"default_weight"``, ``"civitai_model_id"``.
        """
        from src.app_config import lora_volume as _lv
        _lv.reload()

        mgr = ModelMetadataManager(LORAS_DIR)
        models = mgr.list_models()

        result: list[dict] = []
        known_files: set[str] = {m.filename for m in models}

        # Models with metadata
        for m in models:
            result.append({
                "filename": m.filename,
                "name": m.name,
                "model_type": m.model_type,
                "trigger_words": m.trigger_words,
                "base_model": m.base_model,
                "default_weight": m.default_weight,
                "civitai_model_id": m.civitai_model_id,
            })

        # Scan for orphan files (no metadata entry)
        for scan_dir in [LORAS_SUBDIR, LORAS_DIR]:
            if not os.path.isdir(scan_dir):
                continue
            for fname in os.listdir(scan_dir):
                if fname.startswith(".") or fname in known_files:
                    continue
                if not fname.endswith((".safetensors", ".ckpt", ".pt")):
                    continue
                if os.path.isfile(os.path.join(scan_dir, fname)):
                    result.append({
                        "filename": fname,
                        "name": fname.rsplit(".", 1)[0],
                        "model_type": "lora",
                        "trigger_words": [],
                        "base_model": "",
                        "default_weight": 1.0,
                        "civitai_model_id": None,
                    })
                    known_files.add(fname)

        logger.info("list_models: returning %d model(s)", len(result))
        return {"models": result}

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

        os.makedirs(LORAS_SUBDIR, exist_ok=True)
        dest_path = os.path.join(LORAS_SUBDIR, safe_filename)

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

        # Add a basic metadata entry if this is a new file (not a chunk append)
        if not append:
            mgr = ModelMetadataManager(LORAS_DIR)
            if mgr.get_model(safe_filename) is None:
                display_name = safe_filename.rsplit(".", 1)[0]
                mgr.add_model(ModelInfo(
                    filename=safe_filename,
                    model_type="lora",
                    name=display_name,
                    size_bytes=os.path.getsize(dest_path),
                ))
                logger.info("Created metadata entry for '%s'", safe_filename)

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

        # Candidate paths: subdirectory first, then legacy flat path
        subdir_path = os.path.join(LORAS_SUBDIR, safe_filename)
        flat_path = os.path.join(LORAS_DIR, safe_filename)

        # Prevent path traversal for both candidates
        norm_base = os.path.normpath(LORAS_DIR)
        for candidate in (subdir_path, flat_path):
            if not os.path.normpath(candidate).startswith(norm_base):
                raise ValueError(
                    f"Path '{candidate}' is outside the allowed LoRA directory '{LORAS_DIR}'"
                )

        deleted_path: Optional[str] = None
        for candidate in (subdir_path, flat_path):
            if os.path.exists(candidate):
                logger.info("Deleting '%s'", candidate)
                if os.path.isfile(candidate):
                    os.remove(candidate)
                else:
                    shutil.rmtree(candidate)
                deleted_path = candidate
                break

        if deleted_path is None:
            raise FileNotFoundError(
                f"LoRA not found: '{safe_filename}' (checked '{subdir_path}' and '{flat_path}')"
            )

        # Remove metadata entry
        mgr = ModelMetadataManager(LORAS_DIR)
        if mgr.remove_model(safe_filename):
            logger.info("Removed metadata entry for '%s'", safe_filename)

        lora_volume.commit()
        logger.info("Deleted '%s'", deleted_path)

        return {"status": "deleted", "path": deleted_path}

    @modal.method()
    def health(self) -> dict:
        """
        Return GPU and storage health information.

        Returns
        -------
        dict
            Keys: ``status``, ``gpu``, ``vram_total_gb``, ``vram_used_gb``,
            ``model_loaded``, ``models_ready``, ``ip_adapter_loaded``,
            ``loras_dir``, ``loras_storage_free_gb``.
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
            "models_ready": getattr(self, "_models_ready", False),
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
        # First check if the model files actually exist.
        if not check_model_files_exist("ip-adapter"):
            logger.warning(
                "IP-Adapter model file not found — run setup wizard to download it. "
                "Proceeding without IP-Adapter."
            )
            return
        if not check_model_files_exist("clip-vit-h"):
            logger.warning(
                "CLIP ViT-H model not found — run setup wizard to download it. "
                "Proceeding without IP-Adapter."
            )
            return

        # Find the IP-Adapter file (may be in flattened or subfolder location)
        ip_adapter_path = get_model_file_path("ip-adapter")
        if ip_adapter_path is None:
            ip_adapter_dir = get_model_path("ip-adapter")
            ip_adapter_path = os.path.join(
                ip_adapter_dir, "ip-adapter-plus_sdxl_vit-h.safetensors"
            )
        clip_vit_h_path = get_model_path("clip-vit-h")

        logger.info(
            "IP-Adapter not loaded at startup — attempting lazy load "
            "(scale=%.2f) from '%s' with CLIP encoder '%s'",
            scale,
            ip_adapter_path,
            clip_vit_h_path,
        )
        try:
            self._pipe.load_ip_adapter(
                os.path.dirname(ip_adapter_path),
                subfolder="",
                weight_name=os.path.basename(ip_adapter_path),
                image_encoder_folder=clip_vit_h_path,
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


def _inject_trigger_words(prompt: str, lora_names: list[dict], loras_dir: str) -> str:
    """Inject trigger words from metadata into the prompt.

    Parameters
    ----------
    prompt:
        The current positive prompt string.
    lora_names:
        List of LoRA dicts (each with at least a ``"filename"`` key).
    loras_dir:
        Base volume path used to locate the ``.metadata.json`` file.

    Returns
    -------
    str
        The prompt with any missing trigger words prepended.
    """
    mgr = ModelMetadataManager(loras_dir)
    triggers: list[str] = []
    for lora in lora_names:
        filename = lora.get("filename", "")
        if not filename:
            continue
        info = mgr.get_model(filename)
        if info and info.trigger_words:
            for tw in info.trigger_words:
                if tw and tw not in triggers:
                    triggers.append(tw)

    if triggers:
        trigger_str = ", ".join(triggers)
        if trigger_str.lower() not in prompt.lower():
            prompt = f"{trigger_str}, {prompt}" if prompt else trigger_str
            logger.info("Injected trigger words into prompt: %s", triggers)

    return prompt


def _apply_loras(pipe: Any, loras: list[dict], loras_dir: str) -> list[str]:
    """Apply multiple LoRAs simultaneously using diffusers multi-adapter support.

    Parameters
    ----------
    pipe:
        The loaded diffusers pipeline.
    loras:
        List of dicts with keys ``"filename"`` (str, preferred) or ``"name"``
        (str, legacy) and ``"weight"`` (float).
    loras_dir:
        Base volume path where LoRA ``.safetensors`` files are stored.
        Files are looked up in ``LORAS_SUBDIR`` first, then ``loras_dir``.

    Returns
    -------
    list[str]
        Adapter names that were successfully loaded (pass to :func:`_remove_loras`).
    """
    adapter_names: list[str] = []
    adapter_weights: list[float] = []
    failed_adapters: list[str] = []

    for lora in loras:
        name: str = lora.get("name", "")
        filename: str = lora.get("filename", "")
        weight: float = float(lora.get("weight", 1.0))
        if not name and not filename:
            continue

        # Determine the bare filename to look up
        safe_name = os.path.basename(filename) if filename else os.path.basename(name)
        # Use a stable adapter_name for diffusers (prefer display name, fall back to filename)
        adapter_name: str = name or safe_name

        # Look in loras/ subdirectory first, then fall back to legacy flat path
        lora_path = os.path.join(LORAS_SUBDIR, safe_name)
        if not os.path.exists(lora_path):
            lora_path = os.path.join(loras_dir, safe_name)  # Legacy flat path
        if not os.path.exists(lora_path):
            # Last-resort: try appending .safetensors if no extension
            if not os.path.splitext(safe_name)[1]:
                for candidate_dir in (LORAS_SUBDIR, loras_dir):
                    candidate = os.path.join(candidate_dir, f"{safe_name}.safetensors")
                    if os.path.exists(candidate):
                        lora_path = candidate
                        break

        if not os.path.isfile(lora_path):
            # --- Diagnostic: list what files actually exist ---
            logger.warning("Skipping LoRA '%s' — file not found at '%s'", adapter_name, lora_path)
            for diag_dir, diag_label in [(LORAS_SUBDIR, "LORAS_SUBDIR"), (loras_dir, "loras_dir")]:
                if os.path.isdir(diag_dir):
                    files = os.listdir(diag_dir)
                    logger.warning(
                        "  [diag] %s (%s) contains %d files: %s",
                        diag_label, diag_dir, len(files), files[:20],
                    )
                else:
                    logger.warning("  [diag] %s (%s) does not exist", diag_label, diag_dir)
            continue

        logger.info("Loading LoRA '%s' (weight=%.2f) from '%s'", adapter_name, weight, lora_path)
        try:
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            adapter_names.append(adapter_name)
            adapter_weights.append(weight)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to load LoRA '%s': %s — skipping", adapter_name, e)
            failed_adapters.append(adapter_name)
            continue

    # Filter out any failed adapters before activating
    active_names = [n for n in adapter_names if n not in failed_adapters]
    active_weights = [
        w for n, w in zip(adapter_names, adapter_weights) if n not in failed_adapters
    ]

    if active_names:
        logger.info(
            "Activating adapters: %s with weights %s", active_names, active_weights
        )
        pipe.set_adapters(active_names, active_weights)

    return active_names


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
        from compel import CompelForSDXL

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

        # Get device from text encoder to ensure tensors are on the same device
        device = next(pipe.text_encoder.parameters()).device
        compel = CompelForSDXL(pipe, device=device)
        logger.info("CompelForSDXL long-prompt encoder initialised on device: %s", device)
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
        # CompelForSDXL returns a LabelledConditioning dataclass with:
        # embeds, pooled_embeds, negative_embeds, negative_pooled_embeds
        labelled = compel(prompt, negative_prompt=negative_prompt)
        return (
            labelled.embeds,
            labelled.pooled_embeds,
            labelled.negative_embeds,
            labelled.negative_pooled_embeds,
        )
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
    ip_adapter_loaded: bool = False,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    clip_skip: int = 0,
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
    ip_adapter_loaded:
        Whether IP-Adapter is loaded in the pipeline. When True but no
        ip_adapter_image is provided, a dummy image is used to satisfy
        the UNet's image_embeds requirement (scale is 0 so content doesn't matter).
    target_width, target_height:
        Optional explicit output dimensions.
    clip_skip:
        Number of layers to skip in the CLIP text encoder (0 = disabled, 1-12).
        Higher values produce more abstract interpretations.

    Returns
    -------
    PIL.Image.Image
        The generated output image.
    """
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    # Convert image to tensor and ensure it matches pipeline's dtype
    from torchvision import transforms
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device="cuda", dtype=torch.float16)
    kwargs: dict[str, Any] = {
        "image": image_tensor,
        "strength": strength,
        "num_inference_steps": steps,
        "guidance_scale": cfg_scale,
        "generator": generator,
    }

    # Convert control image to tensor with correct dtype
    control_image_tensor = transform(image).unsqueeze(0)
    control_image_tensor = control_image_tensor.to(device="cuda", dtype=torch.float16)

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
    kwargs["control_image"] = control_image_tensor
    kwargs["controlnet_conditioning_scale"] = (
        controlnet_conditioning_scale if controlnet_enabled else 0.0
    )

    if ip_adapter_image is not None:
        kwargs["ip_adapter_image"] = ip_adapter_image
    elif ip_adapter_loaded:
        # IP-Adapter is loaded but disabled - provide dummy image for image_embeds
        # The scale is 0.0 so the actual content doesn't matter
        from PIL import Image
        dummy_image = Image.new("RGB", (224, 224), (0, 0, 0))
        kwargs["ip_adapter_image"] = dummy_image

    # CLIP Skip - pass to pipeline when enabled (value > 0)
    # This skips the last N layers of the CLIP text encoder
    if clip_skip > 0:
        kwargs["clip_skip"] = clip_skip

    logger.info(
        "SDXL generate: steps=%d strength=%.2f cfg=%.1f seed=%d controlnet=%s "
        "(scale=%.2f) ip_adapter=%s target=%s compel=%s clip_skip=%d prompt_len=%d",
        steps,
        strength,
        cfg_scale,
        seed,
        controlnet_enabled,
        controlnet_conditioning_scale if controlnet_enabled else 0.0,
        ip_adapter_image is not None,
        f"{target_width}x{target_height}" if target_width and target_height else "from_image",
        use_compel,
        clip_skip,
        len(prompt),
    )

    result = pipe(**kwargs)
    output_image = result.images[0]
    logger.info("SDXL generation complete")
    return output_image
