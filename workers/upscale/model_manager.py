"""
ModelManager — singleton responsible for dynamic model loading/unloading.

All core model weights are pre-downloaded into the Docker image at build time
and stored in flat directories under ``/app/models/``.
At runtime the manager loads directly from those paths — no HF cache lookups,
no network calls for pre-baked models.

LoRA adapters are the only models loaded dynamically at runtime; they are
expected to reside on the RunPod Network Volume at:

    /runpod-volume/models/loras/   — LoRA .safetensors files
"""

import gc
import logging
import os
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flat model directory paths (baked into the Docker image at build time)
# ---------------------------------------------------------------------------
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")

ILLUSTRIOUS_XL_PATH = os.path.join(MODELS_DIR, "illustrious-xl", "Illustrious-XL-v2.0.safetensors")
CONTROLNET_TILE_PATH = os.path.join(MODELS_DIR, "controlnet-tile")
SDXL_VAE_PATH = os.path.join(MODELS_DIR, "sdxl-vae")
IP_ADAPTER_PATH = os.path.join(MODELS_DIR, "ip-adapter", "ip-adapter-plus_sdxl_vit-h.safetensors")
CLIP_VIT_H_PATH = os.path.join(MODELS_DIR, "clip-vit-h")

# ---------------------------------------------------------------------------
# Network-volume paths (LoRAs + outputs only at runtime)
# ---------------------------------------------------------------------------
VOLUME_ROOT = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
MODELS_ROOT = os.path.join(VOLUME_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_ROOT, "checkpoints")
LORAS_DIR = os.path.join(MODELS_ROOT, "loras")
CONTROLNETS_DIR = os.path.join(MODELS_ROOT, "controlnets")

# ---------------------------------------------------------------------------
# Supported model registry
# Maps short model names → local file paths
# ---------------------------------------------------------------------------
CHECKPOINT_MODELS: dict[str, str] = {
    "illustrious-xl": ILLUSTRIOUS_XL_PATH,
}

CONTROLNET_MODELS: dict[str, str] = {
    "sdxl-tile": CONTROLNET_TILE_PATH,
}


class ModelManager:
    """
    Singleton that owns all loaded model objects.

    Only one diffusion model (SDXL/Illustrious) is kept in VRAM at a time.
    When a different model is requested the current one is deleted and the
    CUDA cache is cleared before loading the new one.
    """

    _instance: Optional["ModelManager"] = None

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Internal state
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # Diffusion pipeline (SDXL/Illustrious)
        self._diffusion_pipe: Any = None
        self._diffusion_model_name: Optional[str] = None

        # ControlNet model (kept alongside the diffusion pipe)
        self._controlnet: Any = None
        self._controlnet_name: Optional[str] = None

        # Track which LoRA is currently fused
        self._active_lora: Optional[str] = None

        # IP-Adapter state
        self._ip_adapter_loaded: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def diffusion_pipe(self) -> Any:
        """Public accessor for the loaded diffusion pipeline."""
        if self._diffusion_pipe is None:
            raise RuntimeError("No diffusion model loaded. Call load_diffusion_model() first.")
        return self._diffusion_pipe

    def load_diffusion_model(self, model_name: str) -> Any:
        """
        Load a diffusion model by short name (e.g. ``"illustrious-xl"``).

        If the requested model is already loaded, returns the existing pipeline
        without reloading.  Otherwise unloads the current model first.
        """
        if self._diffusion_model_name == model_name and self._diffusion_pipe is not None:
            logger.info("Diffusion model '%s' already loaded — reusing", model_name)
            return self._diffusion_pipe

        self.unload_current()

        model_path = self._resolve_checkpoint_path(model_name)
        logger.info("Loading diffusion model '%s' from '%s'", model_name, model_path)

        self._diffusion_pipe = self._load_sdxl(model_path)

        self._diffusion_model_name = model_name
        self._active_lora = None
        logger.info("Diffusion model '%s' loaded successfully", model_name)
        return self._diffusion_pipe

    def load_controlnet(self, model_name: str) -> Any:
        """
        Load a ControlNet model by short name (e.g. ``"sdxl-tile"``).

        The ControlNet is kept alongside the diffusion pipeline; it is NOT
        unloaded when switching diffusion models — call ``unload_current()``
        explicitly if needed.
        """
        if self._controlnet_name == model_name and self._controlnet is not None:
            logger.info("ControlNet '%s' already loaded — reusing", model_name)
            return self._controlnet

        model_path = self._resolve_controlnet_path(model_name)
        logger.info("Loading ControlNet '%s' from '%s'", model_name, model_path)

        from diffusers import ControlNetModel

        self._controlnet = ControlNetModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        self._controlnet_name = model_name
        logger.info("ControlNet '%s' loaded successfully", model_name)
        return self._controlnet

    def apply_lora(self, lora_name: str, weight: float = 1.0) -> None:
        """
        Load and fuse a LoRA adapter into the currently loaded diffusion pipeline.

        ``lora_name`` should be the filename (without extension) of a
        ``.safetensors`` file located in ``LORAS_DIR``.
        """
        if self._diffusion_pipe is None:
            raise RuntimeError("No diffusion model loaded; cannot apply LoRA")

        if self._active_lora == lora_name:
            logger.info("LoRA '%s' already applied — skipping", lora_name)
            return

        # Unfuse any previously fused LoRA before applying a new one
        if self._active_lora is not None:
            logger.info("Unfusing previous LoRA '%s'", self._active_lora)
            try:
                self._diffusion_pipe.unfuse_lora()
                self._diffusion_pipe.unload_lora_weights()
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Could not unfuse previous LoRA: %s", exc)

        lora_path = self._resolve_lora_path(lora_name)
        logger.info("Applying LoRA '%s' (weight=%.2f) from '%s'", lora_name, weight, lora_path)

        self._diffusion_pipe.load_lora_weights(lora_path)
        self._diffusion_pipe.fuse_lora(lora_scale=weight)
        self._active_lora = lora_name
        logger.info("LoRA '%s' applied and fused", lora_name)

    def unload_current(self) -> None:
        """
        Unload all currently loaded models and free VRAM.
        """
        if self._diffusion_pipe is not None:
            logger.info("Unloading diffusion model '%s'", self._diffusion_model_name)
            del self._diffusion_pipe
            self._diffusion_pipe = None
            self._diffusion_model_name = None
            self._active_lora = None
            self._ip_adapter_loaded = False

        if self._controlnet is not None:
            logger.info("Unloading ControlNet '%s'", self._controlnet_name)
            del self._controlnet
            self._controlnet = None
            self._controlnet_name = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("VRAM cleared")

    def get_current_model_info(self) -> Optional[dict[str, Any]]:
        """Return a dict describing the currently loaded model, or ``None``."""
        if self._diffusion_pipe is not None:
            return {
                "type": "diffusion",
                "name": self._diffusion_model_name,
                "lora": self._active_lora,
                "controlnet": self._controlnet_name,
                "ip_adapter_loaded": self._ip_adapter_loaded,
            }
        return None

    def load_ip_adapter(self, scale: float = 0.6) -> None:
        """
        Load IP-Adapter weights onto the current SDXL pipeline.

        Uses the pre-baked ``/app/models/ip-adapter/`` weights with the
        CLIP ViT-H image encoder from ``/app/models/clip-vit-h/``.
        Sets the adapter scale immediately after loading.

        Parameters
        ----------
        scale:
            IP-Adapter influence scale (0.0–1.0).  Default 0.6.
        """
        if self._diffusion_pipe is None:
            raise RuntimeError("No diffusion model loaded; cannot load IP-Adapter")

        if self._ip_adapter_loaded:
            # Already loaded — just update the scale
            logger.info("IP-Adapter already loaded — updating scale to %.2f", scale)
            try:
                self._diffusion_pipe.set_ip_adapter_scale(scale)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Could not update IP-Adapter scale: %s", exc)
            return

        logger.info(
            "Loading IP-Adapter from '%s' with CLIP encoder '%s', scale=%.2f",
            IP_ADAPTER_PATH,
            CLIP_VIT_H_PATH,
            scale,
        )

        try:
            self._diffusion_pipe.load_ip_adapter(
                os.path.dirname(IP_ADAPTER_PATH),
                subfolder="",
                weight_name=os.path.basename(IP_ADAPTER_PATH),
                image_encoder_folder=CLIP_VIT_H_PATH,
            )
            self._diffusion_pipe.set_ip_adapter_scale(scale)
            self._ip_adapter_loaded = True
            logger.info("IP-Adapter loaded successfully (scale=%.2f)", scale)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Failed to load IP-Adapter — continuing without it: %s", exc
            )

    def unload_ip_adapter(self) -> None:
        """
        Unload IP-Adapter from the current SDXL pipeline to free VRAM.

        Sets scale to 0 first (safe fallback), then calls ``unload_ip_adapter()``
        if the pipeline supports it.
        """
        if not self._ip_adapter_loaded or self._diffusion_pipe is None:
            return

        logger.info("Unloading IP-Adapter")
        try:
            # Set scale to 0 as a safe fallback before unloading
            self._diffusion_pipe.set_ip_adapter_scale(0.0)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Could not set IP-Adapter scale to 0: %s", exc)

        try:
            self._diffusion_pipe.unload_ip_adapter()
            logger.info("IP-Adapter unloaded successfully")
        except AttributeError:
            logger.info("Pipeline does not support unload_ip_adapter() — scale set to 0")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Could not unload IP-Adapter: %s", exc)

        self._ip_adapter_loaded = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_checkpoint_path(self, model_name: str) -> str:
        """
        Return the local file path for a checkpoint.

        Checks (in order):
        1. Network-volume override directory (custom checkpoints placed there).
        2. Pre-baked flat model directory under ``/app/models/``.
        """
        # Allow network-volume override (e.g. custom checkpoints placed there)
        local_path = os.path.join(CHECKPOINTS_DIR, model_name)
        if os.path.isdir(local_path):
            logger.debug("Using network-volume checkpoint override: %s", local_path)
            return local_path

        baked_path = CHECKPOINT_MODELS.get(model_name)
        if baked_path is None:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(CHECKPOINT_MODELS.keys())}"
            )
        logger.debug("Using pre-baked model path: %s", baked_path)
        return baked_path

    def _resolve_controlnet_path(self, model_name: str) -> str:
        """
        Return the local directory path for a ControlNet model.

        Checks (in order):
        1. Network-volume override directory.
        2. Pre-baked flat model directory under ``/app/models/``.
        """
        # Allow network-volume override
        local_path = os.path.join(CONTROLNETS_DIR, model_name)
        if os.path.isdir(local_path):
            return local_path

        baked_path = CONTROLNET_MODELS.get(model_name)
        if baked_path is None:
            raise ValueError(
                f"Unknown ControlNet '{model_name}'. "
                f"Available: {list(CONTROLNET_MODELS.keys())}"
            )
        return baked_path

    def _resolve_lora_path(self, lora_name: str) -> str:
        # Try exact filename first, then with .safetensors extension
        for candidate in [lora_name, f"{lora_name}.safetensors"]:
            path = os.path.join(LORAS_DIR, candidate)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            f"LoRA '{lora_name}' not found in '{LORAS_DIR}'"
        )

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------

    def _load_sdxl(self, model_path: str) -> Any:
        """
        Load an SDXL img2img pipeline from *model_path*.

        Supports two source formats:

        1. **Local single ``.safetensors`` file** — loaded with
           ``from_single_file()``.
        2. **Local diffusers directory** (contains ``model_index.json``) —
           loaded with ``from_pretrained()``.
        """
        from diffusers import StableDiffusionXLImg2ImgPipeline

        if model_path.endswith(".safetensors") and os.path.isfile(model_path):
            logger.info(
                "Loading SDXL from local single-file checkpoint: %s", model_path
            )
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                vae=self._load_vae(),
            )
            # NOTE: enable_model_cpu_offload() manages device placement automatically.
            # Do NOT call pipe.to("cuda") before it — that would defeat CPU offloading
            # by moving all weights to VRAM upfront, preventing SDXL + ControlNet +
            # IP-Adapter from sharing VRAM by offloading inactive components to CPU RAM.
            pipe.enable_model_cpu_offload()
            return pipe

        if os.path.isdir(model_path):
            logger.info(
                "Loading SDXL from local diffusers directory: %s", model_path
            )
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                vae=self._load_vae(),
            )
            pipe.enable_model_cpu_offload()
            return pipe

        raise FileNotFoundError(
            f"SDXL model path does not exist or is not a recognised format: {model_path}"
        )

    def _load_vae(self) -> Any:
        """
        Load the SDXL VAE fp16-fix from the pre-baked path.

        Returns the VAE model, or ``None`` if the path does not exist
        (pipeline will use its default VAE in that case).
        """
        if not os.path.isdir(SDXL_VAE_PATH):
            logger.warning(
                "SDXL VAE path not found at '%s' — using pipeline default VAE",
                SDXL_VAE_PATH,
            )
            return None

        from diffusers import AutoencoderKL

        logger.info("Loading SDXL VAE fp16-fix from '%s'", SDXL_VAE_PATH)
        vae = AutoencoderKL.from_pretrained(
            SDXL_VAE_PATH,
            torch_dtype=torch.float16,
        )
        return vae
