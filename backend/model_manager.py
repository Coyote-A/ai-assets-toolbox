"""
ModelManager — singleton responsible for dynamic model loading/unloading.

All model weights are expected to reside on the RunPod Network Volume mounted
at ``/runpod-volume/``.  The directory layout mirrors the architecture doc:

    /runpod-volume/models/
        checkpoints/   — SDXL / Flux base checkpoints (HF snapshot dirs)
        loras/         — LoRA .safetensors files
        controlnets/   — ControlNet model dirs
"""

import gc
import logging
import os
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Network-volume paths
# ---------------------------------------------------------------------------
VOLUME_ROOT = "/runpod-volume"
MODELS_ROOT = os.path.join(VOLUME_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_ROOT, "checkpoints")
LORAS_DIR = os.path.join(MODELS_ROOT, "loras")
CONTROLNETS_DIR = os.path.join(MODELS_ROOT, "controlnets")

# ---------------------------------------------------------------------------
# Supported model registry
# Maps short model names → HuggingFace repo IDs (or local paths on the volume)
# ---------------------------------------------------------------------------
CHECKPOINT_MODELS: dict[str, str] = {
    # TODO: verify HF path — replace with actual Z-Image SDXL repo if different
    "z-image-xl": "stabilityai/stable-diffusion-xl-base-1.0",
    # TODO: verify HF path — AstraliteHeart/pony-diffusion-v6-xl may not be the canonical name
    "pony-v6": "AstraliteHeart/pony-diffusion-v6-xl",
    # TODO: verify HF path — OnomaAIResearch/Illustrious-xl-early-release-v0
    "illustrious-xl": "OnomaAIResearch/Illustrious-xl-early-release-v0",
    "flux-dev": "black-forest-labs/FLUX.1-dev",
}

CONTROLNET_MODELS: dict[str, str] = {
    # TODO: verify HF path — xinsir/controlnet-tile-sdxl-1.0
    "sdxl-tile": "xinsir/controlnet-tile-sdxl-1.0",
    # TODO: verify HF path — jasperai/Flux.1-dev-Controlnet-Upscaler or similar
    "flux-tile": "jasperai/Flux.1-dev-Controlnet-Upscaler",
}

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def _is_flux_model(model_name: str) -> bool:
    return "flux" in model_name.lower()


class ModelManager:
    """
    Singleton that owns all loaded model objects.

    Only one diffusion model (SDXL or Flux) and one utility model (Qwen) are
    kept in VRAM at a time.  When a different model is requested the current
    one is deleted and the CUDA cache is cleared before loading the new one.
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
        # Diffusion pipeline (SDXL or Flux)
        self._diffusion_pipe: Any = None
        self._diffusion_model_name: Optional[str] = None

        # ControlNet model (kept alongside the diffusion pipe)
        self._controlnet: Any = None
        self._controlnet_name: Optional[str] = None

        # Qwen VLM
        self._qwen_model: Any = None
        self._qwen_processor: Any = None
        self._qwen_loaded: bool = False

        # Track which LoRA is currently fused
        self._active_lora: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_diffusion_model(self, model_name: str) -> Any:
        """
        Load a diffusion model by short name (e.g. ``"z-image-xl"``).

        If the requested model is already loaded, returns the existing pipeline
        without reloading.  Otherwise unloads the current model first.
        """
        if self._diffusion_model_name == model_name and self._diffusion_pipe is not None:
            logger.info("Diffusion model '%s' already loaded — reusing", model_name)
            return self._diffusion_pipe

        self.unload_current()

        hf_path = self._resolve_checkpoint_path(model_name)
        logger.info("Loading diffusion model '%s' from '%s'", model_name, hf_path)

        if _is_flux_model(model_name):
            self._diffusion_pipe = self._load_flux(hf_path)
        else:
            self._diffusion_pipe = self._load_sdxl(hf_path)

        self._diffusion_model_name = model_name
        self._active_lora = None
        logger.info("Diffusion model '%s' loaded successfully", model_name)
        return self._diffusion_pipe

    def load_qwen(self) -> tuple[Any, Any]:
        """
        Load Qwen2.5-VL-7B for captioning.

        Returns ``(model, processor)`` tuple.  Unloads any currently loaded
        diffusion model first to free VRAM.
        """
        if self._qwen_loaded and self._qwen_model is not None:
            logger.info("Qwen already loaded — reusing")
            return self._qwen_model, self._qwen_processor

        # Unload diffusion model to free VRAM before loading Qwen (~18 GB)
        self.unload_current()

        logger.info("Loading Qwen2.5-VL-7B from '%s'", QWEN_MODEL_ID)

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        cache_dir = os.path.join(MODELS_ROOT, "qwen", "qwen2.5-vl-7b")
        os.makedirs(cache_dir, exist_ok=True)

        self._qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
        )
        self._qwen_processor = AutoProcessor.from_pretrained(
            QWEN_MODEL_ID,
            cache_dir=cache_dir,
        )
        self._qwen_loaded = True
        logger.info("Qwen2.5-VL-7B loaded successfully")
        return self._qwen_model, self._qwen_processor

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

        hf_path = self._resolve_controlnet_path(model_name)
        logger.info("Loading ControlNet '%s' from '%s'", model_name, hf_path)

        from diffusers import ControlNetModel

        self._controlnet = ControlNetModel.from_pretrained(
            hf_path,
            torch_dtype=torch.float16,
            cache_dir=os.path.join(CONTROLNETS_DIR, model_name),
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

        if self._controlnet is not None:
            logger.info("Unloading ControlNet '%s'", self._controlnet_name)
            del self._controlnet
            self._controlnet = None
            self._controlnet_name = None

        if self._qwen_model is not None:
            logger.info("Unloading Qwen model")
            del self._qwen_model
            del self._qwen_processor
            self._qwen_model = None
            self._qwen_processor = None
            self._qwen_loaded = False

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
            }
        if self._qwen_loaded:
            return {"type": "qwen", "name": "Qwen2.5-VL-7B"}
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_checkpoint_path(self, model_name: str) -> str:
        """
        Return the local path or HF repo ID for a checkpoint.

        Prefers a local directory on the network volume; falls back to the
        HuggingFace repo ID so diffusers can download it automatically.
        """
        local_path = os.path.join(CHECKPOINTS_DIR, model_name)
        if os.path.isdir(local_path):
            logger.debug("Using local checkpoint path: %s", local_path)
            return local_path

        hf_id = CHECKPOINT_MODELS.get(model_name)
        if hf_id is None:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(CHECKPOINT_MODELS.keys())}"
            )
        logger.debug("Local checkpoint not found; using HF repo: %s", hf_id)
        return hf_id

    def _resolve_controlnet_path(self, model_name: str) -> str:
        local_path = os.path.join(CONTROLNETS_DIR, model_name)
        if os.path.isdir(local_path):
            return local_path

        hf_id = CONTROLNET_MODELS.get(model_name)
        if hf_id is None:
            raise ValueError(
                f"Unknown ControlNet '{model_name}'. "
                f"Available: {list(CONTROLNET_MODELS.keys())}"
            )
        return hf_id

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
        from diffusers import StableDiffusionXLImg2ImgPipeline

        cache_dir = CHECKPOINTS_DIR
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir=cache_dir,
        )
        pipe = pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        return pipe

    def _load_flux(self, model_path: str) -> Any:
        from diffusers import FluxImg2ImgPipeline

        cache_dir = CHECKPOINTS_DIR
        pipe = FluxImg2ImgPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
        )
        pipe = pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        return pipe
