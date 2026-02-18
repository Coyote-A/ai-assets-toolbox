"""
ModelManager — singleton responsible for dynamic model loading/unloading.

All core model weights are pre-downloaded into the Docker image at build time
and stored in ``/app/hf_cache`` (the ``HF_HOME`` baked into the image).
At runtime the manager loads from that local cache with ``local_files_only=True``
so there are zero network calls for precached models.

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
# Precached HF cache path (baked into the Docker image at build time)
# ---------------------------------------------------------------------------
HF_CACHE_DIR = os.environ.get("HF_HOME", "/app/hf_cache")

# ---------------------------------------------------------------------------
# RunPod Cached Model directory (populated by RunPod's model caching feature)
# ---------------------------------------------------------------------------
RUNPOD_CACHE_DIR = os.environ.get("RUNPOD_MODEL_CACHE", "/runpod-volume/huggingface-cache/hub")

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
# Maps short model names → HuggingFace repo IDs
# ---------------------------------------------------------------------------
CHECKPOINT_MODELS: dict[str, str] = {
    "illustrious-xl": "OnomaAIResearch/Illustrious-xl-early-release-v0",
}

CONTROLNET_MODELS: dict[str, str] = {
    "sdxl-tile": "xinsir/controlnet-tile-sdxl-1.0",
}

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


class ModelManager:
    """
    Singleton that owns all loaded model objects.

    Only one diffusion model (SDXL/Illustrious) and one utility model (Qwen)
    are kept in VRAM at a time.  When a different model is requested the
    current one is deleted and the CUDA cache is cleared before loading the
    new one.
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

        # Qwen VLM
        self._qwen_model: Any = None
        self._qwen_processor: Any = None
        self._qwen_loaded: bool = False

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

        hf_path = self._resolve_checkpoint_path(model_name)
        logger.info("Loading diffusion model '%s' from '%s'", model_name, hf_path)

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

        Loading priority:
        1. RunPod Cached Model directory (RUNPOD_MODEL_CACHE / network volume)
        2. Baked-in Docker image HF cache (local_files_only=True)
        3. Live HuggingFace download (last resort)
        """
        if self._qwen_loaded and self._qwen_model is not None:
            logger.info("Qwen already loaded — reusing")
            return self._qwen_model, self._qwen_processor

        # Unload diffusion model to free VRAM before loading Qwen (~15 GB)
        self.unload_current()

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        # --- 1. Try RunPod Cached Model path ---
        runpod_path = self._find_runpod_cached_model(QWEN_MODEL_ID)
        if runpod_path is not None:
            logger.info(
                "Loading Qwen2.5-VL-7B from RunPod cache: %s", runpod_path
            )
            self._qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                runpod_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self._qwen_processor = AutoProcessor.from_pretrained(runpod_path)
            self._qwen_loaded = True
            logger.info("Qwen2.5-VL-7B loaded successfully (source: RunPod cache)")
            return self._qwen_model, self._qwen_processor

        # --- 2. Try baked-in Docker image HF cache ---
        logger.info(
            "RunPod cache not found; trying baked-in HF cache at '%s'", HF_CACHE_DIR
        )
        try:
            self._qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                QWEN_MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=HF_CACHE_DIR,
                local_files_only=True,
            )
            self._qwen_processor = AutoProcessor.from_pretrained(
                QWEN_MODEL_ID,
                cache_dir=HF_CACHE_DIR,
                local_files_only=True,
            )
            self._qwen_loaded = True
            logger.info("Qwen2.5-VL-7B loaded successfully (source: baked-in HF cache)")
            return self._qwen_model, self._qwen_processor
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Baked-in HF cache miss for Qwen — falling back to live download: %s", exc
            )

        # --- 3. Last resort: live HuggingFace download ---
        logger.info("Downloading Qwen2.5-VL-7B from HuggingFace (last resort)")
        self._qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self._qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
        self._qwen_loaded = True
        logger.info("Qwen2.5-VL-7B loaded successfully (source: live HF download)")
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
            cache_dir=HF_CACHE_DIR,
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
                "ip_adapter_loaded": self._ip_adapter_loaded,
            }
        if self._qwen_loaded:
            return {"type": "qwen", "name": "Qwen2.5-VL-7B"}
        return None

    def load_ip_adapter(self, scale: float = 0.6) -> None:
        """
        Load IP-Adapter weights onto the current SDXL pipeline.

        Uses the pre-cached ``h94/IP-Adapter`` weights (sdxl_models subfolder).
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
            "Loading IP-Adapter (sdxl_models/ip-adapter_sdxl_vit-h.safetensors) scale=%.2f",
            scale,
        )
        try:
            self._diffusion_pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl_vit-h.safetensors",
                cache_dir=HF_CACHE_DIR,
                local_files_only=True,
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
        Return the HF repo ID for a checkpoint.

        The actual weights are loaded from the precached HF cache at
        ``HF_CACHE_DIR`` (``/app/hf_cache``) via ``local_files_only=True``
        in ``_load_sdxl``.  A network-volume local directory is also checked
        first as an override path (useful for development / custom checkpoints).
        """
        # Allow network-volume override (e.g. custom checkpoints placed there)
        local_path = os.path.join(CHECKPOINTS_DIR, model_name)
        if os.path.isdir(local_path):
            logger.debug("Using network-volume checkpoint override: %s", local_path)
            return local_path

        hf_id = CHECKPOINT_MODELS.get(model_name)
        if hf_id is None:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(CHECKPOINT_MODELS.keys())}"
            )
        # Return the HF repo ID; _load_sdxl will resolve it from HF_CACHE_DIR
        logger.debug("Using precached HF repo: %s (cache: %s)", hf_id, HF_CACHE_DIR)
        return hf_id

    def _resolve_controlnet_path(self, model_name: str) -> str:
        # Allow network-volume override
        local_path = os.path.join(CONTROLNETS_DIR, model_name)
        if os.path.isdir(local_path):
            return local_path

        hf_id = CONTROLNET_MODELS.get(model_name)
        if hf_id is None:
            raise ValueError(
                f"Unknown ControlNet '{model_name}'. "
                f"Available: {list(CONTROLNET_MODELS.keys())}"
            )
        # Return the HF repo ID; load_controlnet will resolve from HF_CACHE_DIR
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

    def _find_runpod_cached_model(self, model_id: str) -> Optional[str]:
        """
        Check if a model exists in RunPod's cached model directory.

        RunPod's model caching feature stores models under:
            <RUNPOD_CACHE_DIR>/models--<org>--<name>/snapshots/<hash>/

        Returns the path to the first available snapshot, or ``None`` if the
        model is not present in the RunPod cache.
        """
        cache_name = model_id.replace("/", "--")
        snapshots_dir = os.path.join(
            RUNPOD_CACHE_DIR, f"models--{cache_name}", "snapshots"
        )
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                logger.debug(
                    "Found RunPod cached model '%s' at: %s", model_id, snapshot_path
                )
                return snapshot_path
        logger.debug(
            "RunPod cached model '%s' not found (checked: %s)", model_id, snapshots_dir
        )
        return None

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------

    def _load_sdxl(self, model_path: str) -> Any:
        from diffusers import StableDiffusionXLImg2ImgPipeline

        # If model_path is an absolute local directory (network-volume override),
        # load directly without cache_dir / local_files_only constraints.
        # Otherwise use the precached HF cache with local_files_only=True.
        is_local_dir = os.path.isabs(model_path) and os.path.isdir(model_path)

        kwargs: dict = {
            "torch_dtype": torch.float16,
            "use_safetensors": True,
        }
        if is_local_dir:
            logger.info("Loading SDXL from local directory override: %s", model_path)
        else:
            kwargs["cache_dir"] = HF_CACHE_DIR
            kwargs["local_files_only"] = True

        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_path, **kwargs)
        # NOTE: enable_model_cpu_offload() manages device placement automatically.
        # Do NOT call pipe.to("cuda") before it — that would defeat CPU offloading
        # by moving all weights to VRAM upfront, preventing SDXL + ControlNet +
        # IP-Adapter from sharing VRAM by offloading inactive components to CPU RAM.
        pipe.enable_model_cpu_offload()
        return pipe
