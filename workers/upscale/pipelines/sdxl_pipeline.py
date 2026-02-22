"""
SDXL img2img pipeline wrapper.

Supports:
- Plain img2img via ``StableDiffusionXLImg2ImgPipeline``
- ControlNet tile conditioning via ``StableDiffusionXLControlNetImg2ImgPipeline``
- Multi-LoRA adapters via ``pipe.load_lora_weights()`` / ``pipe.set_adapters()``
- Long-prompt support via ``compel`` (bypasses CLIP's 77-token limit)

Loading strategy
----------------
When ``model_path`` ends with ``.safetensors`` (single-file checkpoint such as
Illustrious-XL-v2.0) ``from_single_file()`` is used.  For diffusers multi-folder
repos (containing ``model_index.json``) ``from_pretrained()`` is used instead.
"""

import logging
import os
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# TODO: verify HF path — xinsir/controlnet-tile-sdxl-1.0
SDXL_TILE_CONTROLNET_ID = "xinsir/controlnet-tile-sdxl-1.0"


def _is_single_file(model_path: str) -> bool:
    """Return ``True`` when *model_path* points to a single ``.safetensors`` file."""
    return model_path.endswith(".safetensors") and os.path.isfile(model_path)


def _build_compel(pipe: any) -> Optional[any]:
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
            "compel is not installed — prompts will be truncated at 77 tokens. "
            "Add 'compel>=2.0.2' to requirements.txt to enable long-prompt support."
        )
        return None
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to initialise compel: %s — falling back to raw prompts", exc)
        return None


def _encode_prompt_with_compel(
    compel: any,
    prompt: str,
    negative_prompt: str,
) -> tuple[Optional[any], Optional[any], Optional[any], Optional[any]]:
    """Encode *prompt* and *negative_prompt* using compel.

    Returns
    -------
    tuple of (conditioning, pooled, neg_conditioning, neg_pooled)
        All four tensors ready to pass to the pipeline as
        ``prompt_embeds``, ``pooled_prompt_embeds``,
        ``negative_prompt_embeds``, ``negative_pooled_prompt_embeds``.
        Returns ``(None, None, None, None)`` on failure.
    """
    try:
        conditioning, pooled = compel(prompt)
        neg_conditioning, neg_pooled = compel(negative_prompt)
        # Pad to equal length so they can be batched together
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length(
            [conditioning, neg_conditioning]
        )
        return conditioning, pooled, neg_conditioning, neg_pooled
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("compel encoding failed: %s — falling back to raw prompt strings", exc)
        return None, None, None, None


class SDXLPipeline:
    """
    Wrapper around the diffusers SDXL img2img pipeline.

    Parameters
    ----------
    model_path:
        HuggingFace repo ID, local diffusers directory, or absolute path to a
        single ``.safetensors`` checkpoint file.
    controlnet_path:
        Optional HuggingFace repo ID or local path for a ControlNet model.
        When provided the pipeline is built with
        ``StableDiffusionXLControlNetImg2ImgPipeline``.
    """

    def __init__(
        self,
        model_path: str,
        controlnet_path: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.controlnet_path = controlnet_path
        self._pipe = None
        self._controlnet = None
        self._active_lora: Optional[str] = None
        self._compel = None

        self._build_pipeline()

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> None:
        if self.controlnet_path:
            self._build_controlnet_pipeline()
        else:
            self._build_plain_pipeline()

        # Initialise compel after the pipeline is ready
        self._compel = _build_compel(self._pipe)

    def _build_plain_pipeline(self) -> None:
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLImg2ImgPipeline

        logger.info("Building plain SDXL img2img pipeline from '%s'", self.model_path)

        if _is_single_file(self.model_path):
            # Single-file checkpoint (e.g. Illustrious-XL-v2.0.safetensors)
            # from_single_file() does not accept cache_dir / local_files_only /
            # variant — it loads directly from the given file path.
            logger.info("Using from_single_file() for single-file checkpoint")
            self._pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
        else:
            # Diffusers multi-folder format
            self._pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )

        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )
        self._pipe.enable_model_cpu_offload()
        logger.info("SDXL plain pipeline ready")

    def _build_controlnet_pipeline(self) -> None:
        from diffusers import (
            ControlNetModel,
            DPMSolverMultistepScheduler,
            StableDiffusionXLControlNetImg2ImgPipeline,
        )

        logger.info(
            "Building SDXL+ControlNet pipeline — base='%s', controlnet='%s'",
            self.model_path,
            self.controlnet_path,
        )
        self._controlnet = ControlNetModel.from_pretrained(
            self.controlnet_path,
            torch_dtype=torch.float16,
        )

        if _is_single_file(self.model_path):
            # Load the base pipeline from a single-file checkpoint first, then
            # convert it to a ControlNet pipeline by injecting the ControlNet.
            # StableDiffusionXLControlNetImg2ImgPipeline.from_single_file() is
            # not universally available in all diffusers versions, so we build
            # the plain pipeline and use from_pipe() to wrap it with ControlNet.
            logger.info(
                "Using from_single_file() + from_pipe() for single-file checkpoint"
            )
            base_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
            self._pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pipe(
                base_pipe,
                controlnet=self._controlnet,
            )
        else:
            # Diffusers multi-folder format
            self._pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                self.model_path,
                controlnet=self._controlnet,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )

        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )
        self._pipe.enable_model_cpu_offload()
        logger.info("SDXL+ControlNet pipeline ready")

    # ------------------------------------------------------------------
    # LoRA management
    # ------------------------------------------------------------------

    def load_lora(self, lora_path: str, lora_scale: float = 1.0) -> None:
        """Load and fuse a LoRA adapter into the pipeline."""
        if self._active_lora is not None:
            logger.info("Unfusing previous LoRA before loading new one")
            try:
                self._pipe.unfuse_lora()
                self._pipe.unload_lora_weights()
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Could not unfuse LoRA: %s", exc)

        logger.info("Loading LoRA from '%s' (scale=%.2f)", lora_path, lora_scale)
        self._pipe.load_lora_weights(lora_path)
        self._pipe.fuse_lora(lora_scale=lora_scale)
        self._active_lora = lora_path
        logger.info("LoRA fused successfully")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.35,
        steps: int = 30,
        cfg_scale: float = 7.0,
        seed: Optional[int] = None,
        controlnet_image: Optional[Image.Image] = None,
        controlnet_conditioning_scale: float = 0.6,
        ip_adapter_image: Optional[Image.Image] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> Image.Image:
        """
        Run SDXL img2img generation.

        Parameters
        ----------
        image:
            Input PIL image (the tile to upscale).
        prompt:
            Text prompt for generation.
        negative_prompt:
            Negative text prompt.
        strength:
            Denoising strength (0.0–1.0).  Recommended: 0.25–0.45 for upscaling.
        steps:
            Number of inference steps.
        cfg_scale:
            Classifier-free guidance scale.
        seed:
            Optional random seed for reproducibility.
        controlnet_image:
            Control image for ControlNet conditioning.  If ``None`` and the
            pipeline has ControlNet, ``image`` is used as the control image.
        controlnet_conditioning_scale:
            Strength of ControlNet conditioning (0.0–1.5).
        ip_adapter_image:
            Optional PIL Image used as the IP-Adapter style reference.  When
            provided the pipeline must have IP-Adapter loaded via
            ``pipe.load_ip_adapter()``.  Ignored when ``None``.
        target_width, target_height:
            Optional explicit output dimensions.  When provided, the pipeline
            generates at this resolution rather than deriving it from the input
            image size.  Useful for generating at the model's native resolution
            (e.g. 1536×1536 for Illustrious-XL) when the input tile is smaller.

        Returns
        -------
        PIL.Image.Image
            The generated output tile.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        kwargs: dict = {
            "image": image,
            "strength": strength,
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
            "generator": generator,
        }

        # Pass explicit width/height when provided so the pipeline generates at
        # the requested resolution (e.g. 1536×1536 for Illustrious-XL native).
        if target_width is not None:
            kwargs["width"] = target_width
        if target_height is not None:
            kwargs["height"] = target_height

        # ------------------------------------------------------------------
        # Prompt encoding — use compel for long-prompt support when available
        # ------------------------------------------------------------------
        use_compel = self._compel is not None
        if use_compel:
            conditioning, pooled, neg_conditioning, neg_pooled = _encode_prompt_with_compel(
                self._compel, prompt, negative_prompt
            )
            if conditioning is not None:
                kwargs["prompt_embeds"] = conditioning
                kwargs["pooled_prompt_embeds"] = pooled
                kwargs["negative_prompt_embeds"] = neg_conditioning
                kwargs["negative_pooled_prompt_embeds"] = neg_pooled
                logger.debug("Using compel prompt embeddings (long-prompt support active)")
            else:
                # compel failed — fall back to raw strings
                use_compel = False

        if not use_compel:
            kwargs["prompt"] = prompt
            kwargs["negative_prompt"] = negative_prompt

        if self._controlnet is not None:
            # Use the input tile as the control image if none provided
            ctrl_img = controlnet_image if controlnet_image is not None else image
            kwargs["control_image"] = ctrl_img
            kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale

        if ip_adapter_image is not None:
            kwargs["ip_adapter_image"] = ip_adapter_image

        logger.info(
            "SDXL generate: steps=%d strength=%.2f cfg=%.1f seed=%s controlnet=%s ip_adapter=%s "
            "target_size=%s compel=%s prompt_len=%d",
            steps,
            strength,
            cfg_scale,
            seed,
            self._controlnet is not None,
            ip_adapter_image is not None,
            f"{target_width}x{target_height}" if target_width and target_height else "from_image",
            use_compel,
            len(prompt),
        )

        result = self._pipe(**kwargs)
        output_image: Image.Image = result.images[0]
        logger.info("SDXL generation complete")
        return output_image
