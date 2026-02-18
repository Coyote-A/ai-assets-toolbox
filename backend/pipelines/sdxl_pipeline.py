"""
SDXL img2img pipeline wrapper.

Supports:
- Plain img2img via ``StableDiffusionXLImg2ImgPipeline``
- ControlNet tile conditioning via ``StableDiffusionXLControlNetImg2ImgPipeline``
- LoRA adapters via ``pipe.load_lora_weights()`` / ``pipe.fuse_lora()``
"""

import logging
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# TODO: verify HF path — xinsir/controlnet-tile-sdxl-1.0
SDXL_TILE_CONTROLNET_ID = "xinsir/controlnet-tile-sdxl-1.0"


class SDXLPipeline:
    """
    Wrapper around the diffusers SDXL img2img pipeline.

    Parameters
    ----------
    model_path:
        HuggingFace repo ID or local directory path for the SDXL checkpoint.
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

        self._build_pipeline()

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> None:
        if self.controlnet_path:
            self._build_controlnet_pipeline()
        else:
            self._build_plain_pipeline()

    def _build_plain_pipeline(self) -> None:
        from diffusers import StableDiffusionXLImg2ImgPipeline

        logger.info("Building plain SDXL img2img pipeline from '%s'", self.model_path)
        self._pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")
        self._pipe.enable_model_cpu_offload()
        logger.info("SDXL plain pipeline ready")

    def _build_controlnet_pipeline(self) -> None:
        from diffusers import (
            ControlNetModel,
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
        self._pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            self.model_path,
            controlnet=self._controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")
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

        Returns
        -------
        PIL.Image.Image
            The generated output tile.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        kwargs: dict = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "strength": strength,
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
            "generator": generator,
        }

        if self._controlnet is not None:
            # Use the input tile as the control image if none provided
            ctrl_img = controlnet_image if controlnet_image is not None else image
            kwargs["control_image"] = ctrl_img
            kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale

        if ip_adapter_image is not None:
            kwargs["ip_adapter_image"] = ip_adapter_image

        logger.info(
            "SDXL generate: steps=%d strength=%.2f cfg=%.1f seed=%s controlnet=%s ip_adapter=%s",
            steps,
            strength,
            cfg_scale,
            seed,
            self._controlnet is not None,
            ip_adapter_image is not None,
        )

        result = self._pipe(**kwargs)
        output_image: Image.Image = result.images[0]
        logger.info("SDXL generation complete")
        return output_image
