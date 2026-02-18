"""
Flux img2img pipeline wrapper.

Notes
-----
- Flux.1-dev does **not** use ``negative_prompt`` or ``cfg_scale`` in the
  same way as SDXL.  Those parameters are intentionally omitted from the
  ``generate()`` signature.
- Flux ControlNet support in diffusers is experimental / limited at the time
  of writing.  ControlNet integration is marked as TODO below.
- LoRA loading follows the same diffusers pattern as SDXL.
"""

import logging
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# TODO: verify HF path — jasperai/Flux.1-dev-Controlnet-Upscaler or similar community model
FLUX_TILE_CONTROLNET_ID = "jasperai/Flux.1-dev-Controlnet-Upscaler"


class FluxPipeline:
    """
    Wrapper around the diffusers ``FluxImg2ImgPipeline``.

    Parameters
    ----------
    model_path:
        HuggingFace repo ID or local directory path for the Flux checkpoint.
    controlnet_path:
        Optional ControlNet model path.  Flux ControlNet support is
        experimental — if loading fails the pipeline falls back to plain
        img2img.
    """

    def __init__(
        self,
        model_path: str,
        controlnet_path: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.controlnet_path = controlnet_path
        self._pipe = None
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
        from diffusers import FluxImg2ImgPipeline

        logger.info("Building plain Flux img2img pipeline from '%s'", self.model_path)
        self._pipe = FluxImg2ImgPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
        ).to("cuda")
        self._pipe.enable_model_cpu_offload()
        logger.info("Flux plain pipeline ready")

    def _build_controlnet_pipeline(self) -> None:
        """
        TODO: Flux ControlNet img2img pipeline.

        ``FluxControlNetImg2ImgPipeline`` may not be available in all
        diffusers versions.  This method attempts to load it and falls back
        to the plain pipeline if the import fails.
        """
        try:
            # TODO: verify that FluxControlNetModel and
            #       FluxControlNetImg2ImgPipeline are available in the
            #       installed diffusers version.
            from diffusers import FluxControlNetModel, FluxControlNetImg2ImgPipeline  # type: ignore[attr-defined]

            logger.info(
                "Building Flux+ControlNet pipeline — base='%s', controlnet='%s'",
                self.model_path,
                self.controlnet_path,
            )
            controlnet = FluxControlNetModel.from_pretrained(
                self.controlnet_path,
                torch_dtype=torch.float16,
            )
            self._pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
                self.model_path,
                controlnet=controlnet,
                torch_dtype=torch.float16,
            ).to("cuda")
            self._pipe.enable_model_cpu_offload()
            logger.info("Flux+ControlNet pipeline ready")
        except (ImportError, AttributeError) as exc:
            logger.warning(
                "Flux ControlNet pipeline not available (%s); "
                "falling back to plain Flux img2img",
                exc,
            )
            self._build_plain_pipeline()

    # ------------------------------------------------------------------
    # LoRA management
    # ------------------------------------------------------------------

    def load_lora(self, lora_path: str, lora_scale: float = 1.0) -> None:
        """Load and fuse a LoRA adapter into the Flux pipeline."""
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
        strength: float = 0.35,
        steps: int = 28,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Run Flux img2img generation.

        Parameters
        ----------
        image:
            Input PIL image (the tile to upscale).
        prompt:
            Text prompt for generation.
        strength:
            Denoising strength (0.0–1.0).
        steps:
            Number of inference steps.  Flux typically uses fewer steps than
            SDXL (20–30 is common).
        seed:
            Optional random seed for reproducibility.

        Returns
        -------
        PIL.Image.Image
            The generated output tile.

        Notes
        -----
        Flux does not use ``negative_prompt`` or ``guidance_scale`` in the
        same way as SDXL.  ``guidance_scale`` is set to 0.0 (unconditional)
        by default for Flux.1-dev img2img.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        logger.info(
            "Flux generate: steps=%d strength=%.2f seed=%s",
            steps,
            strength,
            seed,
        )

        result = self._pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            num_inference_steps=steps,
            generator=generator,
        )
        output_image: Image.Image = result.images[0]
        logger.info("Flux generation complete")
        return output_image
