"""
Qwen3-VL captioning pipeline.

Uses ``Qwen/Qwen3-VL-8B-Instruct`` to generate detailed image descriptions
suitable for use as Stable Diffusion prompts.
"""

import logging
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

DEFAULT_SYSTEM_PROMPT = (
    "Describe this image tile in detail for use as a Stable Diffusion prompt. "
    "Focus on visual elements, style, colors, and composition."
)


class QwenPipeline:
    """
    Wrapper around Qwen3-VL-8B-Instruct for image captioning.

    The model is loaded in ``float16`` to reduce VRAM usage (~16–18 GB on
    an A100 80 GB).

    Parameters
    ----------
    model_id:
        HuggingFace repo ID or local path.  Defaults to
        ``"Qwen/Qwen3-VL-8B-Instruct"``.
    cache_dir:
        Optional directory to cache downloaded model weights.
    """

    def __init__(
        self,
        model_id: str = QWEN_MODEL_ID,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self._model = None
        self._processor = None

        self._load()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        logger.info("Loading Qwen3-VL-8B from '%s'", self.model_id)

        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=self.cache_dir,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
        )
        logger.info("Qwen3-VL-8B loaded successfully")

    # ------------------------------------------------------------------
    # Captioning
    # ------------------------------------------------------------------

    def caption(
        self,
        image: Image.Image,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 200,
    ) -> str:
        """
        Generate a caption for the given image.

        Parameters
        ----------
        image:
            Input PIL image (typically a 1024×1024 tile).
        system_prompt:
            Custom system prompt.  Defaults to ``DEFAULT_SYSTEM_PROMPT``.
        max_new_tokens:
            Maximum number of tokens to generate.

        Returns
        -------
        str
            The generated caption string.
        """
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        # Build the conversation messages in Qwen3-VL chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": system_prompt},
                ],
            }
        ]

        logger.debug("Running Qwen caption inference")

        # Apply chat template to get the text input
        text_input = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs (image + text)
        inputs = self._processor(
            text=[text_input],
            images=[image],
            return_tensors="pt",
        )
        # Move tensors to the model's device
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        # Decode only the newly generated tokens (skip the input prompt)
        input_token_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_token_len:]
        caption: str = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        logger.debug("Caption generated (%d chars)", len(caption))
        return caption
