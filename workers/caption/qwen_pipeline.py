"""
Qwen3-VL captioning pipeline for the caption worker.

Loads Qwen3-VL-2B-Instruct from a local directory path and generates
detailed image descriptions suitable for use as Stable Diffusion prompts.

The model is loaded once at init and kept in VRAM permanently — no model
swapping needed since this is a dedicated single-purpose worker.
"""

import logging
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Describe the visual content of this image tile for a Stable Diffusion prompt. "
    "Focus ONLY on objects, elements, and composition - NOT style or aesthetic. "
    "Output ONLY comma-separated keywords describing what is in the tile. "
    "NO explanations, NO markdown, NO headers, NO introductory text. "
    "Example output: small pond, blue water, dirt path, grass, rocks, trees"
)


class QwenPipeline:
    """
    Wrapper around Qwen3-VL-2B-Instruct for image captioning.

    The model is loaded in ``float16`` to reduce VRAM usage (~4–5 GB).
    It is loaded once at construction time and stays resident in VRAM.

    Parameters
    ----------
    model_path:
        Local directory path containing the model weights and config files.
        Defaults to ``"/app/models/qwen3-vl-2b"``.
    """

    def __init__(self, model_path: str = "/app/models/qwen3-vl-2b") -> None:
        self.model_path = model_path
        self._model = None
        self._processor = None
        self._load()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        # Import qwen_vl_utils to register Qwen VL processor classes with transformers
        # This MUST be imported before calling AutoProcessor.from_pretrained()
        # Otherwise, video_processing_auto.py will fail with "if class_name in extractors"
        import qwen_vl_utils

        logger.info("Loading Qwen3-VL-2B from '%s'", self.model_path)

        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        logger.info(
            "Qwen3-VL-2B loaded successfully on device: %s",
            next(self._model.parameters()).device,
        )

    # ------------------------------------------------------------------
    # Captioning
    # ------------------------------------------------------------------

    def caption(
        self,
        image: Image.Image,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 300,
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
