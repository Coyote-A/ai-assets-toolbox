"""
Caption action handler.

Accepts a list of image tiles (base64-encoded), runs each through the
Qwen2.5-VL-7B captioning pipeline, and returns per-tile captions.

Request schema
--------------
{
    "action": "caption",
    "tiles": [
        {"tile_id": "0_0", "image_b64": "<base64>"},
        ...
    ],
    "caption_params": {
        "system_prompt": "...",   # optional
        "max_tokens": 200         # optional
    }
}

Response schema
---------------
{
    "captions": [
        {"tile_id": "0_0", "caption": "..."},
        ...
    ]
}
"""

import logging
from typing import Any

from model_manager import ModelManager
from pipelines.qwen_pipeline import QwenPipeline, DEFAULT_SYSTEM_PROMPT
from utils.image_utils import b64_to_pil, crop_to_tile

logger = logging.getLogger(__name__)


def handle_caption(job_input: dict[str, Any]) -> dict[str, Any]:
    """
    Handle the ``caption`` action.

    Parameters
    ----------
    job_input:
        The ``input`` dict from the RunPod job payload.

    Returns
    -------
    dict
        ``{"captions": [{"tile_id": ..., "caption": ...}, ...]}``
    """
    tiles: list[dict[str, Any]] = job_input.get("tiles", [])
    if not tiles:
        # Legacy single-image support: accept top-level image + tile_coords
        image_b64: str = job_input.get("image", "")
        if not image_b64:
            raise ValueError("No 'tiles' or 'image' field found in caption request")
        tile_coords = job_input.get("tile_coords")
        tiles = [{"tile_id": "0_0", "image_b64": image_b64, "tile_coords": tile_coords}]

    caption_params: dict[str, Any] = job_input.get("caption_params", {})
    system_prompt: str = caption_params.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    max_tokens: int = int(caption_params.get("max_tokens", 200))

    logger.info("Caption action: processing %d tile(s)", len(tiles))

    # Load Qwen via ModelManager (unloads any diffusion model first)
    manager = ModelManager.get_instance()
    qwen_model, qwen_processor = manager.load_qwen()

    # Build a lightweight pipeline wrapper around the already-loaded model
    pipeline = _QwenWrapper(qwen_model, qwen_processor)

    results: list[dict[str, Any]] = []
    for tile_entry in tiles:
        tile_id: str = tile_entry.get("tile_id", "unknown")
        image_b64: str = tile_entry.get("image_b64", "")

        if not image_b64:
            logger.warning("Tile '%s' has no image data — skipping", tile_id)
            results.append({"tile_id": tile_id, "caption": ""})
            continue

        pil_image = b64_to_pil(image_b64)

        # Optional crop to tile_coords (legacy single-image path)
        tile_coords = tile_entry.get("tile_coords")
        if tile_coords:
            pil_image = crop_to_tile(pil_image, tile_coords)

        logger.info("Captioning tile '%s' (%dx%d)", tile_id, pil_image.width, pil_image.height)
        caption = pipeline.caption(pil_image, system_prompt=system_prompt, max_new_tokens=max_tokens)
        logger.info("Tile '%s' caption: %s", tile_id, caption[:80])

        results.append({"tile_id": tile_id, "caption": caption})

    return {"captions": results}


class _QwenWrapper:
    """
    Thin wrapper that reuses an already-loaded Qwen model/processor pair
    without re-loading from disk.
    """

    def __init__(self, model: Any, processor: Any) -> None:
        self._model = model
        self._processor = processor

    def caption(
        self,
        image: Any,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 200,
    ) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": system_prompt},
                ],
            }
        ]

        text_input = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text_input],
            images=[image],
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)

        input_token_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_token_len:]
        caption: str = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return caption
