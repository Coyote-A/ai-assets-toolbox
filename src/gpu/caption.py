"""
Caption service — Qwen2.5-VL-3B-Instruct on a Modal T4 GPU container.

This module is part of the unified Modal app (``modal.App("ai-toolbox")``).
It does NOT define its own ``modal.App``; instead it imports ``app`` and
``caption_image`` from ``src.app_config`` and registers ``CaptionService``
against that shared app via ``@app.cls()``.

Design decisions vs. the old ``modal/caption_app.py``
------------------------------------------------------
* **No web endpoint** — called via ``.remote()`` from the Gradio UI (also on
  Modal) or from a local machine with the Modal CLI token configured.
* **No auth** — Modal CLI token handles authentication automatically.
* **No base64** — tile images are passed as raw ``bytes`` (PNG/JPEG).  Modal's
  cloudpickle serialisation handles bytes efficiently without the 33 % base64
  size overhead.
* **No RunPod-compatible response wrapper** — exceptions propagate directly to
  the caller; no ``{"status": "FAILED"}`` dicts.

Public interface
----------------
``caption_tiles(tiles, system_prompt, max_tokens) -> dict[str, str]``
    Input:  ``[{"tile_id": "0_0", "image_bytes": <bytes>}, ...]``
    Output: ``{"0_0": "keyword1, keyword2, ..."}``
"""
from __future__ import annotations

import io
import json
import logging
import os
from typing import Optional

import modal

from src.app_config import app, caption_image, models_volume
from src.services.model_registry import MODELS_MOUNT_PATH, get_model_path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "Describe the visual content of this image tile for a Stable Diffusion prompt. "
    "Focus ONLY on objects, elements, and composition - NOT style or aesthetic. "
    "Output ONLY comma-separated keywords, each keyword appears ONCE, no repetitions. "
    "Keep it brief - maximum 10-15 keywords. "
    "NO explanations, NO markdown, NO headers, NO introductory text. "
    "Example output: small pond, blue water, dirt path, grass, rocks, trees"
)

# ---------------------------------------------------------------------------
# CaptionService
# ---------------------------------------------------------------------------


@app.cls(
    # T4 has 16 GB VRAM; Qwen2.5-VL-3B in float16 uses ~6 GB — plenty of room.
    gpu="T4",
    image=caption_image,
    # Keep the container alive for 5 minutes after the last request so
    # subsequent requests skip the model-load overhead.
    scaledown_window=300,
    # Hard timeout per request (10 minutes is generous for batch tile jobs).
    timeout=600,
    # Mount the models volume so weights downloaded by the setup wizard are
    # accessible at MODELS_MOUNT_PATH (/vol/models).
    volumes={MODELS_MOUNT_PATH: models_volume},
)
class CaptionService:
    """
    Qwen2.5-VL-3B-Instruct captioning service.

    Lifecycle
    ---------
    1. Modal starts a container and calls ``load_model()`` via ``@modal.enter()``.
    2. The model stays resident in VRAM for the lifetime of the container.
    3. Each ``.remote()`` call dispatches to the appropriate ``@modal.method()``.
    4. After ``container_idle_timeout`` seconds of inactivity the container is
       torn down; the next call triggers a new cold start.

    Calling from the Gradio UI (or locally)
    ----------------------------------------
    .. code-block:: python

        from src.gpu.caption import CaptionService
        import io
        from PIL import Image

        # Convert PIL tile to bytes
        buf = io.BytesIO()
        pil_tile.save(buf, format="PNG")

        captions = CaptionService().caption_tiles.remote([
            {"tile_id": "0_0", "image_bytes": buf.getvalue()},
        ])
        # captions == {"0_0": "grass, trees, dirt path, ..."}
    """

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    @modal.enter()
    def load_model(self) -> None:
        """
        Load Qwen2.5-VL-3B-Instruct once when the container starts.

        Called exactly once per container instance by Modal before any method
        receives a call.  Loading here means the model is always warm when a
        request arrives.
        """
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        # qwen_vl_utils must be imported before AutoProcessor so that the
        # Qwen VL processor classes are registered with the transformers
        # auto-class registry.  Without this import AutoProcessor raises
        # "if class_name in extractors" KeyError on video_processing_auto.py.
        import qwen_vl_utils  # noqa: F401 — side-effect import

        # Reload the volume to pick up any newly downloaded model weights.
        models_volume.reload()

        model_dir = get_model_path("qwen2.5-vl-3b")

        # Guard: abort gracefully if the model has not been downloaded yet.
        if not os.path.exists(model_dir):
            logger.warning(
                "Caption model not found at %s — run setup wizard first", model_dir
            )
            self._models_ready = False
            return

        # Architecture validation: read config.json and confirm the model type
        # is Qwen2VLForConditionalGeneration (Qwen2.5-VL architecture).
        # If the volume still has Qwen3-VL weights the architectures field will
        # contain "Qwen2_5_VLForConditionalGeneration" or similar Qwen3 class
        # names, which are incompatible with the Qwen2VL loader.
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, encoding="utf-8") as _cfg_fh:
                    _cfg = json.load(_cfg_fh)
                _architectures = _cfg.get("architectures", [])
                _expected = "Qwen2VLForConditionalGeneration"
                if _architectures and _expected not in _architectures:
                    logger.error(
                        "Wrong model architecture at %s: config.json reports %r "
                        "but expected %r. "
                        "The volume likely has stale weights from a different repo. "
                        "Re-run the setup wizard to force a re-download.",
                        model_dir,
                        _architectures,
                        _expected,
                    )
                    self._models_ready = False
                    return
                logger.info(
                    "Architecture check passed: %s", _architectures or "(not specified)"
                )
            except (OSError, json.JSONDecodeError) as _cfg_err:
                logger.warning(
                    "Could not read config.json at %s: %s — proceeding anyway",
                    config_path,
                    _cfg_err,
                )
        else:
            logger.warning(
                "config.json not found at %s — skipping architecture check", model_dir
            )

        self._models_ready = True
        logger.info("Loading Qwen2.5-VL-3B from '%s'", model_dir)

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            model_dir,
            local_files_only=True,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )

        logger.info(
            "Qwen2.5-VL-3B loaded on device: %s",
            next(self._model.parameters()).device,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bytes_to_pil(self, image_bytes: bytes):
        """Decode raw image bytes (PNG/JPEG) to a PIL Image (RGB)."""
        from PIL import Image

        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def _caption_single(
        self,
        image,
        system_prompt: str,
        max_new_tokens: int,
    ) -> str:
        """
        Run Qwen2.5-VL inference on a single PIL image.

        Parameters
        ----------
        image:
            PIL.Image.Image — the tile to caption.
        system_prompt:
            Instruction text prepended to the image in the chat message.
        max_new_tokens:
            Maximum number of tokens to generate.  Kept low (≤300) so the
            model outputs concise keyword lists without padding.

        Returns
        -------
        str
            The generated caption string (stripped of leading/trailing whitespace).
        """
        import torch

        # Build the Qwen2.5-VL chat-format message list
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": system_prompt},
                ],
            }
        ]

        # Convert messages to the model's expected text format
        text_input = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Encode image + text together
        inputs = self._processor(
            text=[text_input],
            images=[image],
            return_tensors="pt",
        )
        # Move all tensors to the same device as the model
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.5,
            )

        # Slice off the input prompt tokens; decode only the new tokens
        input_token_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_token_len:]
        caption: str = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return caption

    # ------------------------------------------------------------------
    # Public Modal methods
    # ------------------------------------------------------------------

    @modal.method()
    def caption_tiles(
        self,
        tiles: list[dict],
        system_prompt: Optional[str] = None,
        max_tokens: int = 100,
    ) -> dict[str, str]:
        """
        Generate captions for one or more image tiles.

        Parameters
        ----------
        tiles:
            List of tile dicts, each with:
            - ``"tile_id"`` (str): unique identifier, e.g. ``"0_0"``.
            - ``"image_bytes"`` (bytes): raw PNG or JPEG bytes of the tile.
        system_prompt:
            Optional custom instruction prompt.  Defaults to
            ``DEFAULT_SYSTEM_PROMPT`` (keyword-list style).
        max_tokens:
            Maximum number of new tokens to generate per tile.

        Returns
        -------
        dict[str, str]
            Mapping of ``tile_id`` → caption string.
            Tiles with missing or empty ``image_bytes`` map to ``""``.

        Raises
        ------
        RuntimeError
            If models have not been downloaded yet.
        ValueError
            If ``tiles`` is empty.

        Example
        -------
        .. code-block:: python

            captions = CaptionService().caption_tiles.remote([
                {"tile_id": "0_0", "image_bytes": png_bytes},
                {"tile_id": "0_1", "image_bytes": png_bytes_2},
            ])
            # {"0_0": "grass, trees, ...", "0_1": "water, rocks, ..."}
        """
        if not getattr(self, "_models_ready", False):
            raise RuntimeError(
                "Models not downloaded. Please run the setup wizard first."
            )

        if not tiles:
            raise ValueError("'tiles' list must not be empty")

        prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        logger.info("caption_tiles: processing %d tile(s)", len(tiles))

        captions: dict[str, str] = {}

        for tile_entry in tiles:
            tile_id: str = tile_entry.get("tile_id", "unknown")
            image_bytes: bytes = tile_entry.get("image_bytes", b"")

            if not image_bytes:
                logger.warning("Tile '%s' has no image data — skipping", tile_id)
                captions[tile_id] = ""
                continue

            pil_image = self._bytes_to_pil(image_bytes)
            logger.info(
                "Captioning tile '%s' (%dx%d)",
                tile_id,
                pil_image.width,
                pil_image.height,
            )

            caption = self._caption_single(
                pil_image,
                system_prompt=prompt,
                max_new_tokens=max_tokens,
            )
            logger.info("Tile '%s' → %s", tile_id, caption[:80])
            captions[tile_id] = caption

        logger.info("caption_tiles: completed %d tile(s)", len(captions))
        return captions

    @modal.method()
    def health(self) -> dict:
        """
        Return GPU and model health information.

        Returns
        -------
        dict
            Keys: ``status``, ``gpu``, ``vram_total_gb``, ``vram_used_gb``,
            ``model_loaded``, ``model_path``, ``models_ready``.
        """
        import torch

        gpu_name = "none"
        vram_total_gb = 0.0
        vram_used_gb = 0.0

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_total_gb = round(props.total_memory / (1024 ** 3), 2)
            free_bytes, _ = torch.cuda.mem_get_info(0)
            vram_used_gb = round((props.total_memory - free_bytes) / (1024 ** 3), 2)

        model_loaded = hasattr(self, "_model") and self._model is not None
        model_path = get_model_path("qwen2.5-vl-3b")

        return {
            "status": "ok",
            "gpu": gpu_name,
            "vram_total_gb": vram_total_gb,
            "vram_used_gb": vram_used_gb,
            "model_loaded": model_loaded,
            "model_path": model_path,
            "models_ready": getattr(self, "_models_ready", False),
        }
