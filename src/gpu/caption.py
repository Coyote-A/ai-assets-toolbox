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

import hashlib
import io
import json
import logging
import os
import re
from typing import Optional

import modal

from src.app_config import app, caption_image, metadata_extraction_cache, models_volume
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
    "List the main objects in this image as a short comma-separated list. "
    "Maximum 8-10 keywords. Each keyword appears ONCE. "
    "NO duplicate words, NO consecutive commas, NO explanations, NO markdown. "
    "Example: pond, stones, green meadow, mushrooms, wildflowers, wooden fence"
)

# ---------------------------------------------------------------------------
# Metadata extraction prompts
# ---------------------------------------------------------------------------

METADATA_EXTRACTION_SYSTEM_PROMPT = """You are a metadata extraction assistant for AI image generation models (LoRAs).
Extract structured metadata from model descriptions. Be conservative - only extract information that is explicitly stated."""

METADATA_EXTRACTION_USER_PROMPT = """Extract metadata from this model description. Return JSON with these fields:
- trigger_words: list of strings (activation phrases, e.g. ["style of artist", "in the style of"])
- recommended_weight: float or null (typical usage weight, e.g. 0.8)
- recommended_weight_min: float or null (if a range is mentioned)
- recommended_weight_max: float or null (if a range is mentioned)
- clip_skip: int or null (if specified, usually 1-2)
- tags: list of strings (style/category keywords)
- usage_notes: string or null (any special instructions)

Only include fields that are explicitly mentioned. Use null for missing values.

Model: {model_name}
Description:
{description}

Return only valid JSON, no other text."""

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
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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
        # is a known Qwen2-VL variant.  We accept both:
        #   - "Qwen2VLForConditionalGeneration"   (Qwen2-VL)
        #   - "Qwen2_5_VLForConditionalGeneration" (Qwen2.5-VL, note the underscore)
        # Any other architecture indicates stale/wrong weights on the volume.
        _ACCEPTED_ARCHITECTURES = {
            "Qwen2VLForConditionalGeneration",
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2_5VLForConditionalGeneration",  # possible variant
        }
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, encoding="utf-8") as _cfg_fh:
                    _cfg = json.load(_cfg_fh)
                _architectures = _cfg.get("architectures", [])
                if _architectures and not _ACCEPTED_ARCHITECTURES.intersection(_architectures):
                    logger.error(
                        "Wrong model architecture at %s: config.json reports %r "
                        "but expected one of %r. "
                        "The volume likely has stale weights from a different repo. "
                        "Re-run the setup wizard to force a re-download.",
                        model_dir,
                        _architectures,
                        sorted(_ACCEPTED_ARCHITECTURES),
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

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            dtype=torch.float16,
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

    @staticmethod
    def _clean_caption(caption: str, max_tokens: int = 60) -> str:
        """
        Post-process a raw model caption to remove artefacts and enforce length.

        Steps
        -----
        1. Replace runs of commas (with optional surrounding whitespace) with a
           single ``", "`` separator.
        2. Split on commas, strip each keyword, drop empty strings.
        3. Re-join with ``", "``.
        4. Truncate to *max_tokens* whitespace-split tokens so the result stays
           within CLIP's 77-token limit (leaving headroom for any prefix tokens
           added by the CLIP tokenizer).

        Parameters
        ----------
        caption:
            Raw string from the model.
        max_tokens:
            Maximum number of space-separated tokens to keep.  Defaults to 60,
            which comfortably fits inside CLIP's 77-token window.

        Returns
        -------
        str
            Cleaned, length-bounded caption.
        """
        import re

        # Collapse runs of commas / whitespace-comma-whitespace into one comma
        caption = re.sub(r"[\s,]*,[\s,]*", ", ", caption)

        # Split, strip, and drop empty keywords
        keywords = [kw.strip() for kw in caption.split(",")]
        keywords = [kw for kw in keywords if kw]

        # Re-join
        cleaned = ", ".join(keywords)

        # Truncate to max_tokens whitespace tokens
        tokens = cleaned.split()
        if len(tokens) > max_tokens:
            cleaned = " ".join(tokens[:max_tokens])
            # If we cut mid-keyword-phrase, strip a trailing comma
            cleaned = cleaned.rstrip(", ")

        return cleaned

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
            Maximum number of tokens to generate.  Kept low (≤75) so the
            model outputs concise keyword lists without padding.

        Returns
        -------
        str
            The generated caption string (stripped of leading/trailing whitespace).
        """
        import torch

        # Build the Qwen2.5-VL chat-format message list.
        # A dedicated "system" role message improves instruction-following for
        # brevity constraints compared to embedding the instruction only in the
        # user turn.
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe the main objects in this image."},
                ],
            },
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
        max_tokens: int = 50,
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

            raw_caption = self._caption_single(
                pil_image,
                system_prompt=prompt,
                max_new_tokens=max_tokens,
            )
            caption = self._clean_caption(raw_caption)
            logger.info(
                "Tile '%s' → raw=%r  cleaned=%r",
                tile_id,
                raw_caption[:120],
                caption,
            )
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

    # ------------------------------------------------------------------
    # Metadata extraction methods
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(description: str) -> str:
        """Generate cache key from description hash."""
        return hashlib.sha256(description.encode()).hexdigest()[:16]

    @staticmethod
    def _parse_llm_json(response: str) -> dict:
        """
        Parse JSON from LLM response, handling markdown code blocks.

        Tries multiple strategies:
        1. Direct JSON parse
        2. Extract from markdown code block (```json ... ```)
        3. Find JSON object in text

        Returns empty dict if all strategies fail.
        """
        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object in text
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse LLM response as JSON: %s", response[:200])
        return {}

    def _extract_text_only(
        self,
        text: str,
        system_prompt: str,
        max_new_tokens: int = 500,
    ) -> str:
        """
        Run text-only inference (no image) using Qwen2.5-VL.

        Qwen2.5-VL supports text-only input by omitting the image in the message.

        Parameters
        ----------
        text:
            The user prompt text.
        system_prompt:
            System instruction for the model.
        max_new_tokens:
            Maximum tokens to generate.

        Returns
        -------
        str
            Generated text response.
        """
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        text_input = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[text_input],
            images=None,  # No images for text-only inference
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_token_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_token_len:]
        return self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

    @modal.method()
    def extract_metadata(
        self,
        description: str,
        model_name: str = "unknown",
    ) -> dict:
        """
        Extract structured metadata from a model description using LLM.

        This method uses the Qwen2.5-VL model to parse model descriptions
        (typically from CivitAI or HuggingFace) and extract structured metadata
        like trigger words, recommended weights, CLIP skip, etc.

        Results are cached in a Modal Dict to avoid repeated LLM calls for
        the same description.

        Parameters
        ----------
        description:
            The model description text to parse.
        model_name:
            Optional model name for context (improves extraction quality).

        Returns
        -------
        dict
            Extracted metadata fields. Empty dict if extraction fails.
            Fields may include:
            - trigger_words: list of strings
            - recommended_weight: float or None
            - recommended_weight_min: float or None
            - recommended_weight_max: float or None
            - clip_skip: int or None
            - tags: list of strings
            - usage_notes: string or None

        Raises
        ------
        RuntimeError
            If models have not been downloaded yet.
        """
        if not getattr(self, "_models_ready", False):
            raise RuntimeError(
                "Models not downloaded. Please run the setup wizard first."
            )

        if not description or not description.strip():
            return {}

        # Check cache first
        cache_key = self._cache_key(description)
        try:
            if cache_key in metadata_extraction_cache:
                logger.info("Cache hit for metadata extraction: %s", cache_key)
                return metadata_extraction_cache[cache_key]
        except Exception as e:
            logger.warning("Failed to check cache: %s", e)

        # Build the prompt
        user_prompt = METADATA_EXTRACTION_USER_PROMPT.format(
            model_name=model_name,
            description=description,
        )

        logger.info(
            "Extracting metadata for model '%s' (description length: %d)",
            model_name,
            len(description),
        )

        # Run LLM extraction
        try:
            raw_response = self._extract_text_only(
                text=user_prompt,
                system_prompt=METADATA_EXTRACTION_SYSTEM_PROMPT,
                max_new_tokens=500,
            )
            logger.info("LLM raw response: %s", raw_response[:200])

            # Parse JSON response
            result = self._parse_llm_json(raw_response)

            if not result:
                logger.warning("Empty result from LLM for description")
                return {}

            # Validate and clean the result
            cleaned_result = self._validate_metadata_result(result)

            # Cache the result
            try:
                metadata_extraction_cache[cache_key] = cleaned_result
                logger.info("Cached metadata extraction result: %s", cache_key)
            except Exception as e:
                logger.warning("Failed to cache result: %s", e)

            return cleaned_result

        except Exception as e:
            logger.error("Metadata extraction failed: %s", e)
            return {}

    def _validate_metadata_result(self, result: dict) -> dict:
        """
        Validate and clean the extracted metadata result.

        Ensures fields have the correct types and removes invalid entries.
        """
        cleaned = {}

        # trigger_words: should be a list of strings
        if "trigger_words" in result and result["trigger_words"] is not None:
            triggers = result["trigger_words"]
            if isinstance(triggers, list):
                cleaned["trigger_words"] = [
                    str(t) for t in triggers if t is not None
                ]
            elif isinstance(triggers, str):
                # Handle case where LLM returns a single string
                cleaned["trigger_words"] = [triggers]

        # recommended_weight: should be a float
        if "recommended_weight" in result and result["recommended_weight"] is not None:
            try:
                weight = float(result["recommended_weight"])
                if 0.0 <= weight <= 2.0:  # Sanity check
                    cleaned["recommended_weight"] = weight
            except (ValueError, TypeError):
                pass

        # recommended_weight_min/max: should be floats
        for field in ["recommended_weight_min", "recommended_weight_max"]:
            if field in result and result[field] is not None:
                try:
                    weight = float(result[field])
                    if 0.0 <= weight <= 2.0:
                        cleaned[field] = weight
                except (ValueError, TypeError):
                    pass

        # clip_skip: should be an int (typically 1-2)
        if "clip_skip" in result and result["clip_skip"] is not None:
            try:
                clip_skip = int(result["clip_skip"])
                if 1 <= clip_skip <= 12:  # Sanity check
                    cleaned["clip_skip"] = clip_skip
            except (ValueError, TypeError):
                pass

        # tags: should be a list of strings
        if "tags" in result and result["tags"] is not None:
            tags = result["tags"]
            if isinstance(tags, list):
                cleaned["tags"] = [str(t) for t in tags if t is not None]
            elif isinstance(tags, str):
                cleaned["tags"] = [tags]

        # usage_notes: should be a string
        if "usage_notes" in result and result["usage_notes"] is not None:
            notes = result["usage_notes"]
            if isinstance(notes, str) and notes.strip():
                cleaned["usage_notes"] = notes.strip()

        return cleaned
