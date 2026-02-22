"""CivitAI API integration for model metadata and downloads."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

CIVITAI_API_BASE = "https://civitai.com/api/v1"


@dataclass
class CivitAIModelInfo:
    """Parsed model info from CivitAI API."""
    model_id: int
    version_id: int
    name: str
    version_name: str
    model_type: str          # "lora", "checkpoint", etc.
    base_model: str          # "Illustrious", "SDXL 1.0", "Pony", etc.
    trigger_words: list[str]
    description: str
    tags: list[str]
    download_url: str
    filename: str
    size_bytes: int
    # Recommended settings from metadata
    recommended_weight: float | None = None
    nsfw: bool = False


def parse_civitai_input(user_input: str) -> tuple[int | None, int | None]:
    """
    Parse user input to extract CivitAI model ID and/or version ID.

    Supports:

    - Full URL: ``https://civitai.com/models/929497/aesthetic-quality``
    - Full URL with version: ``https://civitai.com/models/929497?modelVersionId=1234567``
    - Model ID: ``929497``
    - Version ID with prefix: ``v:1234567`` or ``version:1234567``

    Returns:
        ``(model_id, version_id)`` — one or both may be ``None``.
    """
    user_input = user_input.strip()

    # Try URL patterns
    url_match = re.match(r'https?://civitai\.com/models/(\d+)', user_input)
    if url_match:
        model_id = int(url_match.group(1))
        # Check for version in query params
        version_match = re.search(r'modelVersionId=(\d+)', user_input)
        version_id = int(version_match.group(1)) if version_match else None
        return model_id, version_id

    # Try version prefix
    version_prefix = re.match(r'(?:v|version):(\d+)', user_input, re.IGNORECASE)
    if version_prefix:
        return None, int(version_prefix.group(1))

    # Try plain number (model ID)
    if user_input.isdigit():
        return int(user_input), None

    return None, None


def fetch_model_info(
    model_id: int | None = None,
    version_id: int | None = None,
    api_token: str | None = None,
) -> Optional[CivitAIModelInfo]:
    """
    Fetch model info from CivitAI API.

    If *version_id* is provided, fetches that specific version.
    If only *model_id* is provided, fetches the latest version.

    Args:
        model_id:   CivitAI model ID (optional if *version_id* is given).
        version_id: CivitAI model-version ID (optional if *model_id* is given).
        api_token:  CivitAI API token for authenticated requests (optional for
                    metadata-only queries, required for gated downloads).

    Returns:
        A :class:`CivitAIModelInfo` instance, or ``None`` on failure.
    """
    import requests  # lazy import — only available in download_image
    headers: dict[str, str] = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    try:
        if version_id:
            # Fetch specific version
            resp = requests.get(
                f"{CIVITAI_API_BASE}/model-versions/{version_id}",
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            version_data = resp.json()
            model_id = version_data.get("modelId", model_id)
            # Fetch parent model for tags / top-level name
            model_data = None
            if model_id:
                try:
                    model_resp = requests.get(
                        f"{CIVITAI_API_BASE}/models/{model_id}",
                        headers=headers,
                        timeout=30,
                    )
                    model_resp.raise_for_status()
                    model_data = model_resp.json()
                except Exception:
                    pass
        elif model_id:
            # Fetch model, get latest version
            resp = requests.get(
                f"{CIVITAI_API_BASE}/models/{model_id}",
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            model_data = resp.json()
            versions = model_data.get("modelVersions", [])
            if not versions:
                logger.warning("No versions found for model %d", model_id)
                return None
            version_data = versions[0]  # Latest version
        else:
            logger.error("Neither model_id nor version_id provided")
            return None

        # Resolve version_id from the fetched version data
        version_id = version_data.get("id")

        # ------------------------------------------------------------------
        # Primary downloadable file
        # ------------------------------------------------------------------
        files = version_data.get("files", [])
        primary_file = None
        for f in files:
            if f.get("primary", False) or f.get("type") == "Model":
                primary_file = f
                break
        if not primary_file and files:
            primary_file = files[0]

        if not primary_file:
            logger.warning("No downloadable file found for version %d", version_id)
            return None

        # ------------------------------------------------------------------
        # Determine internal model type
        # ------------------------------------------------------------------
        raw_type = (model_data or version_data).get("type", "").lower()
        if "lora" in raw_type or "loha" in raw_type or "locon" in raw_type:
            model_type = "lora"
        elif "checkpoint" in raw_type:
            model_type = "checkpoint"
        else:
            model_type = raw_type or "unknown"

        # ------------------------------------------------------------------
        # Trigger words
        # ------------------------------------------------------------------
        trigger_words: list[str] = version_data.get("trainedWords", [])

        # ------------------------------------------------------------------
        # Tags (CivitAI returns either a list of dicts or a list of strings)
        # ------------------------------------------------------------------
        tags: list[str] = []
        if model_data:
            raw_tags = model_data.get("tags", [])
            if raw_tags and isinstance(raw_tags[0], dict):
                tags = [t.get("name", "") for t in raw_tags]
            else:
                tags = [t for t in raw_tags if isinstance(t, str)]

        # ------------------------------------------------------------------
        # Build download URL
        # ------------------------------------------------------------------
        download_url = f"{CIVITAI_API_BASE}/model-versions/{version_id}/download"

        # ------------------------------------------------------------------
        # File size: CivitAI reports sizeKB as a float
        # ------------------------------------------------------------------
        size_kb = primary_file.get("sizeKB", 0) or 0
        size_bytes = int(size_kb * 1024)

        return CivitAIModelInfo(
            model_id=model_id or 0,
            version_id=version_id,
            name=(model_data or version_data).get("name", f"Model {model_id}"),
            version_name=version_data.get("name", ""),
            model_type=model_type,
            base_model=version_data.get("baseModel", ""),
            trigger_words=trigger_words,
            description=((model_data or version_data).get("description") or "")[:500],
            tags=tags,
            download_url=download_url,
            filename=primary_file.get("name", f"model_{version_id}.safetensors"),
            size_bytes=size_bytes,
            recommended_weight=version_data.get("recommendedWeight"),
            nsfw=(model_data or version_data).get("nsfw", False),
        )

    except requests.RequestException as e:
        logger.error("CivitAI API error: %s", e)
        return None
    except (KeyError, IndexError, TypeError) as e:
        logger.error("Failed to parse CivitAI response: %s", e)
        return None
