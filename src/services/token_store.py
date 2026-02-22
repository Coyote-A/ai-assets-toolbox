"""
Token storage service using Modal Volume for server-side persistence.

This module provides a server-side token storage solution that persists
across server restarts. Tokens are stored in a JSON file on a persistent
Modal volume.

Architecture
------------
- Tokens are stored in a JSON file on the lora_volume
- No session ID needed - single user deployment
- On page load, tokens are read from the JSON file
- When tokens are saved, they're written to the JSON file

Security
--------
- Tokens are stored server-side, never exposed to localStorage
- The JSON file is on a persistent Modal volume
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Path to the tokens file on the lora volume
# LORAS_MOUNT_PATH is /vol/loras
TOKENS_FILE_PATH = "/vol/loras/.api_tokens.json"


@dataclass
class TokenData:
    """Container for API tokens."""

    hf_token: Optional[str] = None
    civitai_token: Optional[str] = None


def _ensure_tokens_file() -> None:
    """Ensure the tokens file exists with default structure."""
    os.makedirs(os.path.dirname(TOKENS_FILE_PATH), exist_ok=True)
    if not os.path.exists(TOKENS_FILE_PATH):
        with open(TOKENS_FILE_PATH, "w") as f:
            json.dump({"hf_token": None, "civitai_token": None}, f)


def get_tokens() -> TokenData:
    """
    Retrieve tokens from the JSON file.

    Returns:
        TokenData with hf_token and civitai_token (may be None)
    """
    try:
        _ensure_tokens_file()
        with open(TOKENS_FILE_PATH, "r") as f:
            data = json.load(f)

        logger.debug(
            "get_tokens: retrieved tokens (hf=%s, civitai=%s)",
            "set" if data.get("hf_token") else "empty",
            "set" if data.get("civitai_token") else "empty",
        )
        return TokenData(
            hf_token=data.get("hf_token"),
            civitai_token=data.get("civitai_token"),
        )
    except Exception:
        logger.exception("get_tokens: error retrieving tokens")
        return TokenData()


def save_tokens(
    hf_token: Optional[str] = None,
    civitai_token: Optional[str] = None,
) -> None:
    """
    Save tokens to the JSON file.

    Args:
        hf_token: HuggingFace API token (optional)
        civitai_token: CivitAI API token (optional)
    """
    try:
        _ensure_tokens_file()

        # Read existing data
        with open(TOKENS_FILE_PATH, "r") as f:
            data = json.load(f)

        # Update only provided tokens
        if hf_token is not None:
            data["hf_token"] = hf_token
        if civitai_token is not None:
            data["civitai_token"] = civitai_token

        # Write back
        with open(TOKENS_FILE_PATH, "w") as f:
            json.dump(data, f)

        logger.info(
            "save_tokens: saved tokens (hf=%s, civitai=%s)",
            "set" if hf_token else "unchanged",
            "set" if civitai_token else "unchanged",
        )
    except Exception:
        logger.exception("save_tokens: error saving tokens")


def delete_tokens() -> None:
    """Delete all tokens from the JSON file."""
    try:
        _ensure_tokens_file()
        with open(TOKENS_FILE_PATH, "w") as f:
            json.dump({"hf_token": None, "civitai_token": None}, f)
        logger.info("delete_tokens: deleted all tokens")
    except Exception:
        logger.exception("delete_tokens: error deleting tokens")
