"""
Token storage service using Modal Dict for server-side persistence.

This module provides a server-side token storage solution that persists
across server restarts. Tokens are stored in a Modal Dict, which provides:
- No reload needed: Dict is always current across all containers
- Atomic updates: Built-in concurrency control
- Faster access: In-memory reads
- Simpler code: No file I/O handling

Architecture
------------
- Tokens are stored in a Modal Dict (ai-toolbox-tokens)
- No session ID needed - single user deployment
- On page load, tokens are read from the Dict
- When tokens are saved, they're written to the Dict

Security
--------
- Tokens are stored server-side, never exposed to localStorage
- The Dict is managed by Modal with built-in persistence
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.app_config import token_store

logger = logging.getLogger(__name__)

# Key within the Dict for storing tokens
TOKENS_KEY = "tokens"


@dataclass
class TokenData:
    """Container for API tokens."""

    hf_token: Optional[str] = None
    civitai_token: Optional[str] = None


def get_tokens() -> TokenData:
    """
    Retrieve tokens from the Modal Dict.

    Returns:
        TokenData with hf_token and civitai_token (may be None)
    """
    try:
        data = token_store.get(TOKENS_KEY, {})

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
    Save tokens to the Modal Dict.

    Args:
        hf_token: HuggingFace API token (optional)
        civitai_token: CivitAI API token (optional)
    """
    try:
        # Read existing data
        data = token_store.get(TOKENS_KEY, {})

        # Update only provided tokens
        if hf_token is not None:
            data["hf_token"] = hf_token
        if civitai_token is not None:
            data["civitai_token"] = civitai_token

        # Write back to Dict
        token_store[TOKENS_KEY] = data

        logger.info(
            "save_tokens: saved tokens (hf=%s, civitai=%s)",
            "set" if hf_token else "unchanged",
            "set" if civitai_token else "unchanged",
        )
    except Exception:
        logger.exception("save_tokens: error saving tokens")


def delete_tokens() -> None:
    """Delete all tokens from the Modal Dict."""
    try:
        token_store[TOKENS_KEY] = {"hf_token": None, "civitai_token": None}
        logger.info("delete_tokens: deleted all tokens")
    except Exception:
        logger.exception("delete_tokens: error deleting tokens")
