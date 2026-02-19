"""
Image utility functions.

Provides base64 encode/decode helpers and image transformation utilities
used throughout the backend.
"""

import base64
import io
import logging
from typing import Any, Optional

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base64 encode / decode
# ---------------------------------------------------------------------------

def b64_to_pil(b64_string: str) -> Image.Image:
    """
    Decode a base64-encoded image string to a PIL Image.

    Parameters
    ----------
    b64_string:
        Base64-encoded image data.  May optionally include a data-URI prefix
        (e.g. ``"data:image/png;base64,..."``).

    Returns
    -------
    PIL.Image.Image
        Decoded image in RGB mode.
    """
    # Strip data-URI prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    image_bytes = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    logger.debug("Decoded base64 image: %dx%d", image.width, image.height)
    return image


def pil_to_b64(image: Image.Image, format: str = "PNG") -> str:
    """
    Encode a PIL Image to a base64 string.

    Parameters
    ----------
    image:
        PIL Image to encode.
    format:
        Image format for encoding (``"PNG"`` or ``"JPEG"``).

    Returns
    -------
    str
        Base64-encoded image string (no data-URI prefix).
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
    logger.debug("Encoded PIL image to base64 (%d chars)", len(b64_string))
    return b64_string


def bytes_to_b64(data: bytes) -> str:
    """Encode raw bytes to a base64 string."""
    return base64.b64encode(data).decode("utf-8")


def b64_to_bytes(b64_string: str) -> bytes:
    """Decode a base64 string to raw bytes."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    return base64.b64decode(b64_string)


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

def crop_to_tile(
    image: Image.Image,
    tile_coords: dict[str, int],
) -> Image.Image:
    """
    Crop an image to the specified tile coordinates.

    Parameters
    ----------
    image:
        Source PIL Image.
    tile_coords:
        Dict with keys ``x``, ``y``, ``w``, ``h`` (all in pixels).

    Returns
    -------
    PIL.Image.Image
        Cropped image region.
    """
    x: int = int(tile_coords["x"])
    y: int = int(tile_coords["y"])
    w: int = int(tile_coords["w"])
    h: int = int(tile_coords["h"])

    cropped = image.crop((x, y, x + w, y + h))
    logger.debug("Cropped image to tile (%d,%d,%d,%d) → %dx%d", x, y, w, h, cropped.width, cropped.height)
    return cropped


def resize_image(
    image: Image.Image,
    width: int,
    height: int,
    resample: int = Image.LANCZOS,
) -> Image.Image:
    """
    Resize a PIL Image to the given dimensions.

    Parameters
    ----------
    image:
        Source PIL Image.
    width:
        Target width in pixels.
    height:
        Target height in pixels.
    resample:
        Resampling filter.  Defaults to ``Image.LANCZOS``.

    Returns
    -------
    PIL.Image.Image
        Resized image.
    """
    resized = image.resize((width, height), resample=resample)
    logger.debug("Resized image from %dx%d to %dx%d", image.width, image.height, width, height)
    return resized


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert image to RGB mode if it is not already."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def pad_to_multiple(image: Image.Image, multiple: int = 64) -> Image.Image:
    """
    Pad an image so that both dimensions are multiples of ``multiple``.

    This is sometimes required by diffusion models that expect dimensions
    divisible by 8 or 64.

    Parameters
    ----------
    image:
        Source PIL Image.
    multiple:
        The target multiple (default 64).

    Returns
    -------
    PIL.Image.Image
        Padded image (black padding on right/bottom edges).
    """
    w, h = image.size
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple

    if new_w == w and new_h == h:
        return image

    padded = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    padded.paste(image, (0, 0))
    logger.debug("Padded image from %dx%d to %dx%d", w, h, new_w, new_h)
    return padded
