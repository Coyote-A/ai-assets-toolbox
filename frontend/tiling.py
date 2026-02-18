"""
Client-side tile processing utilities for the AI Assets Toolbox frontend.

All tiling and blending operations run locally; only individual tiles are
sent to the RunPod backend for diffusion processing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TileInfo:
    """Describes a single tile's position within the full image."""
    x: int          # left edge in the full image (pixels)
    y: int          # top edge in the full image (pixels)
    w: int          # tile width (pixels)
    h: int          # tile height (pixels)
    row: int        # row index in the grid (0-based)
    col: int        # column index in the grid (0-based)

    @property
    def tile_id(self) -> str:
        return f"{self.row}_{self.col}"

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


@dataclass
class RegionInfo:
    """Describes a user-selected region for targeted upscaling."""
    x: int          # left edge in the full image (pixels)
    y: int          # top edge in the full image (pixels)
    w: int          # region width (pixels)
    h: int          # region height (pixels)
    padding: int    # padding around the region in pixels
    prompt: str     # prompt for this region
    negative_prompt: str  # negative prompt for this region

    @property
    def region_id(self) -> str:
        return f"region_{self.x}_{self.y}_{self.w}_{self.h}"

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def to_dict(self) -> dict:
        """Convert to dictionary for API payload."""
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "padding": self.padding,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
        }


# ---------------------------------------------------------------------------
# Grid calculation
# ---------------------------------------------------------------------------

def calculate_tiles(
    image_size: Tuple[int, int],
    tile_size: int = 1024,
    overlap: int = 128,
) -> List[TileInfo]:
    """
    Calculate the tile grid for an image.

    The image is first assumed to already be at the target resolution.
    Tiles are placed with the given overlap so that adjacent tiles share
    `overlap` pixels on each shared edge.

    Args:
        image_size: (width, height) of the image in pixels.
        tile_size:  size of each square tile (default 1024).
        overlap:    number of pixels shared between adjacent tiles (default 128).

    Returns:
        Ordered list of TileInfo objects (row-major order).
    """
    width, height = image_size
    stride = tile_size - overlap

    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be less than tile_size ({tile_size})")

    # Number of tiles needed to cover the image
    num_cols = max(1, math.ceil((width - overlap) / stride))
    num_rows = max(1, math.ceil((height - overlap) / stride))

    tiles: List[TileInfo] = []
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * stride
            y = row * stride

            # Clamp to image boundary
            x = min(x, max(0, width - tile_size))
            y = min(y, max(0, height - tile_size))

            w = min(tile_size, width - x)
            h = min(tile_size, height - y)

            tiles.append(TileInfo(x=x, y=y, w=w, h=h, row=row, col=col))

    return tiles


# ---------------------------------------------------------------------------
# Tile extraction
# ---------------------------------------------------------------------------

def extract_tile(image: Image.Image, tile_info: TileInfo) -> Image.Image:
    """
    Crop a single tile from the image.

    Args:
        image:     source PIL Image.
        tile_info: TileInfo describing the crop region.

    Returns:
        Cropped PIL Image (may be smaller than tile_size at edges).
    """
    return image.crop((tile_info.x, tile_info.y, tile_info.x2, tile_info.y2))


def extract_all_tiles(
    image: Image.Image,
    tile_size: int = 1024,
    overlap: int = 128,
) -> List[Tuple[TileInfo, Image.Image]]:
    """
    Extract all tiles from an image.

    Returns:
        List of (TileInfo, PIL Image) pairs in row-major order.
    """
    tiles = calculate_tiles(image.size, tile_size=tile_size, overlap=overlap)
    return [(t, extract_tile(image, t)) for t in tiles]


def extract_region_image(
    image: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    padding: int = 0,
) -> Image.Image:
    """
    Crop a region from the image with optional padding and bounds clamping.

    Args:
        image:   source PIL Image.
        x, y:   top-left corner of the region.
        w, h:   width/height of the region.
        padding: extra pixels around the region (applied on all sides).

    Returns:
        Cropped PIL Image.
    """
    img_w, img_h = image.size
    pad = max(0, int(padding))

    left = max(0, int(x) - pad)
    top = max(0, int(y) - pad)
    right = min(img_w, int(x) + int(w) + pad)
    bottom = min(img_h, int(y) + int(h) + pad)

    if right < left:
        right = left
    if bottom < top:
        bottom = top

    return image.crop((left, top, right, bottom))


# ---------------------------------------------------------------------------
# Tile blending / reassembly
# ---------------------------------------------------------------------------

def _make_weight_map(h: int, w: int, overlap: int, tile_info: TileInfo, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Build a 2-D float32 weight map for a tile using linear gradient feathering
    in the overlap regions on each edge that is shared with a neighbour.

    The weight ramps from 0 → 1 over the overlap zone on each shared edge.
    Edges that touch the image boundary keep weight = 1 (no feathering needed).
    """
    img_w, img_h = image_size
    weight = np.ones((h, w), dtype=np.float32)

    # Left edge feathering (if not at image left boundary)
    if tile_info.x > 0:
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        weight[:, :overlap] *= ramp[np.newaxis, :]

    # Right edge feathering (if not at image right boundary)
    if tile_info.x2 < img_w:
        ramp = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
        weight[:, w - overlap:] *= ramp[np.newaxis, :]

    # Top edge feathering (if not at image top boundary)
    if tile_info.y > 0:
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        weight[:overlap, :] *= ramp[:, np.newaxis]

    # Bottom edge feathering (if not at image bottom boundary)
    if tile_info.y2 < img_h:
        ramp = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
        weight[h - overlap:, :] *= ramp[:, np.newaxis]

    return weight


def blend_tiles(
    tiles: List[Tuple[TileInfo, Image.Image]],
    image_size: Tuple[int, int],
    tile_size: int = 1024,
    overlap: int = 128,
) -> Image.Image:
    """
    Reassemble processed tiles into a full image using linear gradient feathering.

    Blending is performed in linear (float32) colour space to avoid gamma
    artefacts, then converted back to uint8 for the output image.

    Args:
        tiles:      list of (TileInfo, PIL Image) pairs.
        image_size: (width, height) of the target output image.
        tile_size:  tile size used during extraction.
        overlap:    overlap used during extraction.

    Returns:
        Reconstructed PIL Image.
    """
    img_w, img_h = image_size
    accumulator = np.zeros((img_h, img_w, 3), dtype=np.float32)
    weight_sum = np.zeros((img_h, img_w, 1), dtype=np.float32)

    for tile_info, tile_img in tiles:
        # Ensure tile is RGB
        tile_rgb = tile_img.convert("RGB")
        tile_arr = np.array(tile_rgb, dtype=np.float32)  # (h, w, 3)

        h, w = tile_arr.shape[:2]
        weight = _make_weight_map(h, w, overlap, tile_info, image_size)  # (h, w)

        y1, y2 = tile_info.y, tile_info.y + h
        x1, x2 = tile_info.x, tile_info.x + w

        accumulator[y1:y2, x1:x2] += tile_arr * weight[:, :, np.newaxis]
        weight_sum[y1:y2, x1:x2] += weight[:, :, np.newaxis]

    # Avoid division by zero for pixels not covered by any tile
    weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)
    result_arr = np.clip(accumulator / weight_sum, 0, 255).astype(np.uint8)
    return Image.fromarray(result_arr, mode="RGB")


# ---------------------------------------------------------------------------
# Grid overlay visualisation
# ---------------------------------------------------------------------------

def draw_tile_grid(
    image: Image.Image,
    tiles: List[TileInfo],
    selected_indices: Optional[List[int]] = None,
    processed_indices: Optional[List[int]] = None,
) -> Image.Image:
    """
    Draw a tile grid overlay on a copy of the image.

    Args:
        image:             source PIL Image.
        tiles:             list of TileInfo objects.
        selected_indices:  tile indices to highlight in blue.
        processed_indices: tile indices to highlight in green.

    Returns:
        New PIL Image with the grid drawn on top.
    """
    overlay = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    selected_set = set(selected_indices or [])
    processed_set = set(processed_indices or [])

    for idx, tile in enumerate(tiles):
        rect = [tile.x, tile.y, tile.x2 - 1, tile.y2 - 1]

        if idx in selected_set:
            fill_color = (0, 100, 255, 60)
            outline_color = (0, 100, 255, 220)
            outline_width = 3
        elif idx in processed_set:
            fill_color = (0, 200, 80, 50)
            outline_color = (0, 200, 80, 200)
            outline_width = 2
        else:
            fill_color = (255, 255, 255, 20)
            outline_color = (200, 200, 200, 160)
            outline_width = 1

        draw.rectangle(rect, fill=fill_color, outline=outline_color, width=outline_width)

        # Draw tile index label in the centre
        cx = tile.x + tile.w // 2
        cy = tile.y + tile.h // 2
        label = str(idx)
        draw.text((cx - 8, cy - 8), label, fill=(255, 255, 255, 220))

    return overlay.convert("RGB")


# ---------------------------------------------------------------------------
# Upscale helper
# ---------------------------------------------------------------------------

def upscale_image(
    image: Image.Image,
    target_width: int,
    target_height: int,
) -> Image.Image:
    """
    Bicubic upscale of an image to the target resolution.

    Args:
        image:         source PIL Image.
        target_width:  desired output width in pixels.
        target_height: desired output height in pixels.

    Returns:
        Upscaled PIL Image.
    """
    return image.resize((target_width, target_height), Image.BICUBIC)


# ---------------------------------------------------------------------------
# Region overlay visualisation
# ---------------------------------------------------------------------------

def draw_region_overlay(
    image: Image.Image,
    regions: List[dict],
    selected_index: Optional[int] = None,
) -> Image.Image:
    """
    Draw region overlays on a copy of the image.

    Args:
    
    image:           source PIL Image.
    regions:         list of region dicts with x, y, w, h, padding.
    selected_index:  index of the currently selected region to highlight.

    Returns:
        New PIL Image with the regions drawn on top.
    """
    overlay = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    for idx, region in enumerate(regions):
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)
        padding = region.get("padding", 0)

        # Draw padding region (lighter)
        if padding > 0:
            pad_rect = [
                x - padding,
                y - padding,
                x + w + padding,
                y + h + padding,
            ]
            draw.rectangle(pad_rect, fill=(255, 200, 0, 30), outline=(255, 200, 0, 150), width=2)

        # Draw main region rectangle
        rect = [x, y, x + w, y + h]
        if idx == selected_index:
            fill_color = (0, 100, 255, 60)
            outline_color = (0, 100, 255, 220)
            outline_width = 3
        else:
            fill_color = (255, 100, 0, 50)
            outline_color = (255, 100, 0, 200)
            outline_width = 2

        draw.rectangle(rect, fill=fill_color, outline=outline_color, width=outline_width)

        # Draw region index label in the top-left corner
        label = f"#{idx}"
        draw.text((x + 5, y + 5), label, fill=(255, 255, 255, 220))

    return overlay.convert("RGB")
