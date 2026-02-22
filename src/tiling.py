"""
Tiling utility module for the AI Assets Toolbox.

Handles splitting images into overlapping tiles, merging processed tiles back
into a full image with linear-gradient feathering, and grid visualisation.

All operations work directly with PIL Images — no base64 encoding.
This module is used by the Gradio UI (CPU container) before and after calling
the GPU services via Modal ``.remote()``.

Key functions
-------------
``calculate_tiles(image_size, tile_size, overlap)``
    Calculate the tile grid for an image.

``extract_tile(image, tile_info)``
    Crop a single tile from the image.

``extract_all_tiles(image, tile_size, overlap)``
    Extract all tiles from an image.

``blend_tiles(tiles, image_size, tile_size, overlap)``
    Reassemble processed tiles into a full image with feathered blending.

``calculate_offset_tiles(image_size, tile_size, overlap)``
    Calculate a half-stride offset tile grid for the seam-removal pass.

``blend_offset_pass(base_image, offset_tiles, offset_results, feather_size)``
    Blend offset-pass tiles onto the base image using feathered masks.

``draw_tile_grid(image, tiles, selected_indices, processed_indices)``
    Draw a tile grid overlay on a copy of the image.

``upscale_image(image, target_width, target_height)``
    Bicubic upscale of an image to the target resolution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
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

    Tiles are placed with the given overlap so that adjacent tiles share
    ``overlap`` pixels on each shared edge.

    Args:
        image_size: (width, height) of the image in pixels.
        tile_size:  size of each square tile (default 1024).
        overlap:    number of pixels shared between adjacent tiles (default 128).

    Returns:
        Ordered list of TileInfo objects (row-major order).

    Raises:
        ValueError: if overlap >= tile_size.
    """
    width, height = image_size
    stride = tile_size - overlap

    if stride <= 0:
        raise ValueError(
            f"overlap ({overlap}) must be less than tile_size ({tile_size})"
        )

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

    Args:
        image:     source PIL Image.
        tile_size: size of each square tile (default 1024).
        overlap:   number of pixels shared between adjacent tiles (default 128).

    Returns:
        List of (TileInfo, PIL Image) pairs in row-major order.
    """
    tiles = calculate_tiles(image.size, tile_size=tile_size, overlap=overlap)
    return [(t, extract_tile(image, t)) for t in tiles]


def split_into_tiles(
    image: Image.Image,
    tile_size: int = 1024,
    overlap: int = 128,
) -> List[dict]:
    """
    Split an image into tiles and return a list of tile dicts.

    This is the primary interface used by the Gradio UI workflow.  Each dict
    contains the PIL Image and position metadata needed to reassemble the
    result with :func:`merge_tiles`.

    Args:
        image:     source PIL Image.
        tile_size: size of each square tile (default 1024).
        overlap:   number of pixels shared between adjacent tiles (default 128).

    Returns:
        List of dicts, each with:
        - ``"tile_id"`` (str): unique identifier, e.g. ``"0_0"``.
        - ``"image"`` (PIL.Image.Image): the cropped tile.
        - ``"info"`` (TileInfo): position metadata.
        - ``"prompt_override"`` (None): placeholder for per-tile prompts.
    """
    pairs = extract_all_tiles(image, tile_size=tile_size, overlap=overlap)
    return [
        {
            "tile_id": tile_info.tile_id,
            "image": tile_img,
            "info": tile_info,
            "prompt_override": None,
        }
        for tile_info, tile_img in pairs
    ]


# ---------------------------------------------------------------------------
# Tile blending / reassembly
# ---------------------------------------------------------------------------

def _make_weight_map(
    h: int,
    w: int,
    overlap: int,
    tile_info: TileInfo,
    image_size: Tuple[int, int],
) -> np.ndarray:
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
        tile_size:  tile size used during extraction (unused but kept for API symmetry).
        overlap:    overlap used during extraction.

    Returns:
        Reconstructed PIL Image (RGB).
    """
    img_w, img_h = image_size
    accumulator = np.zeros((img_h, img_w, 3), dtype=np.float32)
    weight_sum = np.zeros((img_h, img_w, 1), dtype=np.float32)

    for tile_info, tile_img in tiles:
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


def merge_tiles(
    tiles: List[dict],
    output_width: int,
    output_height: int,
    overlap: int = 128,
) -> Image.Image:
    """
    Merge processed tiles back into a full image with feathered blending.

    This is the primary interface used by the Gradio UI workflow, complementing
    :func:`split_into_tiles`.

    Args:
        tiles:         list of tile dicts, each with:
                       - ``"info"`` (TileInfo): position metadata.
                       - ``"image"`` (PIL.Image.Image): the processed tile.
        output_width:  width of the target output image in pixels.
        output_height: height of the target output image in pixels.
        overlap:       overlap used during tile extraction.

    Returns:
        Merged PIL Image (RGB).
    """
    pairs: List[Tuple[TileInfo, Image.Image]] = [
        (t["info"], t["image"]) for t in tiles
    ]
    return blend_tiles(pairs, (output_width, output_height), overlap=overlap)


# ---------------------------------------------------------------------------
# Seam fix — offset grid calculation and blending
# ---------------------------------------------------------------------------

def calculate_offset_tiles(
    image_size: Tuple[int, int],
    tile_size: int = 1024,
    overlap: int = 128,
) -> List[TileInfo]:
    """
    Calculate a half-stride offset tile grid for the seam-removal pass.

    The offset grid shifts the origin by ``stride // 2`` pixels in both X and
    Y directions (where ``stride = tile_size - overlap``).  This places each
    offset tile so that its centre straddles the seam boundaries produced by
    the normal grid, giving the diffusion model full context around those seams.

    Args:
        image_size: (width, height) of the image in pixels.
        tile_size:  size of each square tile (default 1024).
        overlap:    number of pixels shared between adjacent tiles (default 128).

    Returns:
        Ordered list of TileInfo objects (row-major order), deduplicated.

    Raises:
        ValueError: if overlap >= tile_size.
    """
    width, height = image_size
    stride = tile_size - overlap

    if stride <= 0:
        raise ValueError(
            f"overlap ({overlap}) must be less than tile_size ({tile_size})"
        )

    offset = stride // 2

    num_cols = max(1, math.ceil((width + offset - overlap) / stride))
    num_rows = max(1, math.ceil((height + offset - overlap) / stride))

    tiles: List[TileInfo] = []
    for row in range(num_rows):
        for col in range(num_cols):
            raw_x = col * stride - offset
            raw_y = row * stride - offset

            x = max(0, raw_x)
            y = max(0, raw_y)

            x = min(x, max(0, width - tile_size))
            y = min(y, max(0, height - tile_size))

            w = min(tile_size, width - x)
            h = min(tile_size, height - y)

            if w <= 0 or h <= 0:
                continue

            tiles.append(TileInfo(x=x, y=y, w=w, h=h, row=row, col=col))

    # Deduplicate by (x, y) position
    seen: set = set()
    result: List[TileInfo] = []
    for tile in tiles:
        key = (tile.x, tile.y)
        if key in seen:
            continue
        seen.add(key)
        result.append(tile)
    return result


def _make_feather_mask(h: int, w: int, feather_size: int) -> np.ndarray:
    """
    Build a 2-D float32 feather mask for a tile.

    The mask value at each pixel equals:
        min(dist_from_left, dist_from_right, dist_from_top, dist_from_bottom)
        divided by feather_size, clamped to [0, 1].

    This produces a gradient that is 1.0 in the centre and fades to 0.0 at
    the edges over ``feather_size`` pixels.
    """
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)

    dist_top    = ys
    dist_bottom = (h - 1) - ys
    dist_left   = xs
    dist_right  = (w - 1) - xs

    dist_v = np.minimum(dist_top[:, np.newaxis], dist_bottom[:, np.newaxis])
    dist_h = np.minimum(dist_left[np.newaxis, :], dist_right[np.newaxis, :])

    min_dist = np.minimum(dist_v, dist_h)
    mask = np.clip(min_dist / max(feather_size, 1), 0.0, 1.0)
    return mask


def blend_offset_pass(
    base_image: Image.Image,
    offset_tiles: List[TileInfo],
    offset_results: List[Image.Image],
    feather_size: int = 32,
) -> Image.Image:
    """
    Blend offset-pass tiles onto the base image using feathered masks.

    For each offset tile a gradient mask is created that is 1.0 in the centre
    and fades to 0.0 at the edges over ``feather_size`` pixels.  The offset
    tile is composited over the base image using this mask, so that only the
    central region (where Pass-1 seams lie) is replaced while the edges blend
    smoothly into the surrounding base image.

    Args:
        base_image:     The Pass-1 assembled result (PIL Image, RGB).
        offset_tiles:   List of TileInfo objects describing each offset tile's
                        position in the full image.
        offset_results: List of processed PIL Images for each offset tile,
                        in the same order as ``offset_tiles``.
        feather_size:   Number of pixels over which the blend fades at tile
                        edges (default 32).

    Returns:
        New PIL Image with seams blended away.
    """
    if not offset_tiles or not offset_results:
        return base_image

    img_w, img_h = base_image.size
    base_arr = np.array(base_image.convert("RGB"), dtype=np.float32)

    blend_acc = np.zeros_like(base_arr)
    blend_wt  = np.zeros((img_h, img_w), dtype=np.float32)

    for tile_info, tile_img in zip(offset_tiles, offset_results):
        tile_arr = np.array(tile_img.convert("RGB"), dtype=np.float32)
        th, tw = tile_arr.shape[:2]

        tw = min(tw, img_w - tile_info.x)
        th = min(th, img_h - tile_info.y)
        if tw <= 0 or th <= 0:
            continue

        tile_arr = tile_arr[:th, :tw]
        mask = _make_feather_mask(th, tw, feather_size)

        y1, y2 = tile_info.y, tile_info.y + th
        x1, x2 = tile_info.x, tile_info.x + tw

        blend_acc[y1:y2, x1:x2] += tile_arr * mask[:, :, np.newaxis]
        blend_wt[y1:y2, x1:x2]  += mask

    alpha = np.clip(blend_wt, 0.0, 1.0)[:, :, np.newaxis]
    safe_wt = np.where(blend_wt > 0, blend_wt, 1.0)[:, :, np.newaxis]
    offset_avg = blend_acc / safe_wt

    result_arr = offset_avg * alpha + base_arr * (1.0 - alpha)
    result_arr = np.clip(result_arr, 0, 255).astype(np.uint8)
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
        New PIL Image (RGB) with the grid drawn on top.
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

        cx = tile.x + tile.w // 2
        cy = tile.y + tile.h // 2
        label = str(idx)
        draw.text((cx - 8, cy - 8), label, fill=(255, 255, 255, 220))

    return overlay.convert("RGB")


def generate_mask_blur_preview(
    tile_image: Image.Image,
    tile_info: TileInfo,
    overlap: int,
    img_width: int,
    img_height: int,
) -> Image.Image:
    """
    Generate a visualisation of the feathering/blend zones for a tile.

    Areas where the weight map is less than 1.0 (i.e. the feathered overlap
    edges) are tinted with a semi-transparent orange-red overlay so the user
    can see which parts of the tile will be blended with neighbouring tiles.

    Args:
        tile_image: The tile's PIL Image (original or processed).
        tile_info:  TileInfo describing this tile's position in the full image.
        overlap:    Overlap in pixels used when the tile grid was calculated.
        img_width:  Full image width in pixels.
        img_height: Full image height in pixels.

    Returns:
        A new PIL Image (RGB) with the feather zones highlighted.
    """
    tile_rgb = tile_image.convert("RGB")
    w, h = tile_rgb.size

    weight = _make_weight_map(h, w, overlap, tile_info, (img_width, img_height))

    tile_arr = np.array(tile_rgb, dtype=np.float32)
    feather_mask = 1.0 - weight

    tint_r = np.full((h, w), 255.0, dtype=np.float32)
    tint_g = np.full((h, w), 120.0, dtype=np.float32)
    tint_b = np.full((h, w), 0.0,   dtype=np.float32)
    tint = np.stack([tint_r, tint_g, tint_b], axis=-1)

    alpha = (feather_mask * 0.55)[:, :, np.newaxis]
    result_arr = tile_arr * (1.0 - alpha) + tint * alpha
    result_arr = np.clip(result_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(result_arr, mode="RGB")


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
# Grid info helper (convenience wrapper)
# ---------------------------------------------------------------------------

def calculate_tile_grid(
    image_width: int,
    image_height: int,
    tile_size: int = 1024,
    overlap: int = 128,
) -> dict:
    """
    Calculate tile grid metadata for an image.

    Args:
        image_width:  image width in pixels.
        image_height: image height in pixels.
        tile_size:    size of each square tile (default 1024).
        overlap:      number of pixels shared between adjacent tiles (default 128).

    Returns:
        Dict with keys:
        - ``"tiles"`` (list[TileInfo]): ordered tile list (row-major).
        - ``"num_tiles"`` (int): total number of tiles.
        - ``"num_rows"`` (int): number of tile rows.
        - ``"num_cols"`` (int): number of tile columns.
        - ``"tile_size"`` (int): tile size used.
        - ``"overlap"`` (int): overlap used.
    """
    tiles = calculate_tiles((image_width, image_height), tile_size=tile_size, overlap=overlap)
    num_rows = max(t.row for t in tiles) + 1 if tiles else 0
    num_cols = max(t.col for t in tiles) + 1 if tiles else 0
    return {
        "tiles": tiles,
        "num_tiles": len(tiles),
        "num_rows": num_rows,
        "num_cols": num_cols,
        "tile_size": tile_size,
        "overlap": overlap,
    }
