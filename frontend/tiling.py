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


def generate_mask_blur_preview(
    tile_image: Image.Image,
    tile_info: TileInfo,
    overlap: int,
    img_width: int,
    img_height: int,
) -> Image.Image:
    """
    Generate a visualization of the feathering/blend zones for a tile.

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
        A new PIL Image (RGBA converted to RGB) with the feather zones
        highlighted as a semi-transparent orange-red tint.
    """
    tile_rgb = tile_image.convert("RGB")
    w, h = tile_rgb.size

    # Build the weight map for this tile
    weight = _make_weight_map(h, w, overlap, tile_info, (img_width, img_height))

    # Create the overlay: pixels where weight < 1.0 get an orange-red tint.
    # The tint intensity is proportional to (1 - weight) so the centre
    # (weight == 1.0) is untouched and the deepest feather zone is most vivid.
    tile_arr = np.array(tile_rgb, dtype=np.float32)  # (h, w, 3)

    # Feather mask: 1.0 where fully feathered, 0.0 where no feathering
    feather_mask = (1.0 - weight)  # (h, w), range [0, 1]

    # Orange-red tint colour (R=255, G=120, B=0)
    tint_r = np.full((h, w), 255.0, dtype=np.float32)
    tint_g = np.full((h, w), 120.0, dtype=np.float32)
    tint_b = np.full((h, w), 0.0,   dtype=np.float32)
    tint = np.stack([tint_r, tint_g, tint_b], axis=-1)  # (h, w, 3)

    # Blend: result = tile * (1 - alpha) + tint * alpha
    # where alpha = feather_mask * 0.55  (max 55% opacity at the deepest edge)
    alpha = (feather_mask * 0.55)[:, :, np.newaxis]  # (h, w, 1)
    result_arr = tile_arr * (1.0 - alpha) + tint * alpha
    result_arr = np.clip(result_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(result_arr, mode="RGB")


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

    Tiles that would start before the image boundary are clamped so that they
    begin at x=0 / y=0 but still cover the same area.  Tiles that extend
    beyond the right / bottom edge are clamped to the image boundary.

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

    offset = stride // 2

    # Number of tiles needed to cover the image starting from -offset.
    # We need enough tiles so that the last tile reaches the image edge.
    num_cols = max(1, math.ceil((width + offset - overlap) / stride))
    num_rows = max(1, math.ceil((height + offset - overlap) / stride))

    tiles: List[TileInfo] = []
    for row in range(num_rows):
        for col in range(num_cols):
            # Raw position (may be negative for the first tile)
            raw_x = col * stride - offset
            raw_y = row * stride - offset

            # Clamp start to image boundary
            x = max(0, raw_x)
            y = max(0, raw_y)

            # Clamp so the tile does not start past the last valid position
            x = min(x, max(0, width - tile_size))
            y = min(y, max(0, height - tile_size))

            w = min(tile_size, width - x)
            h = min(tile_size, height - y)

            # Skip degenerate tiles (zero area)
            if w <= 0 or h <= 0:
                continue

            tiles.append(TileInfo(x=x, y=y, w=w, h=h, row=row, col=col))

    # Deduplicate by (x, y) position — small images can clamp multiple tiles
    # to the same origin, producing duplicate work sent to the backend.
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

    Args:
        h:            tile height in pixels.
        w:            tile width in pixels.
        feather_size: number of pixels over which the fade occurs.

    Returns:
        float32 ndarray of shape (h, w) with values in [0, 1].
    """
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)

    dist_top    = ys                         # distance from top edge
    dist_bottom = (h - 1) - ys              # distance from bottom edge
    dist_left   = xs                         # distance from left edge
    dist_right  = (w - 1) - xs              # distance from right edge

    # Broadcast to 2-D
    dist_v = np.minimum(dist_top[:, np.newaxis], dist_bottom[:, np.newaxis])  # (h, 1)
    dist_h = np.minimum(dist_left[np.newaxis, :], dist_right[np.newaxis, :])  # (1, w)

    min_dist = np.minimum(dist_v, dist_h)  # (h, w)

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
    base_arr = np.array(base_image.convert("RGB"), dtype=np.float32)  # (H, W, 3)

    # Accumulate weighted offset tiles
    blend_acc = np.zeros_like(base_arr)            # weighted sum of offset tile pixels
    blend_wt  = np.zeros((img_h, img_w), dtype=np.float32)  # total weight per pixel

    for tile_info, tile_img in zip(offset_tiles, offset_results):
        tile_arr = np.array(tile_img.convert("RGB"), dtype=np.float32)
        th, tw = tile_arr.shape[:2]

        # Clamp tile dimensions to what actually fits in the image
        tw = min(tw, img_w - tile_info.x)
        th = min(th, img_h - tile_info.y)
        if tw <= 0 or th <= 0:
            continue

        tile_arr = tile_arr[:th, :tw]

        mask = _make_feather_mask(th, tw, feather_size)  # (th, tw)

        y1, y2 = tile_info.y, tile_info.y + th
        x1, x2 = tile_info.x, tile_info.x + tw

        blend_acc[y1:y2, x1:x2] += tile_arr * mask[:, :, np.newaxis]
        blend_wt[y1:y2, x1:x2]  += mask

    # Composite: result = offset_avg * alpha + base * (1 - alpha)
    # where alpha = blend_wt clamped to [0, 1]
    alpha = np.clip(blend_wt, 0.0, 1.0)[:, :, np.newaxis]  # (H, W, 1)

    # Where blend_wt > 0, compute the weighted average of offset tiles
    safe_wt = np.where(blend_wt > 0, blend_wt, 1.0)[:, :, np.newaxis]
    offset_avg = blend_acc / safe_wt  # (H, W, 3)

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
