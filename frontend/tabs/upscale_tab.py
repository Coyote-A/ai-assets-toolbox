"""
Tile Upscale tab — the main upscaling workflow for the AI Assets Toolbox.

Layout
------
Full-width image upload at the top.
Two-column layout below:
  Left column  (~65%): Unified tile grid (HTML/CSS/JS) + selected tile detail + action buttons
  Right column (~35%): Settings organized in collapsible accordions

State
-----
tiles_state          : list[dict]  — per-tile data (info, prompt, processed image)
selected_idx_state   : int         — currently selected tile index (-1 = none)
original_img_state   : PIL.Image   — the uploaded (and upscaled) image
grid_cols_state      : int         — number of columns in the tile grid
full_image_b64_state : str         — base64 JPEG of the full upscaled image for grid display

LoRAs
-----
Three CivitAI LoRAs are hardcoded and automatically included in every upscale request
with weight 1.0.  They are downloaded at container startup by ensure_loras_downloaded().
"""
from __future__ import annotations

import base64
import io
import sys
import os
from typing import Any, Dict, List, Optional, Tuple

# Allow imports from the frontend package root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from PIL import Image

import config
from api_client import RunPodClient, RunPodError
from tiling import (
    TileInfo,
    extract_all_tiles,
    extract_tile,
    blend_tiles,
    upscale_image,
    calculate_offset_tiles,
    blend_offset_pass,
    generate_mask_blur_preview,
)


# ---------------------------------------------------------------------------
# Image ↔ base64 helpers
# ---------------------------------------------------------------------------

def _pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _b64_to_pil(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

RESOLUTION_PRESETS = [
    "8192 (8K)",
    "4096 (4K)",
    "2048 (2K)",
    "1024",
    "Same as original",
]

# Map label → long-side pixel value (None = keep original)
RESOLUTION_LONG_SIDE: Dict[str, Optional[int]] = {
    "8192 (8K)": 8192,
    "4096 (4K)": 4096,
    "2048 (2K)": 2048,
    "1024": 1024,
    "Same as original": None,
}

# The single active model (hardcoded — only illustrious-xl is supported)
ACTIVE_MODEL = "illustrious-xl"

# Hardcoded LoRAs — automatically included in every upscale request
HARDCODED_LORAS = [
    {"name": "lora_929497", "weight": 1.0},
    {"name": "lora_100435", "weight": 1.0},
    {"name": "lora_1231943", "weight": 1.0},
]

# Generation resolution presets: (label, value) where 0 means "same as grid tile size"
GENERATION_RES_CHOICES = [
    ("Same as grid", 0),
    ("1536×1536 ✨ Recommended", 1536),
    ("2048×2048 (Max detail)", 2048),
]
# Map label → value for the dropdown
_GEN_RES_LABEL_TO_VALUE = {label: val for label, val in GENERATION_RES_CHOICES}
_GEN_RES_LABELS = [label for label, _ in GENERATION_RES_CHOICES]


def _compute_target_size(
    original_size: Tuple[int, int],
    resolution_choice: str,
) -> Tuple[int, int]:
    """Compute target (width, height) from the long-side resolution preset.

    The shorter side is scaled proportionally to preserve the original aspect
    ratio.  If the preset is "Same as original", the original size is returned
    unchanged.
    """
    orig_w, orig_h = original_size
    long_side = RESOLUTION_LONG_SIDE.get(resolution_choice)
    if long_side is None:
        # "Same as original" — no upscale
        return orig_w, orig_h
    if orig_w >= orig_h:
        target_w = long_side
        target_h = max(1, int(orig_h * long_side / orig_w))
    else:
        target_h = long_side
        target_w = max(1, int(orig_w * long_side / orig_h))
    return target_w, target_h


# ---------------------------------------------------------------------------
# Grid image encoder
# ---------------------------------------------------------------------------

def _encode_grid_image(img: Image.Image, max_side: int = 2048, quality: int = 75) -> str:
    """Encode a PIL image as a JPEG base64 string for use in the tile grid.

    Optionally downscales to max_side on the longest dimension to keep the
    payload small while still providing enough resolution for thumbnail display.
    """
    display_img = img.convert("RGB")
    w, h = display_img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        display_img = display_img.resize(
            (int(w * scale), int(h * scale)), Image.LANCZOS
        )
    buf = io.BytesIO()
    display_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# HTML tile grid builder
# ---------------------------------------------------------------------------

_TILE_GRID_CSS = """
<style>
/* ── Grid container: full image as background, tiles are absolute overlays ── */
.tile-grid-wrap {
  position: relative;
  width: 100%;
  border-radius: 8px;
  overflow: hidden;
  background-size: 100% 100%;
  background-repeat: no-repeat;
}
/* ── Individual tile: absolute overlay ───────────────────────────────────── */
.tile {
  position: absolute;
  cursor: pointer;
  overflow: hidden;
  box-shadow: inset 0 0 0 0.5px rgba(255, 255, 255, 0.12);
  transition: box-shadow 0.15s ease;
  background-repeat: no-repeat;
  background-size: 100% 100%;
}
/* ── Selection dimming ────────────────────────────────────────────────────── */
.tile-grid-wrap.has-selection .tile:not(.selected)::before {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.45);
  pointer-events: none;
  z-index: 2;
  transition: background 0.15s ease;
}
.tile.selected {
  z-index: 3;
  box-shadow: inset 0 0 0 0.5px rgba(255, 255, 255, 0.12),
              inset 0 0 0 2px rgba(30, 111, 255, 0.8);
}
/* ── Status indicators ────────────────────────────────────────────────────── */
.tile[data-status=processed]::after {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(0, 200, 80, 0.08);
  pointer-events: none;
  z-index: 1;
}
.tile[data-status=processing] {
  animation: tile-pulse 1.2s ease-in-out infinite;
}
@keyframes tile-pulse {
  0%, 100% { box-shadow: inset 0 0 0 0.5px rgba(255,255,255,0.12), 0 0 0 0 rgba(255, 167, 38, 0.4); }
  50%       { box-shadow: inset 0 0 0 0.5px rgba(255,255,255,0.12), 0 0 0 4px rgba(255, 167, 38, 0); }
}
.tile:hover {
  box-shadow: inset 0 0 0 0.5px rgba(255, 255, 255, 0.12),
              inset 0 0 0 1.5px rgba(74, 128, 255, 0.6);
}
/* ── Tile icons ───────────────────────────────────────────────────────────── */
.tile-status-icon {
  position: absolute;
  bottom: 3px;
  right: 3px;
  font-size: 14px;
  pointer-events: none;
  z-index: 5;
}
.tile[data-status=processed] .tile-status-icon::after {
  content: '\u2713';
  color: #00c853;
  text-shadow: 0 0 3px rgba(0, 0, 0, 0.8);
}
.tile[data-status=processing] .tile-status-icon::after { content: '\u23f3'; }
/* ── Overlap zone strips ──────────────────────────────────────────────────── */
.tile .overlap-zone {
  position: absolute;
  pointer-events: none;
  z-index: 4;
  display: none;
}
.tile.selected .overlap-zone { display: block; }
.tile .overlap-zone.left {
  left: 0; top: 0; bottom: 0;
  background: linear-gradient(to right, rgba(255, 165, 0, 0.35), transparent);
}
.tile .overlap-zone.right {
  right: 0; top: 0; bottom: 0;
  background: linear-gradient(to left, rgba(255, 165, 0, 0.35), transparent);
}
.tile .overlap-zone.top {
  left: 0; right: 0; top: 0;
  background: linear-gradient(to bottom, rgba(255, 165, 0, 0.35), transparent);
}
.tile .overlap-zone.bottom {
  left: 0; right: 0; bottom: 0;
  background: linear-gradient(to top, rgba(255, 165, 0, 0.35), transparent);
}
@media (max-width: 900px) {
}
</style>
"""

# Inline onclick JS template — embedded directly on each tile div so it fires
# reliably even after Gradio re-renders gr.HTML (Gradio does NOT re-execute
# <script> blocks on HTML updates).
_TILE_ONCLICK_JS = (
    "(function(el){"
    "var wrap=el.closest('.tile-grid-wrap');"
    "var wasSelected=el.classList.contains('selected');"
    "wrap&&wrap.querySelectorAll('.tile').forEach(function(t){t.classList.remove('selected');});"
    "if(wasSelected){"
    "wrap&&wrap.classList.remove('has-selection');"
    "}else{"
    "el.classList.add('selected');"
    "wrap&&wrap.classList.add('has-selection');"
    "}"
    "var tb=document.querySelector('#tile-selected-idx textarea');"
    "if(tb){"
    "var ns=Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,'value').set;"
    "ns.call(tb,wasSelected?'-1':el.dataset.idx);"
    "tb.dispatchEvent(new Event('input',{bubbles:true}));"
    "}"
    "})(this)"
)


def _build_tile_grid_html(
    tiles_data: List[Dict],
    full_image_b64: str,
    selected_idx: int,
    img_w: int,
    img_h: int,
    overlap: int = 128,
) -> str:
    """Build the HTML/CSS string for the custom tile grid component.

    Layout strategy (Bug 2 + Bug 4 fix):
    - The container div (.tile-grid-wrap) has the full image as its CSS
      background-image with background-size: 100% 100%.  Its height is set
      via padding-bottom to preserve the image aspect ratio.
    - Each tile div is position:absolute with left/top/width/height expressed
      as percentages of the container, derived directly from the tile's pixel
      coordinates in the display image.  This is pixel-perfect and handles
      non-square edge tiles automatically.
    - Processed tiles get their own background-image (background-size: 100% 100%)
      which covers only that tile's area.
    - Inline onclick handlers (Bug 3 fix) are embedded on each tile div so
      they fire reliably even after Gradio re-renders gr.HTML.

    Args:
        tiles_data: List of tile dicts with 'info' (TileInfo), 'processed_b64' (optional).
                    The TileInfo coords must already be in display-image space.
        full_image_b64: Base64 JPEG of the full (possibly downscaled) image for display
        selected_idx: Currently selected tile index (-1 for none)
        img_w: Full display image width in pixels
        img_h: Full display image height in pixels
        overlap: Overlap in original image pixels (used for overlap zone visualization)

    Returns:
        HTML string with embedded CSS
    """
    if not tiles_data:
        return "<p style='color:#888; padding:16px;'>Upload an image to see the tile grid.</p>"

    # Determine grid dimensions from tile infos
    grid_cols = max(t["info"].col for t in tiles_data) + 1
    grid_rows = max(t["info"].row for t in tiles_data) + 1

    has_selection_class = " has-selection" if selected_idx >= 0 else ""

    # padding-bottom trick: makes the container maintain the image aspect ratio
    # while width: 100% fills the available space.
    padding_bottom_pct = (img_h / img_w * 100) if img_w > 0 else 100.0

    tile_divs = []
    for i, tile in enumerate(tiles_data):
        ti: TileInfo = tile["info"]
        is_processed = bool(tile.get("processed_b64"))
        status = "processed" if is_processed else "default"
        selected_class = " selected" if i == selected_idx else ""

        # Absolute position as % of container (= % of display image dimensions)
        left_pct   = ti.x / img_w * 100 if img_w > 0 else 0.0
        top_pct    = ti.y / img_h * 100 if img_h > 0 else 0.0
        width_pct  = ti.w / img_w * 100 if img_w > 0 else 100.0
        height_pct = ti.h / img_h * 100 if img_h > 0 else 100.0

        pos_style = (
            f"left:{left_pct:.4f}%;top:{top_pct:.4f}%;"
            f"width:{width_pct:.4f}%;height:{height_pct:.4f}%;"
        )

        if is_processed and tile.get("processed_b64"):
            # Processed tile: show its own image filling the tile area exactly
            proc_b64 = tile["processed_b64"]
            bg_style = (
                f"background-image:url('data:image/jpeg;base64,{proc_b64}');"
                "background-size:100% 100%;"
            )
        else:
            # Unprocessed tile: transparent — the container background shows through
            bg_style = ""

        # Overlap zone widths as % of this tile's own dimensions
        overlap_pct_x = (overlap / ti.w * 100) if ti.w > 0 else 0.0
        overlap_pct_y = (overlap / ti.h * 100) if ti.h > 0 else 0.0

        overlap_divs = []
        if ti.col > 0:
            overlap_divs.append(
                f'<div class="overlap-zone left" style="width:{overlap_pct_x:.3f}%;"></div>'
            )
        if ti.col < grid_cols - 1:
            overlap_divs.append(
                f'<div class="overlap-zone right" style="width:{overlap_pct_x:.3f}%;"></div>'
            )
        if ti.row > 0:
            overlap_divs.append(
                f'<div class="overlap-zone top" style="height:{overlap_pct_y:.3f}%;"></div>'
            )
        if ti.row < grid_rows - 1:
            overlap_divs.append(
                f'<div class="overlap-zone bottom" style="height:{overlap_pct_y:.3f}%;"></div>'
            )
        overlap_html = "".join(overlap_divs)

        tile_divs.append(
            f'<div class="tile{selected_class}" '
            f'data-idx="{i}" data-row="{ti.row}" data-col="{ti.col}" '
            f'data-total-rows="{grid_rows}" data-total-cols="{grid_cols}" '
            f'data-status="{status}" '
            f'onclick="{_TILE_ONCLICK_JS}" '
            f'style="{pos_style}{bg_style}">'
            f'<span class="tile-status-icon"></span>'
            f'{overlap_html}'
            f'</div>'
        )

    tiles_html = "\n".join(tile_divs)

    # Container: full image as background, padding-bottom preserves aspect ratio.
    # The inner div is position:absolute inset:0 to give tiles a positioned parent.
    html = (
        f'<div class="tile-grid-wrap{has_selection_class}" '
        f'style="'
        f'padding-bottom:{padding_bottom_pct:.4f}%;'
        f'background-image:url(\'data:image/jpeg;base64,{full_image_b64}\');">'
        f'{tiles_html}'
        f'</div>'
        f'{_TILE_GRID_CSS}'
    )
    return html


# ---------------------------------------------------------------------------
# Core processing functions
# ---------------------------------------------------------------------------

def _build_tile_state(tile_info: TileInfo, tile_img: Image.Image) -> Dict[str, Any]:
    """Create the initial state dict for a single tile."""
    return {
        "info": tile_info,
        "tile_id": tile_info.tile_id,
        "original_b64": _pil_to_b64(tile_img),
        "processed_b64": None,
        "prompt": "",
    }


def _compute_display_dims(orig_w: int, orig_h: int, max_side: int = 2048) -> Tuple[int, int]:
    """Compute display image dimensions after optional downscale."""
    if max(orig_w, orig_h) > max_side:
        scale = max_side / max(orig_w, orig_h)
        return int(orig_w * scale), int(orig_h * scale)
    return orig_w, orig_h


def _scale_tiles_for_display(
    tiles_state: List[Dict],
    orig_w: int,
    orig_h: int,
    display_w: int,
    display_h: int,
) -> List[Dict]:
    """Return a copy of tiles_state with TileInfo coords scaled to display image space."""
    scale_x = display_w / orig_w
    scale_y = display_h / orig_h
    scaled = []
    for tile in tiles_state:
        ti: TileInfo = tile["info"]
        scaled_ti = TileInfo(
            row=ti.row,
            col=ti.col,
            x=int(ti.x * scale_x),
            y=int(ti.y * scale_y),
            w=int(ti.w * scale_x),
            h=int(ti.h * scale_y),
        )
        scaled.append({**tile, "info": scaled_ti})
    return scaled


def _on_image_upload(
    image: Optional[Image.Image],
    resolution_choice: str,
    tile_size: int,
    overlap: int,
) -> Tuple[
    List[Dict],             # tiles_state
    int,                    # selected_idx_state (-1)
    Optional[Image.Image],  # original_img_state
    int,                    # grid_cols_state
    str,                    # full_image_b64_state
    str,                    # tile_grid_html (gr.HTML value)
    str,                    # status
]:
    """Handle image upload: upscale to target resolution, calculate tiles, build HTML grid."""
    _empty_html = "<p style='color:#888; padding:16px;'>Upload an image to see the tile grid.</p>"
    if image is None:
        return [], -1, None, 1, "", _empty_html, "No image uploaded."

    target_w, target_h = _compute_target_size(image.size, resolution_choice)
    target_w = min(target_w, config.MAX_RESOLUTION)
    target_h = min(target_h, config.MAX_RESOLUTION)

    upscaled = upscale_image(image, target_w, target_h)
    tile_pairs = extract_all_tiles(upscaled, tile_size=tile_size, overlap=overlap)

    tiles_state = [_build_tile_state(ti, img) for ti, img in tile_pairs]

    # Determine grid column count from tile infos
    if tiles_state:
        grid_cols = max(t["info"].col for t in tiles_state) + 1
    else:
        grid_cols = 1

    # Encode the full image as JPEG for the grid display (downscaled to 2048 max)
    full_image_b64 = _encode_grid_image(upscaled, max_side=2048, quality=75)

    # Compute display image dimensions (after potential downscale)
    display_w, display_h = _compute_display_dims(upscaled.width, upscaled.height, max_side=2048)

    # Scale tile coords to display image space for CSS background-position math
    scaled_tiles = _scale_tiles_for_display(tiles_state, upscaled.width, upscaled.height, display_w, display_h)

    grid_html = _build_tile_grid_html(scaled_tiles, full_image_b64, -1, display_w, display_h, overlap)

    status = (
        f"✅ Image upscaled to {target_w}×{target_h}. "
        f"Grid: {len(tiles_state)} tiles ({tile_size}px, overlap {overlap}px). "
        f"Click a tile to select it."
    )
    return tiles_state, -1, upscaled, grid_cols, full_image_b64, grid_html, status


def _rebuild_grid_html(
    tiles_state: List[Dict],
    full_image_b64: str,
    selected_idx: int,
    original_img: Optional[Image.Image],
    overlap: int = 128,
) -> str:
    """Rebuild the tile grid HTML from current state.

    Handles coordinate scaling from original image space to display image space.
    """
    _empty_html = "<p style='color:#888; padding:16px;'>Upload an image to see the tile grid.</p>"
    if not tiles_state or original_img is None or not full_image_b64:
        return _empty_html

    orig_w, orig_h = original_img.size
    display_w, display_h = _compute_display_dims(orig_w, orig_h, max_side=2048)
    scaled_tiles = _scale_tiles_for_display(tiles_state, orig_w, orig_h, display_w, display_h)
    return _build_tile_grid_html(scaled_tiles, full_image_b64, selected_idx, display_w, display_h, overlap)


def _on_prompt_edit(
    new_prompt: str,
    selected_idx: int,
    tiles_state: List[Dict],
) -> List[Dict]:
    """Update the prompt for the currently selected tile."""
    if selected_idx < 0 or selected_idx >= len(tiles_state):
        return tiles_state
    tiles_state[selected_idx]["prompt"] = new_prompt
    return tiles_state


def _caption_all_tiles(
    tiles_state: List[Dict],
    system_prompt: Optional[str],
    client: RunPodClient,
) -> Tuple[List[Dict], str]:
    """Send all tiles to RunPod for captioning and store results in state."""
    if not tiles_state:
        return tiles_state, "⚠️ No tiles to caption."

    tiles_b64 = [
        {"tile_id": t["tile_id"], "image_b64": t["original_b64"]}
        for t in tiles_state
    ]

    try:
        captions = client.caption_tiles(tiles_b64, system_prompt=system_prompt or None)
    except RunPodError as exc:
        return tiles_state, f"❌ Caption error: {exc}"

    caption_map = {c["tile_id"]: c["caption"] for c in captions}
    for tile in tiles_state:
        if tile["tile_id"] in caption_map:
            tile["prompt"] = caption_map[tile["tile_id"]]

    return tiles_state, f"✅ Captioned {len(captions)} tile(s)."


def _caption_selected_tile(
    selected_idx: int,
    tiles_state: List[Dict],
    system_prompt: Optional[str],
    client: RunPodClient,
) -> Tuple[List[Dict], str]:
    """Send only the currently selected tile to RunPod for captioning."""
    if selected_idx < 0 or selected_idx >= len(tiles_state):
        return tiles_state, "⚠️ No tile selected. Click a tile in the grid first."

    tile = tiles_state[selected_idx]
    tiles_b64 = [{"tile_id": tile["tile_id"], "image_b64": tile["original_b64"]}]

    try:
        captions = client.caption_tiles(tiles_b64, system_prompt=system_prompt or None)
    except RunPodError as exc:
        return tiles_state, f"❌ Caption error: {exc}"

    if captions:
        caption_text = captions[0]["caption"]
        tiles_state[selected_idx]["prompt"] = caption_text
        return tiles_state, caption_text

    return tiles_state, ""


def _upscale_tiles_batch(
    tiles_state: List[Dict],
    indices: Optional[List[int]],
    original_img: Optional[Image.Image],
    # generation params
    global_prompt: str,
    negative_prompt: str,
    tile_size: int,
    overlap: int,
    strength: float,
    steps: int,
    cfg_scale: float,
    seed: int,
    controlnet_enabled: bool,
    conditioning_scale: float,
    client: RunPodClient,
    # IP-Adapter params (optional)
    ip_adapter_enabled: bool = False,
    ip_adapter_image: Optional[Image.Image] = None,
    ip_adapter_scale: float = 0.6,
    # Generation resolution (0 = same as tile_size)
    gen_res: int = 0,
) -> Tuple[List[Dict], Optional[Image.Image], str]:
    """
    Process a batch of tiles (all or selected) through RunPod upscaling.

    If gen_res > tile_size, each tile is upscaled to gen_res before being sent
    to the backend for generation, then downscaled back to tile_size after
    receiving the result.  The grid, tile positions, and final assembly all
    remain at the grid resolution (tile_size).

    Returns updated tiles_state, assembled result image, and status text.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)

    if not tiles_state or original_img is None:
        return tiles_state, None, "⚠️ No tiles to process."

    # Resolve effective generation resolution — must be >= tile_size
    effective_gen_res = gen_res if gen_res and gen_res > tile_size else tile_size

    target_indices = indices if indices is not None else list(range(len(tiles_state)))

    # Encode IP-Adapter style image to base64 if provided and enabled
    ip_adapter_b64: Optional[str] = None
    if ip_adapter_enabled and ip_adapter_image is not None:
        # Resize to 224×224 before encoding (ViT-H CLIP encoder expected size)
        style_img = ip_adapter_image.convert("RGB").resize((224, 224))
        ip_adapter_b64 = _pil_to_b64(style_img)

    tiles_payload = []
    for idx in target_indices:
        tile = tiles_state[idx]
        tile_prompt = tile.get("prompt", "") or ""

        # Upscale tile image to generation resolution if needed
        tile_img = _b64_to_pil(tile["original_b64"])
        if effective_gen_res != tile_size:
            _log.info(
                "Upscaling tile %s from %dx%d to %dx%d for generation",
                tile["tile_id"], tile_img.width, tile_img.height,
                effective_gen_res, effective_gen_res,
            )
            tile_img = tile_img.resize((effective_gen_res, effective_gen_res), Image.LANCZOS)
        tile_b64 = _pil_to_b64(tile_img)

        entry: Dict[str, Any] = {
            "tile_id": tile["tile_id"],
            "image_b64": tile_b64,
            "prompt_override": tile_prompt if tile_prompt else None,
            "model": ACTIVE_MODEL,
            "loras": HARDCODED_LORAS,
            "global_prompt": global_prompt,
            "negative_prompt": negative_prompt,
            "controlnet_enabled": controlnet_enabled,
            "conditioning_scale": conditioning_scale,
            "strength": strength,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "ip_adapter_enabled": ip_adapter_enabled and ip_adapter_b64 is not None,
            "ip_adapter_scale": ip_adapter_scale,
            "target_width": effective_gen_res,
            "target_height": effective_gen_res,
        }
        if ip_adapter_b64 is not None:
            entry["ip_adapter_image"] = ip_adapter_b64
        tiles_payload.append(entry)

    try:
        results = client.upscale_tiles(tiles_payload)
    except RunPodError as exc:
        return tiles_state, None, f"❌ Upscale error: {exc}"

    result_map = {r["tile_id"]: r["image_b64"] for r in results}
    for tile in tiles_state:
        if tile["tile_id"] in result_map:
            processed_b64 = result_map[tile["tile_id"]]
            # Downscale back to grid tile size if generation was at higher resolution
            if effective_gen_res != tile_size:
                proc_img = _b64_to_pil(processed_b64)
                _log.info(
                    "Downscaling processed tile %s from %dx%d back to %dx%d",
                    tile["tile_id"], proc_img.width, proc_img.height,
                    tile_size, tile_size,
                )
                proc_img = proc_img.resize((tile_size, tile_size), Image.LANCZOS)
                processed_b64 = _pil_to_b64(proc_img)
            tile["processed_b64"] = processed_b64

    # Assemble result: use processed tiles where available, original elsewhere
    assembled_pairs = []
    for tile in tiles_state:
        tile_info: TileInfo = tile["info"]
        if tile.get("processed_b64"):
            img = _b64_to_pil(tile["processed_b64"])
        else:
            img = _b64_to_pil(tile["original_b64"])
        assembled_pairs.append((tile_info, img))

    result_img = blend_tiles(assembled_pairs, original_img.size, tile_size=tile_size, overlap=overlap)
    processed_count = sum(1 for t in tiles_state if t.get("processed_b64"))
    gen_res_label = f" (gen @ {effective_gen_res}px)" if effective_gen_res != tile_size else ""
    status = f"✅ Processed {len(results)} tile(s){gen_res_label}. Total processed: {processed_count}/{len(tiles_state)}."
    return tiles_state, result_img, status


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def create_upscale_tab(client: RunPodClient) -> None:
    """Render the Tile Upscale tab inside a Gradio Blocks context."""

    with gr.Tab("🖼️ Tile Upscale"):

        # ----------------------------------------------------------------
        # State
        # ----------------------------------------------------------------
        tiles_state = gr.State(value=[])
        selected_idx_state = gr.State(value=-1)
        original_img_state = gr.State(value=None)
        grid_cols_state = gr.State(value=1)
        full_image_b64_state = gr.State(value="")

        # Hidden textbox for JS → Python tile selection communication
        tile_selected_idx_tb = gr.Textbox(
            value="-1",
            visible=False,
            elem_id="tile-selected-idx",
        )

        # ----------------------------------------------------------------
        # TOP: Full-width image upload + resolution
        # ----------------------------------------------------------------
        gr.Markdown("## 🖼️ Tile-Based AI Upscaling")
        gr.Markdown(
            "Upload an image to split it into tiles. "
            "Click any tile in the grid to select it, then upscale individually or all at once."
        )

        with gr.Row():
            with gr.Column(scale=3):
                image_input = gr.Image(
                    label="📁 Upload Image",
                    type="pil",
                    height=200,
                )
            with gr.Column(scale=2):
                resolution_dd = gr.Dropdown(
                    choices=RESOLUTION_PRESETS,
                    value="4096 (4K)",
                    label="Target Resolution (longer side)",
                    info=(
                        "The image's longer side will be scaled to this value. "
                        "The shorter side is calculated automatically to preserve the original aspect ratio."
                    ),
                )
                regen_grid_btn = gr.Button("🔄 Regenerate Grid", variant="secondary", size="sm")

        status_text = gr.Textbox(
            label="Status",
            interactive=False,
            lines=1,
            placeholder="Upload an image to get started…",
            elem_id="status-bar",
        )

        # ----------------------------------------------------------------
        # MAIN: Two-column layout
        # ----------------------------------------------------------------
        with gr.Row(equal_height=False):

            # ============================================================
            # LEFT COLUMN (~65%): Tile grid + selected tile + actions
            # ============================================================
            with gr.Column(scale=13, min_width=480):

                # ---- Custom HTML tile grid ----
                gr.Markdown("### 🗂️ Tile Grid — click to select")
                tile_grid_html = gr.HTML(
                    value="<p style='color:#888; padding:16px;'>Upload an image to see the tile grid.</p>",
                    elem_id="tile-grid-container",
                )

                # ---- Selected tile detail panel ----
                with gr.Group(elem_classes=["selected-tile-panel"]):
                    gr.Markdown("### 📌 Selected Tile")
                    with gr.Row():
                        tile_orig_preview = gr.Image(
                            label="Original + Blend Zones",
                            type="pil",
                            interactive=False,
                            height=200,
                            show_label=True,
                        )
                        tile_proc_preview = gr.Image(
                            label="Processed",
                            type="pil",
                            interactive=False,
                            height=200,
                            show_label=True,
                        )
                    tile_prompt_box = gr.Textbox(
                        label="Tile Prompt (editable — overrides global prompt for this tile)",
                        lines=2,
                        placeholder="Leave empty to use the global prompt, or enter a tile-specific prompt…",
                    )

                # ---- Action buttons ----
                gr.Markdown("### 🚀 Actions")
                with gr.Row(elem_classes=["action-buttons"]):
                    upscale_sel_btn = gr.Button(
                        "⬆️ Upscale Selected Tile",
                        variant="secondary",
                        scale=1,
                    )
                    upscale_all_btn = gr.Button(
                        "⬆️⬆️ Upscale All Tiles",
                        variant="primary",
                        scale=2,
                    )
                with gr.Row():
                    caption_btn = gr.Button(
                        "🏷️ Auto-Caption All Tiles",
                        variant="secondary",
                        scale=1,
                    )
                    caption_sel_btn = gr.Button(
                        "🏷️ Caption Selected",
                        variant="secondary",
                        scale=1,
                    )

                # ---- Final result ----
                gr.Markdown("### 🖼️ Final Result")
                result_image = gr.Image(
                    label="Assembled Upscaled Image",
                    type="pil",
                    interactive=False,
                    height=400,
                )
                with gr.Row():
                    download_btn = gr.Button("💾 Download Result", variant="secondary", scale=1)
                    copy_to_input_btn = gr.Button("🔄 Output → Input", variant="secondary", scale=1)
                download_file = gr.File(visible=False, label="Download")

            # ============================================================
            # RIGHT COLUMN (~35%): Settings accordions
            # ============================================================
            with gr.Column(scale=7, min_width=300):

                # ---- 📐 Grid Settings ----
                with gr.Accordion("📐 Grid Settings", open=True):
                    with gr.Row():
                        tile_size_num = gr.Number(
                            value=config.TILE_SIZE,
                            label="Tile Size (px)",
                            precision=0,
                            scale=1,
                            info="Size of each square tile. 1024 recommended for SDXL.",
                        )
                        overlap_num = gr.Number(
                            value=config.DEFAULT_OVERLAP,
                            label="Overlap (px)",
                            precision=0,
                            scale=1,
                            info="Pixel overlap between adjacent tiles for seamless blending.",
                        )
                    gen_res_dd = gr.Dropdown(
                        choices=_GEN_RES_LABELS,
                        value=_GEN_RES_LABELS[1],  # "1536×1536 ✨ Recommended" default
                        label="Generation Resolution",
                        info=(
                            "Higher generation resolution gives the model more pixels to add detail. "
                            "1536×1536 is the native Illustrious-XL resolution and recommended for best quality. "
                            "The tile grid stays at the grid resolution; only individual tiles are upscaled "
                            "for generation then downscaled back."
                        ),
                    )

                # ---- 🎨 Generation Settings ----
                with gr.Accordion("🎨 Generation Settings", open=True):
                    global_prompt = gr.Textbox(
                        label="Global Style Prompt",
                        placeholder="masterpiece, best quality, highly detailed, sharp focus",
                        lines=2,
                        info="Applied to all tiles unless a tile-specific prompt is set.",
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="blurry, low quality, artifacts, watermark, jpeg artifacts",
                        lines=2,
                        info="Concepts to suppress. Applied to all tiles.",
                    )
                    with gr.Row():
                        strength_sl = gr.Slider(
                            0.0, 1.0, value=0.35, step=0.05,
                            label="Denoise Strength",
                            info="How much to change the tile. 0.3–0.5 recommended.",
                            scale=1,
                        )
                        steps_sl = gr.Slider(
                            1, 100, value=30, step=1,
                            label="Steps",
                            info="Diffusion steps. 20–40 is a good range.",
                            scale=1,
                        )
                    with gr.Row():
                        cfg_sl = gr.Slider(
                            1.0, 20.0, value=7.0, step=0.5,
                            label="CFG Scale",
                            info="Prompt adherence. 6–8 recommended.",
                            scale=1,
                        )
                        seed_num = gr.Number(
                            value=-1,
                            label="Seed",
                            precision=0,
                            scale=1,
                            info="-1 = random seed each run.",
                        )

                # ---- 🔧 ControlNet ----
                with gr.Accordion("🔧 ControlNet", open=False):
                    controlnet_cb = gr.Checkbox(
                        value=True,
                        label="Enable ControlNet (Tile)",
                        info="Uses xinsir/controlnet-tile-sdxl-1.0 to preserve structure.",
                    )
                    cond_scale_sl = gr.Slider(
                        0.0, 1.5, value=0.7, step=0.05,
                        label="Conditioning Scale",
                        info="Strength of ControlNet guidance. 0.5–0.9 recommended.",
                    )

                # ---- 🖼️ IP-Adapter · Style Transfer ----
                with gr.Accordion("🖼️ IP-Adapter  ·  Style Transfer", open=False):
                    gr.Markdown(
                        "Upload a style reference image to guide the diffusion process "
                        "for consistent style across all tiles. "
                        "Uses `h94/IP-Adapter` (ViT-H CLIP encoder, SDXL). "
                        "Adds ~1.6 GB VRAM when enabled."
                    )
                    ip_adapter_cb = gr.Checkbox(
                        value=False,
                        label="Enable IP-Adapter",
                        info="When enabled, the style reference image guides all tile generations.",
                    )
                    ip_style_image = gr.Image(
                        label="Style Reference Image",
                        type="pil",
                        height=160,
                    )
                    ip_scale_sl = gr.Slider(
                        0.0, 1.0, value=0.6, step=0.05,
                        label="IP-Adapter Scale",
                        info="How strongly the style reference influences generation. 0.4–0.7 recommended.",
                    )

                # ---- 🔲 Seam Fix · Grid Offset Pass ----
                with gr.Accordion("🔲 Seam Fix  ·  Grid Offset Pass", open=False):
                    gr.Markdown(
                        "Runs a **second upscale pass** with the tile grid shifted by half a tile "
                        "in both X and Y directions. Each offset tile straddles the seam boundaries "
                        "of Pass 1, so the diffusion model sees the seam area as the centre of a "
                        "tile — giving it full context from both sides. The offset tiles are then "
                        "blended back onto the Pass 1 result using a feathered gradient mask, "
                        "effectively hiding visible seams between tiles.\n\n"
                        "_Only applies to **Upscale All Tiles** — not single-tile upscale._"
                    )
                    seam_fix_cb = gr.Checkbox(
                        value=False,
                        label="Enable Seam Fix (Grid Offset Pass)",
                        info="When enabled, a second offset-grid pass is run after the main upscale.",
                    )
                    seam_fix_strength_sl = gr.Slider(
                        0.1, 0.5, value=0.35, step=0.05,
                        label="Seam Fix Denoise Strength",
                        info="Denoise strength for the offset pass. Lower preserves more detail. 0.3–0.4 recommended.",
                    )
                    seam_fix_feather_sl = gr.Slider(
                        8, 64, value=32, step=8,
                        label="Feather / Blend Size (px)",
                        info="Width of the gradient blend zone at tile edges. Larger = smoother transition.",
                    )

        # ================================================================
        # Event handlers
        # ================================================================

        # ----------------------------------------------------------------
        # Image upload → calculate tiles → build HTML tile grid
        # ----------------------------------------------------------------
        def on_upload(img, res, ts, ov):
            return _on_image_upload(img, res, int(ts), int(ov))

        _upload_outputs = [
            tiles_state, selected_idx_state, original_img_state,
            grid_cols_state, full_image_b64_state,
            tile_grid_html, status_text,
        ]

        image_input.upload(
            fn=on_upload,
            inputs=[image_input, resolution_dd, tile_size_num, overlap_num],
            outputs=_upload_outputs,
        )

        image_input.change(
            fn=on_upload,
            inputs=[image_input, resolution_dd, tile_size_num, overlap_num],
            outputs=_upload_outputs,
        )

        # Regenerate grid button (re-runs the same upload logic)
        regen_grid_btn.click(
            fn=on_upload,
            inputs=[image_input, resolution_dd, tile_size_num, overlap_num],
            outputs=_upload_outputs,
        )

        # ----------------------------------------------------------------
        # JS tile click → hidden textbox → Python selection handler
        # ----------------------------------------------------------------
        def on_tile_selected(
            idx_str: str,
            tiles: List[Dict],
            full_b64: str,
            orig_img: Optional[Image.Image],
            overlap: float,
        ):
            """Handle tile selection from JS: update selected index and detail panel."""
            try:
                idx = int(idx_str)
            except (ValueError, TypeError):
                idx = -1

            if not tiles or idx < 0 or idx >= len(tiles):
                grid_html = _rebuild_grid_html(tiles, full_b64, -1, orig_img)
                return -1, None, None, "", grid_html

            tile = tiles[idx]
            prompt = tile.get("prompt", "")

            # Show original tile with mask blur (feather zone) overlay
            orig_tile_pil = _b64_to_pil(tile["original_b64"])
            if orig_img is not None:
                img_w, img_h = orig_img.size
                tile_info: TileInfo = tile["info"]
                overlap_px = max(1, int(overlap))
                orig_tile = generate_mask_blur_preview(
                    orig_tile_pil, tile_info, overlap_px, img_w, img_h
                )
            else:
                orig_tile = orig_tile_pil

            # Show processed tile if available
            proc_tile = _b64_to_pil(tile["processed_b64"]) if tile.get("processed_b64") else None

            # Rebuild grid HTML with new selection highlighted
            grid_html = _rebuild_grid_html(tiles, full_b64, idx, orig_img)

            return idx, orig_tile, proc_tile, prompt, grid_html

        tile_selected_idx_tb.change(
            fn=on_tile_selected,
            inputs=[tile_selected_idx_tb, tiles_state, full_image_b64_state, original_img_state, overlap_num],
            outputs=[selected_idx_state, tile_orig_preview, tile_proc_preview, tile_prompt_box, tile_grid_html],
        )

        # ----------------------------------------------------------------
        # Tile prompt editing → update tiles_state
        # ----------------------------------------------------------------
        tile_prompt_box.change(
            fn=_on_prompt_edit,
            inputs=[tile_prompt_box, selected_idx_state, tiles_state],
            outputs=[tiles_state],
        )

        # ----------------------------------------------------------------
        # Upscale All Tiles
        # ----------------------------------------------------------------
        def on_upscale_all(
            tiles, orig_img,
            g_prompt, neg_prompt,
            ts, ov,
            strength, steps, cfg, seed,
            cn_enabled, cond_scale,
            ip_enabled, ip_img, ip_scale,
            seam_fix_enabled, seam_fix_strength, seam_fix_feather,
            gen_res_label,
            full_b64,
        ):
            """Upscale all tiles and return updated grid HTML + assembled result.

            When seam_fix_enabled is True, a second pass is run with the tile
            grid shifted by half a stride in both X and Y.  The offset tiles
            are extracted from the Pass-1 result, processed at the lower
            seam_fix_strength, then blended back using a feathered mask.
            """
            tile_size = int(ts)
            overlap   = int(ov)
            gen_res_val = _GEN_RES_LABEL_TO_VALUE.get(gen_res_label, 0)

            # ------------------------------------------------------------------
            # Pass 1 — standard tile grid
            # ------------------------------------------------------------------
            updated, result, msg = _upscale_tiles_batch(
                tiles, None, orig_img,
                g_prompt, neg_prompt,
                tile_size, overlap,
                strength, int(steps), cfg, int(seed),
                cn_enabled, cond_scale,
                client,
                ip_adapter_enabled=ip_enabled,
                ip_adapter_image=ip_img,
                ip_adapter_scale=ip_scale,
                gen_res=gen_res_val,
            )

            if result is None:
                # Pass 1 failed — return early
                grid_html = _rebuild_grid_html(updated, full_b64, -1, orig_img)
                return updated, grid_html, result, msg

            # ------------------------------------------------------------------
            # Pass 2 — offset grid seam fix (optional)
            # ------------------------------------------------------------------
            if seam_fix_enabled:
                try:
                    offset_tile_infos = calculate_offset_tiles(
                        result.size,
                        tile_size=tile_size,
                        overlap=overlap,
                    )

                    if offset_tile_infos:
                        # Extract offset tiles from the Pass-1 assembled result
                        offset_tile_imgs = [
                            extract_tile(result, ti) for ti in offset_tile_infos
                        ]

                        effective_gen_res = gen_res_val if gen_res_val and gen_res_val > tile_size else tile_size

                        # Build payload for the offset pass
                        offset_payload = []
                        for i, (ti, tile_img) in enumerate(
                            zip(offset_tile_infos, offset_tile_imgs)
                        ):
                            # Upscale offset tile to generation resolution if needed
                            if effective_gen_res != tile_size:
                                tile_img = tile_img.resize(
                                    (effective_gen_res, effective_gen_res), Image.LANCZOS
                                )
                            entry: Dict[str, Any] = {
                                "tile_id": f"seam_{ti.row}_{ti.col}",
                                "image_b64": _pil_to_b64(tile_img),
                                "prompt_override": None,
                                "model": ACTIVE_MODEL,
                                "loras": HARDCODED_LORAS,
                                "global_prompt": g_prompt,
                                "negative_prompt": neg_prompt,
                                "controlnet_enabled": cn_enabled,
                                "conditioning_scale": cond_scale,
                                "strength": float(seam_fix_strength),
                                "steps": int(steps),
                                "cfg_scale": cfg,
                                "seed": int(seed),
                                "ip_adapter_enabled": False,
                                "target_width": effective_gen_res,
                                "target_height": effective_gen_res,
                            }
                            offset_payload.append(entry)

                        # Send offset tiles to the backend
                        offset_results_raw = client.upscale_tiles(offset_payload)
                        result_map = {
                            r["tile_id"]: r["image_b64"] for r in offset_results_raw
                        }

                        # Reconstruct processed offset tile images in order
                        offset_processed: List[Image.Image] = []
                        for ti in offset_tile_infos:
                            tid = f"seam_{ti.row}_{ti.col}"
                            if tid in result_map:
                                proc_img = _b64_to_pil(result_map[tid])
                                # Downscale back to tile_size if needed
                                if effective_gen_res != tile_size:
                                    proc_img = proc_img.resize(
                                        (tile_size, tile_size), Image.LANCZOS
                                    )
                                offset_processed.append(proc_img)
                            else:
                                # Fallback: use the original extracted tile
                                offset_processed.append(
                                    extract_tile(result, ti)
                                )

                        # Blend offset tiles onto the Pass-1 result
                        result = blend_offset_pass(
                            result,
                            offset_tile_infos,
                            offset_processed,
                            feather_size=int(seam_fix_feather),
                        )

                        n_offset = len(offset_tile_infos)
                        msg = (
                            msg.rstrip(".")
                            + f" | Pass 2/2: Seam fix applied ({n_offset} offset tiles)."
                        )
                except RunPodError as exc:
                    msg = msg.rstrip(".") + f" | ⚠️ Seam fix error: {exc}"
                except Exception as exc:  # noqa: BLE001
                    msg = msg.rstrip(".") + f" | ⚠️ Seam fix error: {exc}"

            # Rebuild grid HTML to show processed checkmarks
            grid_html = _rebuild_grid_html(updated, full_b64, -1, orig_img)
            return updated, grid_html, result, msg

        upscale_all_btn.click(
            fn=on_upscale_all,
            inputs=[
                tiles_state, original_img_state,
                global_prompt, negative_prompt,
                tile_size_num, overlap_num,
                strength_sl, steps_sl, cfg_sl, seed_num,
                controlnet_cb, cond_scale_sl,
                ip_adapter_cb, ip_style_image, ip_scale_sl,
                seam_fix_cb, seam_fix_strength_sl, seam_fix_feather_sl,
                gen_res_dd,
                full_image_b64_state,
            ],
            outputs=[tiles_state, tile_grid_html, result_image, status_text],
        )

        # ----------------------------------------------------------------
        # Upscale Selected Tile
        # ----------------------------------------------------------------
        def on_upscale_selected(
            tiles, selected_idx, orig_img,
            g_prompt, neg_prompt,
            ts, ov,
            strength, steps, cfg, seed,
            cn_enabled, cond_scale,
            ip_enabled, ip_img, ip_scale,
            gen_res_label,
            full_b64,
        ):
            """Upscale only the currently selected tile."""
            if selected_idx < 0:
                grid_html = _rebuild_grid_html(tiles, full_b64, selected_idx, orig_img)
                return tiles, grid_html, None, None, "⚠️ No tile selected. Click a tile in the grid first."
            gen_res_val = _GEN_RES_LABEL_TO_VALUE.get(gen_res_label, 0)
            updated, result, msg = _upscale_tiles_batch(
                tiles, [selected_idx], orig_img,
                g_prompt, neg_prompt,
                int(ts), int(ov),
                strength, int(steps), cfg, int(seed),
                cn_enabled, cond_scale,
                client,
                ip_adapter_enabled=ip_enabled,
                ip_adapter_image=ip_img,
                ip_adapter_scale=ip_scale,
                gen_res=gen_res_val,
            )
            # Rebuild grid HTML with updated tile status, keep selection
            grid_html = _rebuild_grid_html(updated, full_b64, selected_idx, orig_img)
            tile = updated[selected_idx]
            proc_tile = _b64_to_pil(tile["processed_b64"]) if tile.get("processed_b64") else None
            return updated, grid_html, proc_tile, result, msg

        upscale_sel_btn.click(
            fn=on_upscale_selected,
            inputs=[
                tiles_state, selected_idx_state, original_img_state,
                global_prompt, negative_prompt,
                tile_size_num, overlap_num,
                strength_sl, steps_sl, cfg_sl, seed_num,
                controlnet_cb, cond_scale_sl,
                ip_adapter_cb, ip_style_image, ip_scale_sl,
                gen_res_dd,
                full_image_b64_state,
            ],
            outputs=[tiles_state, tile_grid_html, tile_proc_preview, result_image, status_text],
        )

        # ----------------------------------------------------------------
        # Auto-caption all tiles
        # ----------------------------------------------------------------
        def on_caption_all(tiles, sys_prompt: str, full_b64: str, orig_img):
            updated, msg = _caption_all_tiles(tiles, sys_prompt, client)
            grid_html = _rebuild_grid_html(updated, full_b64, -1, orig_img)
            return updated, grid_html, msg

        caption_btn.click(
            fn=on_caption_all,
            inputs=[tiles_state, global_prompt, full_image_b64_state, original_img_state],
            outputs=[tiles_state, tile_grid_html, status_text],
        )

        # ----------------------------------------------------------------
        # Auto-caption selected tile
        # ----------------------------------------------------------------
        def on_caption_selected(tiles, selected_idx, sys_prompt: str):
            updated, caption_or_msg = _caption_selected_tile(selected_idx, tiles, sys_prompt, client)
            # If captioning succeeded, caption_or_msg is the caption text; put it in the prompt box
            if caption_or_msg.startswith("⚠️") or caption_or_msg.startswith("❌"):
                return updated, caption_or_msg, gr.update()
            return updated, f"✅ Captioned tile {selected_idx}.", caption_or_msg

        caption_sel_btn.click(
            fn=on_caption_selected,
            inputs=[tiles_state, selected_idx_state, global_prompt],
            outputs=[tiles_state, status_text, tile_prompt_box],
        )

        # ----------------------------------------------------------------
        # Download Result
        # ----------------------------------------------------------------
        def on_download_result(result_img: Optional[Image.Image]):
            if result_img is None:
                return None
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            result_img.save(tmp.name, format="PNG")
            tmp.close()
            return tmp.name

        download_btn.click(
            fn=on_download_result,
            inputs=[result_image],
            outputs=[download_file],
        )

        # ----------------------------------------------------------------
        # Copy Output → Input
        # ----------------------------------------------------------------
        def on_copy_to_input(result_img: Optional[Image.Image]):
            return result_img

        copy_to_input_btn.click(
            fn=on_copy_to_input,
            inputs=[result_image],
            outputs=[image_input],
        )
