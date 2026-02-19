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
loras_state          : list[dict]  — active LoRAs [{name, weight}, ...]
full_image_b64_state : str         — base64 JPEG of the full upscaled image for grid display
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
    "8K × 8K (8192×8192)",
    "8K × 4K (8192×4096)",
    "4K × 4K (4096×4096)",
    "4K × 2K (4096×2048)",
    "Custom W×H",
]

# Resolution preset values
RESOLUTION_VALUES = {
    "8K × 8K (8192×8192)": (8192, 8192),
    "8K × 4K (8192×4096)": (8192, 4096),
    "4K × 4K (4096×4096)": (4096, 4096),
    "4K × 2K (4096×2048)": (4096, 2048),
}

# The single active model (hardcoded — only illustrious-xl is supported)
ACTIVE_MODEL = "illustrious-xl"

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
    custom_w: int,
    custom_h: int,
) -> Tuple[int, int]:
    if resolution_choice in RESOLUTION_VALUES:
        return RESOLUTION_VALUES[resolution_choice]
    return int(custom_w), int(custom_h)


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
.tile-grid {
  display: grid;
  gap: 0;
  width: 100%;
  border-radius: 8px;
  overflow: hidden;
  background: #000;
}
.tile {
  position: relative;
  aspect-ratio: 1 / 1;
  background-repeat: no-repeat;
  border: none;
  border-radius: 0;
  cursor: pointer;
  overflow: hidden;
  /* Thin semi-transparent grid line overlay via inset box-shadow */
  box-shadow: inset 0 0 0 0.5px rgba(255, 255, 255, 0.12);
  transition: box-shadow 0.15s ease;
}
/* ── Selection dimming ─────────────────────────────────────────────────── */
/* When any tile is selected, dim all non-selected tiles */
.tile-grid.has-selection .tile:not(.selected)::before {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.45);
  pointer-events: none;
  z-index: 2;
  transition: background 0.15s ease;
}
/* Selected tile stays bright and on top */
.tile-grid .tile.selected {
  z-index: 3;
  box-shadow: inset 0 0 0 0.5px rgba(255, 255, 255, 0.12),
              inset 0 0 0 2px rgba(30, 111, 255, 0.8);
}
/* ── Status indicators ─────────────────────────────────────────────────── */
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
/* ── Tile labels & icons ───────────────────────────────────────────────── */
.tile-label {
  position: absolute;
  top: 3px;
  left: 3px;
  background: rgba(0, 0, 0, 0.7);
  color: #fff;
  font-size: 11px;
  padding: 1px 5px;
  border-radius: 3px;
  pointer-events: none;
  font-family: monospace;
  z-index: 5;
}
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
/* ── Overlap zone strips ───────────────────────────────────────────────── */
.tile .overlap-zone {
  position: absolute;
  pointer-events: none;
  z-index: 4;
  display: none;
}
.tile.selected .overlap-zone {
  display: block;
}
.tile .overlap-zone.left {
  left: 0;
  top: 0;
  bottom: 0;
  background: linear-gradient(to right, rgba(255, 165, 0, 0.35), transparent);
}
.tile .overlap-zone.right {
  right: 0;
  top: 0;
  bottom: 0;
  background: linear-gradient(to left, rgba(255, 165, 0, 0.35), transparent);
}
.tile .overlap-zone.top {
  left: 0;
  right: 0;
  top: 0;
  background: linear-gradient(to bottom, rgba(255, 165, 0, 0.35), transparent);
}
.tile .overlap-zone.bottom {
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(to top, rgba(255, 165, 0, 0.35), transparent);
}
@media (max-width: 900px) {
  .tile-label { font-size: 9px; }
}
</style>
"""

_TILE_GRID_JS = """
<script>
(function() {
  var root = document.getElementById('tile-grid-root');
  if (!root) return;

  var grid = root.querySelector('.tile-grid');
  var tiles = root.querySelectorAll('.tile');
  var selectedIdx = parseInt(root.dataset.selected, 10);

  // Apply selected class and has-selection from server-side state
  tiles.forEach(function(tile) {
    tile.classList.toggle('selected', parseInt(tile.dataset.idx, 10) === selectedIdx);
  });
  if (selectedIdx >= 0 && grid) {
    grid.classList.add('has-selection');
  }

  // Click handler — write to hidden textbox to notify Python
  root.addEventListener('click', function(e) {
    var tileEl = e.target.closest('.tile');
    if (!tileEl) return;

    var idx = parseInt(tileEl.dataset.idx, 10);
    var wasSelected = tileEl.classList.contains('selected');

    // Update visual selection immediately (no round-trip needed)
    tiles.forEach(function(t) { t.classList.remove('selected'); });

    if (wasSelected) {
      // Deselect: click same tile again
      if (grid) grid.classList.remove('has-selection');
      idx = -1;
    } else {
      tileEl.classList.add('selected');
      if (grid) grid.classList.add('has-selection');
    }

    // Write to hidden Gradio textbox to trigger Python .change() handler
    var hiddenInput = document.querySelector('#tile-selected-idx textarea');
    if (hiddenInput) {
      var nativeInputValueSetter =
        Object.getOwnPropertyDescriptor(
          window.HTMLTextAreaElement.prototype, 'value'
        ).set;
      nativeInputValueSetter.call(hiddenInput, String(idx));
      hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
    }
  });
})();
</script>
"""


def _build_tile_grid_html(
    tiles_data: List[Dict],
    full_image_b64: str,
    selected_idx: int,
    img_w: int,
    img_h: int,
    overlap: int = 128,
) -> str:
    """Build the HTML/CSS/JS string for the custom tile grid component.

    Args:
        tiles_data: List of tile dicts with 'info' (TileInfo), 'processed_b64' (optional).
                    The TileInfo coords must already be in display-image space.
        full_image_b64: Base64 JPEG of the full (possibly downscaled) image for display
        selected_idx: Currently selected tile index (-1 for none)
        img_w: Full display image width in pixels
        img_h: Full display image height in pixels
        overlap: Overlap in original image pixels (used for overlap zone visualization)

    Returns:
        HTML string with embedded CSS and JS
    """
    if not tiles_data:
        return "<p style='color:#888; padding:16px;'>Upload an image to see the tile grid.</p>"

    # Determine grid dimensions from tile infos
    grid_cols = max(t["info"].col for t in tiles_data) + 1
    grid_rows = max(t["info"].row for t in tiles_data) + 1

    has_selection_class = " has-selection" if selected_idx >= 0 else ""

    tile_divs = []
    for i, tile in enumerate(tiles_data):
        ti: TileInfo = tile["info"]
        is_processed = bool(tile.get("processed_b64"))
        status = "processed" if is_processed else "default"
        selected_class = " selected" if i == selected_idx else ""

        if is_processed and tile.get("processed_b64"):
            # Show the processed tile image directly as background (full tile fill)
            proc_b64 = tile["processed_b64"]
            bg_style = (
                f"background-image: url('data:image/jpeg;base64,{proc_b64}');"
                "background-size: 100% 100%;"
                "background-position: 0% 0%;"
            )
        else:
            # Crop the full image using CSS background-position math.
            # background-size: (img_w / tile_w * 100)% (img_h / tile_h * 100)%
            # background-position: (x / (img_w - tile_w) * 100)% (y / (img_h - tile_h) * 100)%
            tile_w = ti.w
            tile_h = ti.h

            bg_size_x = img_w / tile_w * 100 if tile_w > 0 else 100
            bg_size_y = img_h / tile_h * 100 if tile_h > 0 else 100

            bp_x = ti.x / (img_w - tile_w) * 100 if img_w > tile_w else 0.0
            bp_y = ti.y / (img_h - tile_h) * 100 if img_h > tile_h else 0.0

            bg_style = (
                f"background-image: url('data:image/jpeg;base64,{full_image_b64}');"
                f"background-size: {bg_size_x:.4f}% {bg_size_y:.4f}%;"
                f"background-position: {bp_x:.4f}% {bp_y:.4f}%;"
            )

        label = f"{ti.row},{ti.col}"

        # Compute overlap zone widths as percentage of tile display size
        # overlap is in original image pixels; tile display size is ti.w / ti.h pixels
        overlap_pct_x = (overlap / ti.w * 100) if ti.w > 0 else 0.0
        overlap_pct_y = (overlap / ti.h * 100) if ti.h > 0 else 0.0

        # Build overlap zone divs (shown only when tile is selected)
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
            f'style="{bg_style}">'
            f'<span class="tile-label">{label}</span>'
            f'<span class="tile-status-icon"></span>'
            f'{overlap_html}'
            f'</div>'
        )

    tiles_html = "\n".join(tile_divs)
    aspect = f"{img_w} / {img_h}"

    html = (
        f'<div id="tile-grid-root" '
        f'data-rows="{grid_rows}" data-cols="{grid_cols}" '
        f'data-img-w="{img_w}" data-img-h="{img_h}" '
        f'data-selected="{selected_idx}" '
        f'data-overlap="{overlap}">'
        f'<div class="tile-grid{has_selection_class}" '
        f'style="grid-template-columns: repeat({grid_cols}, 1fr); aspect-ratio: {aspect};">'
        f'{tiles_html}'
        f'</div>'
        f'</div>'
        f'{_TILE_GRID_CSS}'
        f'{_TILE_GRID_JS}'
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
    custom_w: int,
    custom_h: int,
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

    target_w, target_h = _compute_target_size(image.size, resolution_choice, custom_w, custom_h)
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


def _upscale_tiles_batch(
    tiles_state: List[Dict],
    indices: Optional[List[int]],
    original_img: Optional[Image.Image],
    # generation params
    loras: List[Dict],
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
            "loras": loras if loras else [],
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
# LoRA HTML renderer (module-level so it can be reused)
# ---------------------------------------------------------------------------

def _render_loras_html(loras: List[Dict]) -> str:
    if not loras:
        return "<p style='color: #888; font-size: 0.85em;'>No LoRAs added. Select a LoRA above and click ➕ Add.</p>"
    rows = ""
    for i, entry in enumerate(loras):
        name = entry.get("name", "")
        weight = entry.get("weight", 1.0)
        bg = "#1e2030" if i % 2 == 0 else "#252840"
        rows += (
            f"<div style='display:flex;align-items:center;gap:8px;"
            f"padding:5px 8px;border-bottom:1px solid #333;background:{bg};'>"
            f"<span style='flex:1;font-size:0.85em;overflow:hidden;text-overflow:ellipsis;"
            f"white-space:nowrap;color:#ddd;' title='{name}'>{name}</span>"
            f"<span style='font-size:0.8em;color:#aaa;min-width:44px;text-align:right;'>"
            f"&times;{weight:.2f}</span>"
            f"</div>"
        )
    return (
        "<div style='border:1px solid #444;border-radius:6px;max-height:160px;overflow-y:auto;'>"
        + rows
        + "</div>"
        + f"<p style='font-size:0.75em;color:#888;margin:4px 0 0;'>{len(loras)} LoRA(s) active</p>"
    )


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
        loras_state = gr.State(value=[])
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
                    value="4K × 4K (4096×4096)",
                    label="Target Resolution",
                    info="The image will be upscaled to this resolution before tiling.",
                )
                with gr.Row():
                    custom_w = gr.Number(
                        value=2048, label="Width (px)", precision=0,
                        visible=False, scale=1,
                    )
                    custom_h = gr.Number(
                        value=2048, label="Height (px)", precision=0,
                        visible=False, scale=1,
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
                            label="Original",
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

                # ---- Final result ----
                gr.Markdown("### 🖼️ Final Result")
                result_image = gr.Image(
                    label="Assembled Upscaled Image",
                    type="pil",
                    interactive=False,
                    height=400,
                )

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

                # ---- 📁 LoRA ----
                with gr.Accordion("📁 LoRA", open=False):
                    with gr.Row():
                        lora_add_dd = gr.Dropdown(
                            choices=["None"],
                            value="None",
                            label="Select LoRA",
                            info="LoRAs are loaded from the RunPod network volume. Use the LoRA Manager tab to upload new ones.",
                            scale=3,
                        )
                        lora_add_weight = gr.Slider(
                            minimum=0.0, maximum=2.0, value=0.7, step=0.05,
                            label="Weight",
                            info="Blend strength for this LoRA. 0.5–1.0 is typical; >1.0 may over-saturate the style.",
                            scale=2,
                        )
                    with gr.Row():
                        lora_add_btn = gr.Button("➕ Add LoRA", variant="secondary", scale=1)
                        lora_remove_btn = gr.Button("➖ Remove Last", variant="secondary", scale=1)
                        lora_refresh_btn = gr.Button("🔄 Refresh", variant="secondary", scale=1)
                    loras_display = gr.HTML(
                        "<p style='color: #888; font-size: 0.85em;'>No LoRAs added.</p>"
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
        # Show/hide custom resolution fields
        # ----------------------------------------------------------------
        def _toggle_custom(choice: str):
            visible = choice == "Custom W×H"
            return gr.update(visible=visible), gr.update(visible=visible)

        resolution_dd.change(
            fn=_toggle_custom,
            inputs=[resolution_dd],
            outputs=[custom_w, custom_h],
        )

        # ----------------------------------------------------------------
        # Image upload → calculate tiles → build HTML tile grid
        # ----------------------------------------------------------------
        def on_upload(img, res, cw, ch, ts, ov):
            return _on_image_upload(img, res, cw, ch, int(ts), int(ov))

        _upload_outputs = [
            tiles_state, selected_idx_state, original_img_state,
            grid_cols_state, full_image_b64_state,
            tile_grid_html, status_text,
        ]

        image_input.upload(
            fn=on_upload,
            inputs=[image_input, resolution_dd, custom_w, custom_h, tile_size_num, overlap_num],
            outputs=_upload_outputs,
        )

        image_input.change(
            fn=on_upload,
            inputs=[image_input, resolution_dd, custom_w, custom_h, tile_size_num, overlap_num],
            outputs=_upload_outputs,
        )

        # Regenerate grid button (re-runs the same upload logic)
        regen_grid_btn.click(
            fn=on_upload,
            inputs=[image_input, resolution_dd, custom_w, custom_h, tile_size_num, overlap_num],
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

            # Show original tile
            orig_tile = _b64_to_pil(tile["original_b64"])
            # Show processed tile if available
            proc_tile = _b64_to_pil(tile["processed_b64"]) if tile.get("processed_b64") else None

            # Rebuild grid HTML with new selection highlighted
            grid_html = _rebuild_grid_html(tiles, full_b64, idx, orig_img)

            return idx, orig_tile, proc_tile, prompt, grid_html

        tile_selected_idx_tb.change(
            fn=on_tile_selected,
            inputs=[tile_selected_idx_tb, tiles_state, full_image_b64_state, original_img_state],
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
        # Populate LoRA dropdown on tab load
        # ----------------------------------------------------------------
        def _load_loras():
            try:
                models = client.list_models("lora")
                names = ["None"] + [m.get("name", "") for m in models if m.get("name")]
                return gr.update(choices=names, value="None")
            except Exception:  # noqa: BLE001
                return gr.update(choices=["None"], value="None")

        # ----------------------------------------------------------------
        # Multi-LoRA add / remove
        # ----------------------------------------------------------------
        def _add_lora_fn(loras: List[Dict], name: str, weight: float) -> Tuple[List[Dict], str]:
            if not name or name == "None":
                return loras, _render_loras_html(loras)
            updated = [dict(e) for e in loras]
            for entry in updated:
                if entry["name"] == name:
                    entry["weight"] = weight
                    return updated, _render_loras_html(updated)
            updated.append({"name": name, "weight": weight})
            return updated, _render_loras_html(updated)

        def _remove_last_lora_fn(loras: List[Dict]) -> Tuple[List[Dict], str]:
            updated = list(loras)
            if updated:
                updated.pop()
            return updated, _render_loras_html(updated)

        lora_add_btn.click(
            fn=_add_lora_fn,
            inputs=[loras_state, lora_add_dd, lora_add_weight],
            outputs=[loras_state, loras_display],
        )

        lora_remove_btn.click(
            fn=_remove_last_lora_fn,
            inputs=[loras_state],
            outputs=[loras_state, loras_display],
        )

        lora_refresh_btn.click(
            fn=_load_loras,
            inputs=[],
            outputs=[lora_add_dd],
        )

        # ----------------------------------------------------------------
        # Upscale All Tiles
        # ----------------------------------------------------------------
        def on_upscale_all(
            tiles, orig_img,
            loras,
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
                loras,
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
                                "loras": loras if loras else [],
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
                loras_state,
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
            loras,
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
                loras,
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
                loras_state,
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
