"""
Gradio UI for the AI Assets Toolbox — served as a Modal ASGI app.

This module creates the Gradio ``Blocks`` interface that runs in a lightweight
CPU container on Modal.  GPU work is dispatched to the upscale and caption
services via Modal ``.remote()`` calls — no HTTP, no base64 transport.

Usage
-----
The main entrypoint (``modal_app.py``) imports :func:`create_gradio_app` and
wraps it with ``@modal.asgi_app()``:

.. code-block:: python

    from src.ui.gradio_app import create_gradio_app

    @app.function(image=gradio_image)
    @modal.asgi_app()
    def ui():
        return create_gradio_app()

Architecture
------------
* **Upscale workflow** — the UI splits the uploaded image into tiles locally
  (using :mod:`src.tiling`), converts each tile to PNG bytes, and sends the
  batch to :class:`~src.gpu.upscale.UpscaleService` via ``.remote()``.  The
  returned PNG bytes are decoded back to PIL Images and merged with feathered
  blending.

* **Caption workflow** — tile PNG bytes are sent to
  :class:`~src.gpu.caption.CaptionService` via ``.remote()``.  The returned
  caption strings are stored as per-tile prompt overrides.

* **Model manager** — LoRA and checkpoint management via
  :class:`~src.services.download.DownloadService`.  Metadata (trigger words,
  weights, base model) is stored in ``.metadata.json`` on the LoRA volume.
  On first page load, :meth:`~src.services.download.DownloadService.migrate_metadata`
  seeds the metadata store from the legacy hardcoded LoRA list.
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image

from src.tiling import (
    TileInfo,
    calculate_tiles,
    calculate_offset_tiles,
    extract_tile,
    extract_all_tiles,
    blend_tiles,
    blend_offset_pass,
    generate_mask_blur_preview,
    upscale_image as bicubic_upscale,
)

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

# Resolution presets for the target-resolution dropdown
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

# Maximum resolution cap (prevents OOM on very large images)
MAX_RESOLUTION = 8192

# Default tile / overlap settings
DEFAULT_TILE_SIZE = 1024
DEFAULT_OVERLAP = 128

# ---------------------------------------------------------------------------
# PIL ↔ bytes helpers
# ---------------------------------------------------------------------------


def _pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Encode a PIL Image to raw bytes (PNG by default)."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _bytes_to_pil(data: bytes) -> Image.Image:
    """Decode raw image bytes (PNG/JPEG) to a PIL Image (RGB)."""
    return Image.open(io.BytesIO(data)).convert("RGB")


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _compute_target_size(
    original_size: Tuple[int, int],
    resolution_choice: str,
) -> Tuple[int, int]:
    """Compute target (width, height) from the long-side resolution preset."""
    orig_w, orig_h = original_size
    long_side = RESOLUTION_LONG_SIDE.get(resolution_choice)
    if long_side is None:
        return orig_w, orig_h
    if orig_w >= orig_h:
        target_w = long_side
        target_h = max(1, int(orig_h * long_side / orig_w))
    else:
        target_h = long_side
        target_w = max(1, int(orig_w * long_side / orig_h))
    return target_w, target_h


def _compute_display_dims(orig_w: int, orig_h: int, max_side: int = 2048) -> Tuple[int, int]:
    """Compute display image dimensions after optional downscale."""
    if max(orig_w, orig_h) > max_side:
        scale = max_side / max(orig_w, orig_h)
        return int(orig_w * scale), int(orig_h * scale)
    return orig_w, orig_h


# ---------------------------------------------------------------------------
# Trigger word injection
# ---------------------------------------------------------------------------


def _inject_trigger_words(
    prompt: str,
    lora_states: List[Dict[str, Any]],
    available_loras: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Prepend trigger words for active LoRAs to the prompt.

    Trigger words are sourced from *available_loras* (fetched dynamically from
    :class:`~src.services.download.DownloadService`).  If *available_loras* is
    ``None`` or empty the function returns the prompt unchanged — actual
    injection happens server-side in the GPU container.

    Args:
        prompt:          the base prompt string.
        lora_states:     list of dicts with ``"enabled"`` (bool) and
                         ``"name"`` (str).
        available_loras: list of model-info dicts as returned by
                         ``DownloadService().list_user_models.remote()``.
                         Each dict has at least ``"name"`` and
                         ``"trigger_words"`` keys.

    Returns:
        Updated prompt string with trigger words prepended.
    """
    if not available_loras:
        return prompt
    lora_map: Dict[str, List[str]] = {
        m["name"]: m.get("trigger_words", []) for m in available_loras
    }
    injected: List[str] = []
    for state in lora_states:
        if not state.get("enabled", True):
            continue
        name = state.get("name", "")
        for word in lora_map.get(name, []):
            if word not in prompt and word not in injected:
                injected.append(word)
    if injected:
        trigger_text = ", ".join(injected)
        return f"{trigger_text}, {prompt}" if prompt else trigger_text
    return prompt


# ---------------------------------------------------------------------------
# HTML tile grid builder
# ---------------------------------------------------------------------------

_TILE_GRID_CSS = """
<style>
.tile-grid-wrap {
  position: relative;
  width: 100%;
  border-radius: 8px;
  overflow: hidden;
  background-size: 100% 100%;
  background-repeat: no-repeat;
}
.tile {
  position: absolute;
  cursor: pointer;
  overflow: hidden;
  box-shadow: inset 0 0 0 0.5px rgba(255, 255, 255, 0.12);
  transition: box-shadow 0.15s ease;
  background-repeat: no-repeat;
  background-size: 100% 100%;
}
.tile.selected {
  z-index: 3;
  box-shadow: inset 0 0 0 0.5px rgba(255, 255, 255, 0.12),
              inset 0 0 0 2px rgba(30, 111, 255, 0.8);
}
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
</style>
"""

# Inline onclick JS — embedded on each tile div so it fires after Gradio re-renders.
# Tries both <textarea> and <input> to be compatible with Gradio 4.x and 6.x.
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
    "var tb=document.querySelector('#tile-selected-idx textarea')"
    "||document.querySelector('#tile-selected-idx input');"
    "if(tb){"
    "var proto=tb.tagName==='TEXTAREA'?window.HTMLTextAreaElement.prototype:window.HTMLInputElement.prototype;"
    "var ns=Object.getOwnPropertyDescriptor(proto,'value').set;"
    "ns.call(tb,wasSelected?'-1':el.dataset.idx);"
    "tb.dispatchEvent(new Event('input',{bubbles:true}));"
    "tb.dispatchEvent(new Event('change',{bubbles:true}));"
    "}"
    "})(this)"
)


def _encode_grid_image(img: Image.Image, max_side: int = 2048, quality: int = 75) -> str:
    """Encode a PIL image as a JPEG base64 string for use in the tile grid."""
    import base64
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


def _build_tile_grid_html(
    tiles_data: List[Dict],
    full_image_b64: str,
    selected_idx: int,
    img_w: int,
    img_h: int,
    overlap: int = 128,
) -> str:
    """Build the HTML/CSS string for the custom tile grid component."""
    if not tiles_data:
        return "<p style='color:#888; padding:16px;'>Upload an image to see the tile grid.</p>"

    grid_cols = max(t["info"].col for t in tiles_data) + 1
    grid_rows = max(t["info"].row for t in tiles_data) + 1

    has_selection_class = " has-selection" if selected_idx >= 0 else ""
    padding_bottom_pct = (img_h / img_w * 100) if img_w > 0 else 100.0

    tile_divs = []
    for i, tile in enumerate(tiles_data):
        ti: TileInfo = tile["info"]
        is_processed = bool(tile.get("processed_bytes"))
        status = "processed" if is_processed else "default"
        selected_class = " selected" if i == selected_idx else ""

        left_pct   = ti.x / img_w * 100 if img_w > 0 else 0.0
        top_pct    = ti.y / img_h * 100 if img_h > 0 else 0.0
        width_pct  = ti.w / img_w * 100 if img_w > 0 else 100.0
        height_pct = ti.h / img_h * 100 if img_h > 0 else 100.0

        pos_style = (
            f"left:{left_pct:.4f}%;top:{top_pct:.4f}%;"
            f"width:{width_pct:.4f}%;height:{height_pct:.4f}%;"
        )

        if is_processed and tile.get("processed_b64_display"):
            proc_b64 = tile["processed_b64_display"]
            bg_style = (
                f"background-image:url('data:image/jpeg;base64,{proc_b64}');"
                "background-size:100% 100%;"
            )
        else:
            bg_style = ""

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


def _rebuild_grid_html(
    tiles_state: List[Dict],
    full_image_b64: str,
    selected_idx: int,
    original_img: Optional[Image.Image],
    overlap: int = 128,
) -> str:
    """Rebuild the tile grid HTML from current state."""
    _empty = "<p style='color:#888; padding:16px;'>Upload an image to see the tile grid.</p>"
    if not tiles_state or original_img is None or not full_image_b64:
        return _empty
    orig_w, orig_h = original_img.size
    display_w, display_h = _compute_display_dims(orig_w, orig_h, max_side=2048)
    scaled_tiles = _scale_tiles_for_display(tiles_state, orig_w, orig_h, display_w, display_h)
    return _build_tile_grid_html(scaled_tiles, full_image_b64, selected_idx, display_w, display_h, overlap)


# ---------------------------------------------------------------------------
# Tile state helpers
# ---------------------------------------------------------------------------


def _build_tile_state(tile_info: TileInfo, tile_img: Image.Image) -> Dict[str, Any]:
    """Create the initial state dict for a single tile."""
    return {
        "info": tile_info,
        "tile_id": tile_info.tile_id,
        "original_bytes": _pil_to_bytes(tile_img),
        "processed_bytes": None,
        "processed_b64_display": None,  # JPEG b64 for grid display only
        "prompt": "",
    }


def _tile_pil(tile: Dict[str, Any], use_processed: bool = False) -> Image.Image:
    """Decode a tile's image bytes to a PIL Image."""
    if use_processed and tile.get("processed_bytes"):
        return _bytes_to_pil(tile["processed_bytes"])
    return _bytes_to_pil(tile["original_bytes"])


# ---------------------------------------------------------------------------
# Image upload handler
# ---------------------------------------------------------------------------


def _on_image_upload(
    image: Optional[Image.Image],
    resolution_choice: str,
    tile_size: int,
    overlap: int,
) -> Tuple[List[Dict], int, Optional[Image.Image], str, str, str]:
    """
    Handle image upload: upscale to target resolution, calculate tiles, build HTML grid.

    Returns:
        (tiles_state, selected_idx, original_img, full_image_b64, grid_html, status)
    """
    _empty_html = "<p style='color:#888; padding:16px;'>Upload an image to see the tile grid.</p>"
    if image is None:
        return [], -1, None, "", _empty_html, "No image uploaded."

    target_w, target_h = _compute_target_size(image.size, resolution_choice)
    target_w = min(target_w, MAX_RESOLUTION)
    target_h = min(target_h, MAX_RESOLUTION)

    upscaled = bicubic_upscale(image, target_w, target_h)
    tile_pairs = extract_all_tiles(upscaled, tile_size=tile_size, overlap=overlap)

    tiles_state = [_build_tile_state(ti, img) for ti, img in tile_pairs]

    full_image_b64 = _encode_grid_image(upscaled, max_side=2048, quality=75)
    display_w, display_h = _compute_display_dims(upscaled.width, upscaled.height, max_side=2048)
    scaled_tiles = _scale_tiles_for_display(tiles_state, upscaled.width, upscaled.height, display_w, display_h)
    grid_html = _build_tile_grid_html(scaled_tiles, full_image_b64, -1, display_w, display_h, overlap)

    status = (
        f"✅ Image upscaled to {target_w}×{target_h}. "
        f"Grid: {len(tiles_state)} tiles ({tile_size}px, overlap {overlap}px). "
        f"Click a tile to select it."
    )
    return tiles_state, -1, upscaled, full_image_b64, grid_html, status


# ---------------------------------------------------------------------------
# Caption workflow
# ---------------------------------------------------------------------------


def _caption_all_tiles(
    tiles_state: List[Dict],
    system_prompt: Optional[str],
) -> Tuple[List[Dict], str]:
    """Send all tiles to CaptionService and store captions as prompt overrides."""
    if not tiles_state:
        return tiles_state, "⚠️ No tiles to caption."

    try:
        from src.gpu.caption import CaptionService

        tiles_payload = [
            {"tile_id": t["tile_id"], "image_bytes": t["original_bytes"]}
            for t in tiles_state
        ]
        captions: Dict[str, str] = CaptionService().caption_tiles.remote(
            tiles_payload,
            system_prompt=system_prompt or None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Caption error")
        return tiles_state, f"❌ Caption error: {exc}"

    for tile in tiles_state:
        caption = captions.get(tile["tile_id"], "")
        if caption:
            tile["prompt"] = caption

    return tiles_state, f"✅ Captioned {len(captions)} tile(s)."


def _caption_selected_tile(
    selected_idx: int,
    tiles_state: List[Dict],
    system_prompt: Optional[str],
) -> Tuple[List[Dict], str]:
    """Send only the currently selected tile to CaptionService."""
    if selected_idx < 0 or selected_idx >= len(tiles_state):
        return tiles_state, "⚠️ No tile selected. Click a tile in the grid first."

    tile = tiles_state[selected_idx]
    try:
        from src.gpu.caption import CaptionService

        captions: Dict[str, str] = CaptionService().caption_tiles.remote(
            [{"tile_id": tile["tile_id"], "image_bytes": tile["original_bytes"]}],
            system_prompt=system_prompt or None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Caption error")
        return tiles_state, f"❌ Caption error: {exc}"

    caption = captions.get(tile["tile_id"], "")
    if caption:
        tiles_state[selected_idx]["prompt"] = caption
        return tiles_state, caption

    return tiles_state, ""


# ---------------------------------------------------------------------------
# Upscale workflow
# ---------------------------------------------------------------------------


def _upscale_tiles_batch(
    tiles_state: List[Dict],
    indices: Optional[List[int]],
    original_img: Optional[Image.Image],
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
    lora_states: List[Dict],
    ip_adapter_enabled: bool = False,
    ip_adapter_image: Optional[Image.Image] = None,
    ip_adapter_scale: float = 0.6,
    gen_res: int = 0,
) -> Tuple[List[Dict], Optional[Image.Image], str]:
    """
    Process a batch of tiles through UpscaleService.

    Args:
        tiles_state:         current tile state list.
        indices:             tile indices to process (None = all).
        original_img:        the full upscaled image (used for assembly).
        global_prompt:       positive prompt applied to all tiles.
        negative_prompt:     negative prompt applied to all tiles.
        tile_size:           grid tile size in pixels.
        overlap:             overlap in pixels.
        strength:            denoising strength (0–1).
        steps:               diffusion steps.
        cfg_scale:           CFG scale.
        seed:                seed (-1 = random).
        controlnet_enabled:  whether to use ControlNet Tile.
        conditioning_scale:  ControlNet conditioning scale.
        lora_states:         list of dicts with ``"name"``, ``"enabled"``, ``"weight"``.
        ip_adapter_enabled:  whether to use IP-Adapter.
        ip_adapter_image:    style reference PIL Image (optional).
        ip_adapter_scale:    IP-Adapter influence scale.
        gen_res:             generation resolution per tile (0 = same as tile_size).

    Returns:
        (updated_tiles_state, assembled_result_image, status_text)
    """
    if not tiles_state or original_img is None:
        return tiles_state, None, "⚠️ No tiles to process."

    # Inject trigger words for active LoRAs
    effective_prompt = _inject_trigger_words(global_prompt, lora_states)

    # Resolve effective generation resolution
    effective_gen_res = gen_res if gen_res and gen_res > tile_size else tile_size

    target_indices = indices if indices is not None else list(range(len(tiles_state)))

    # Encode IP-Adapter style image to bytes if provided
    ip_adapter_bytes: Optional[bytes] = None
    if ip_adapter_enabled and ip_adapter_image is not None:
        style_img = ip_adapter_image.convert("RGB").resize((224, 224))
        ip_adapter_bytes = _pil_to_bytes(style_img)

    # Build tiles payload
    tiles_payload = []
    for idx in target_indices:
        tile = tiles_state[idx]
        tile_prompt = tile.get("prompt", "") or ""

        # Optionally upscale tile to generation resolution
        tile_img = _tile_pil(tile)
        if effective_gen_res != tile_size:
            tile_img = tile_img.resize((effective_gen_res, effective_gen_res), Image.LANCZOS)

        # Combine global + tile-specific prompt
        if tile_prompt:
            combined_prompt = f"{effective_prompt}, {tile_prompt}" if effective_prompt else tile_prompt
        else:
            combined_prompt = None

        tiles_payload.append({
            "tile_id": tile["tile_id"],
            "image_bytes": _pil_to_bytes(tile_img),
            "prompt_override": combined_prompt,
        })

    # Build model_config and gen_params for UpscaleService
    active_loras = [
        {"name": s["name"], "weight": s.get("weight", 1.0)}
        for s in lora_states
        if s.get("enabled", True)
    ]
    model_config = {
        "loras": active_loras,
        "controlnet": {
            "enabled": controlnet_enabled,
            "conditioning_scale": conditioning_scale,
        },
    }
    gen_params = {
        "steps": steps,
        "cfg_scale": cfg_scale,
        "denoising_strength": strength,
        "seed": seed if seed >= 0 else None,
    }

    try:
        from src.gpu.upscale import UpscaleService

        results: List[Dict] = UpscaleService().upscale_tiles.remote(
            tiles=tiles_payload,
            model_config=model_config,
            gen_params=gen_params,
            global_prompt=effective_prompt,
            negative_prompt=negative_prompt,
            ip_adapter_enabled=ip_adapter_enabled and ip_adapter_bytes is not None,
            ip_adapter_image_bytes=ip_adapter_bytes,
            ip_adapter_scale=ip_adapter_scale,
            target_width=effective_gen_res,
            target_height=effective_gen_res,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Upscale error")
        return tiles_state, None, f"❌ Upscale error: {exc}"

    # Store results back into tiles_state
    result_map = {r["tile_id"]: r["image_bytes"] for r in results}
    for tile in tiles_state:
        if tile["tile_id"] in result_map:
            proc_bytes = result_map[tile["tile_id"]]
            # Downscale back to grid tile size if generation was at higher resolution
            if effective_gen_res != tile_size:
                proc_img = _bytes_to_pil(proc_bytes)
                proc_img = proc_img.resize((tile_size, tile_size), Image.LANCZOS)
                proc_bytes = _pil_to_bytes(proc_img)
            tile["processed_bytes"] = proc_bytes
            # Encode a small JPEG for grid display
            tile["processed_b64_display"] = _encode_grid_image(
                _bytes_to_pil(proc_bytes), max_side=512, quality=70
            )

    # Assemble result image
    assembled_pairs = []
    for tile in tiles_state:
        tile_info: TileInfo = tile["info"]
        img = _tile_pil(tile, use_processed=bool(tile.get("processed_bytes")))
        assembled_pairs.append((tile_info, img))

    result_img = blend_tiles(assembled_pairs, original_img.size, tile_size=tile_size, overlap=overlap)
    processed_count = sum(1 for t in tiles_state if t.get("processed_bytes"))
    gen_res_label = f" (gen @ {effective_gen_res}px)" if effective_gen_res != tile_size else ""
    status = (
        f"✅ Processed {len(results)} tile(s){gen_res_label}. "
        f"Total processed: {processed_count}/{len(tiles_state)}."
    )
    return tiles_state, result_img, status


# ---------------------------------------------------------------------------
# Gradio app builder
# ---------------------------------------------------------------------------


def create_gradio_app() -> gr.Blocks:
    """
    Create and return the Gradio Blocks interface.

    This function is called by the Modal entrypoint to obtain the ASGI app.
    It does NOT use ``@modal.asgi_app()`` directly — that decorator is applied
    in the main ``modal_app.py`` entrypoint.

    On first visit the setup wizard is shown.  Once all models are downloaded
    (or if they were already present) the wizard is hidden and the main tool
    UI is revealed.

    Returns:
        A configured :class:`gradio.Blocks` instance.
    """
    from src.ui.setup_wizard import create_setup_wizard  # noqa: PLC0415

    with gr.Blocks(title="AI Assets Toolbox") as demo:
        gr.Markdown("# 🎨 AI Assets Toolbox")
        gr.Markdown(
            "Tile-based AI upscaling powered by Illustrious-XL + ControlNet Tile on Modal GPU."
        )

        # ------------------------------------------------------------------
        # Main tool UI — initially hidden until setup is complete
        # ------------------------------------------------------------------
        with gr.Group(visible=False) as tool_group:
            with gr.Tabs():
                _build_upscale_tab()
                _build_model_manager_tab()

        # ------------------------------------------------------------------
        # Setup wizard — shown on first visit; hidden once models are ready.
        # create_setup_wizard() returns a 6-tuple; we unpack all parts so we
        # can wire the Start button and the BrowserState token restore here,
        # where both wizard_group and tool_group are in scope.
        # ------------------------------------------------------------------
        (
            wizard_group,
            _check_models_downloaded,
            start_btn,
            restore_tokens_fn,
            restore_token_inputs,
            restore_token_outputs,
        ) = create_setup_wizard()

        # Wire the Start button: hide wizard, show tool
        def _on_setup_complete():
            """Hide the wizard and reveal the main tool."""
            return gr.Group(visible=False), gr.Group(visible=True)

        start_btn.click(
            fn=_on_setup_complete,
            outputs=[wizard_group, tool_group],
        )

        # ------------------------------------------------------------------
        # Page-load handler — two responsibilities:
        #   1. Restore saved API tokens from BrowserState into the textboxes.
        #   2. Skip the wizard entirely if all models are already downloaded.
        #
        # Gradio 6.0 allows multiple demo.load() registrations; each fires
        # independently on page load.
        # ------------------------------------------------------------------

        # 1. Restore tokens
        demo.load(
            fn=restore_tokens_fn,
            inputs=restore_token_inputs,
            outputs=restore_token_outputs,
        )

        # 2. Check model status and show/hide wizard accordingly
        def _on_page_load():
            """
            Called on every page load.  If all models are already present on
            the volume, skip straight to the main tool UI.
            """
            try:
                from src.services.download import DownloadService  # noqa: PLC0415

                status = DownloadService().check_status.remote()
                all_ready = all(v["downloaded"] for v in status.values())
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Could not reach DownloadService on page load — showing wizard."
                )
                all_ready = False

            if all_ready:
                # Models present: hide wizard, show tool
                return gr.Group(visible=False), gr.Group(visible=True)
            # Models missing: show wizard, hide tool
            return gr.Group(visible=True), gr.Group(visible=False)

        demo.load(
            fn=_on_page_load,
            outputs=[wizard_group, tool_group],
        )

        # 3. Run metadata migration (seeds .metadata.json from legacy hardcoded
        #    LoRAs and moves flat files into the loras/ subdirectory).
        def _on_migrate():
            try:
                from src.services.download import DownloadService  # noqa: PLC0415
                result = DownloadService().migrate_metadata.remote()
                logger.info(
                    "migrate_metadata: migrated=%d moved=%d",
                    result.get("migrated", 0),
                    result.get("moved", 0),
                )
            except Exception:  # noqa: BLE001
                logger.warning("migrate_metadata failed on page load — non-fatal.")

        demo.load(fn=_on_migrate)

    return demo


# ---------------------------------------------------------------------------
# Upscale tab
# ---------------------------------------------------------------------------


def _build_upscale_tab() -> None:
    """Render the Tile Upscale tab inside a Gradio Blocks context."""

    with gr.Tab("🖼️ Tile Upscale"):

        # ----------------------------------------------------------------
        # State
        # ----------------------------------------------------------------
        tiles_state = gr.State(value=[])
        selected_idx_state = gr.State(value=-1)
        original_img_state = gr.State(value=None)
        full_image_b64_state = gr.State(value="")
        # Dynamic LoRA state: list of model-info dicts from DownloadService
        available_loras_state = gr.State(value=[])
        # Active LoRA selections: list of {"name", "enabled", "weight"} dicts
        lora_selections_state = gr.State(value=[])

        # Hidden textbox for JS → Python tile selection communication.
        # NOTE: visible must be True so Gradio 6.0 renders it in the DOM.
        # The CSS below hides it visually via #tile-selected-idx.
        tile_selected_idx_tb = gr.Textbox(
            value="-1",
            visible=True,
            elem_id="tile-selected-idx",
        )
        gr.HTML(
            value=(
                "<style>"
                "#tile-selected-idx {"
                "  position: absolute !important;"
                "  width: 0 !important;"
                "  height: 0 !important;"
                "  overflow: hidden !important;"
                "  opacity: 0 !important;"
                "  pointer-events: none !important;"
                "  margin: 0 !important;"
                "  padding: 0 !important;"
                "  border: none !important;"
                "}"
                "</style>"
            ),
            visible=True,
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
                        "The shorter side is calculated automatically to preserve the aspect ratio."
                    ),
                )
                regen_grid_btn = gr.Button("🔄 Regenerate Grid", variant="secondary", size="sm")

        status_text = gr.Textbox(
            label="Status",
            interactive=False,
            lines=1,
            placeholder="Upload an image to get started…",
        )

        # ----------------------------------------------------------------
        # MAIN: Two-column layout
        # ----------------------------------------------------------------
        with gr.Row(equal_height=False):

            # ============================================================
            # LEFT COLUMN (~65%): Tile grid + selected tile + actions
            # ============================================================
            with gr.Column(scale=13, min_width=480):

                gr.Markdown("### 🗂️ Tile Grid — click to select")
                tile_grid_html = gr.HTML(
                    value="<p style='color:#888; padding:16px;'>Upload an image to see the tile grid.</p>",
                    sanitize_html=False,
                )

                # ---- Selected tile detail panel ----
                with gr.Group():
                    gr.Markdown("### 📌 Selected Tile")
                    with gr.Row():
                        tile_orig_preview = gr.Image(
                            label="Original + Blend Zones",
                            type="pil",
                            interactive=False,
                            height=200,
                        )
                        tile_proc_preview = gr.Image(
                            label="Processed",
                            type="pil",
                            interactive=False,
                            height=200,
                        )
                    tile_prompt_box = gr.Textbox(
                        label="Tile Prompt (appends to global prompt)",
                        lines=2,
                        placeholder="Leave empty to use the global prompt, or enter a tile-specific prompt…",
                    )

                # ---- Action buttons ----
                gr.Markdown("### 🚀 Actions")
                with gr.Row():
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
                            value=DEFAULT_TILE_SIZE,
                            label="Tile Size (px)",
                            precision=0,
                            scale=1,
                            info="Size of each square tile. 1024 recommended for SDXL.",
                            elem_id="upscale_tile_size",
                        )
                        overlap_num = gr.Number(
                            value=DEFAULT_OVERLAP,
                            label="Overlap (px)",
                            precision=0,
                            scale=1,
                            info="Pixel overlap between adjacent tiles for seamless blending.",
                            elem_id="upscale_overlap",
                        )
                    gen_res_dd = gr.Dropdown(
                        choices=["Same as grid", "1536×1536 ✨ Recommended", "2048×2048 (Max detail)"],
                        value="1536×1536 ✨ Recommended",
                        label="Generation Resolution",
                        info=(
                            "Higher generation resolution gives the model more pixels to add detail. "
                            "1536×1536 is the native Illustrious-XL resolution. "
                            "The tile grid stays at the grid resolution; only individual tiles are "
                            "upscaled for generation then downscaled back."
                        ),
                        elem_id="upscale_gen_res",
                    )

                # ---- 🎨 Generation Settings ----
                with gr.Accordion("🎨 Generation Settings", open=True):
                    global_prompt = gr.Textbox(
                        label="Global Style Prompt",
                        placeholder="masterpiece, best quality, highly detailed, sharp focus",
                        lines=2,
                        info="Applied to all tiles unless a tile-specific prompt is set.",
                        elem_id="upscale_global_prompt",
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="blurry, low quality, artifacts, watermark, jpeg artifacts",
                        lines=2,
                        info="Concepts to suppress. Applied to all tiles.",
                        elem_id="upscale_negative_prompt",
                    )
                    with gr.Row():
                        strength_sl = gr.Slider(
                            0.0, 1.0, value=0.35, step=0.05,
                            label="Denoise Strength",
                            info="How much to change the tile. 0.3–0.5 recommended.",
                            scale=1,
                            elem_id="upscale_strength",
                        )
                        steps_sl = gr.Slider(
                            1, 100, value=30, step=1,
                            label="Steps",
                            info="Diffusion steps. 20–40 is a good range.",
                            scale=1,
                            elem_id="upscale_steps",
                        )
                    with gr.Row():
                        cfg_sl = gr.Slider(
                            1.0, 20.0, value=7.0, step=0.5,
                            label="CFG Scale",
                            info="Prompt adherence. 6–8 recommended.",
                            scale=1,
                            elem_id="upscale_cfg_scale",
                        )
                        seed_num = gr.Number(
                            value=-1,
                            label="Seed",
                            precision=0,
                            scale=1,
                            info="-1 = random seed each run.",
                        )

                    # LoRA Controls (dynamic — loaded from DownloadService)
                        gr.Markdown("### LoRA Controls")
                        with gr.Row():
                            lora_refresh_btn = gr.Button(
                                "🔄 Refresh LoRA List",
                                variant="secondary",
                                scale=1,
                                size="sm",
                            )
                        lora_controls_html = gr.HTML(
                            value=(
                                "<p style='color:#888;font-style:italic;font-size:0.9em;'>"
                                "Loading LoRA list… click 🔄 Refresh LoRA List if it doesn't appear."
                                "</p>"
                            ),
                            sanitize_html=False,
                            elem_id="upscale-lora-controls",
                        )
                        # Hidden textbox: JS checkboxes/sliders write JSON here
                        lora_selection_tb = gr.Textbox(
                            value="[]",
                            visible=True,
                            elem_id="upscale-lora-selection",
                        )
                        gr.HTML(
                            value=(
                                "<style>"
                                "#upscale-lora-selection {"
                                "  position: absolute !important;"
                                "  width: 0 !important;"
                                "  height: 0 !important;"
                                "  overflow: hidden !important;"
                                "  opacity: 0 !important;"
                                "  pointer-events: none !important;"
                                "  margin: 0 !important;"
                                "  padding: 0 !important;"
                                "  border: none !important;"
                                "}"
                                "</style>"
                            ),
                            visible=True,
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
                        "blended back onto the Pass 1 result using a feathered gradient mask.\n\n"
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
                        info="Denoise strength for the offset pass. 0.3–0.4 recommended.",
                    )
                    seam_fix_feather_sl = gr.Slider(
                        8, 64, value=32, step=8,
                        label="Feather / Blend Size (px)",
                        info="Width of the gradient blend zone at tile edges.",
                    )

        # ================================================================
        # Event handlers
        # ================================================================

        # ----------------------------------------------------------------
        # Image upload → calculate tiles → build HTML tile grid
        # ----------------------------------------------------------------
        def on_upload(img, res, ts, ov):
            ts, ov = int(ts), int(ov)
            tiles, sel, orig, b64, grid_html, status = _on_image_upload(img, res, ts, ov)
            return tiles, sel, orig, b64, grid_html, status

        _upload_outputs = [
            tiles_state, selected_idx_state, original_img_state,
            full_image_b64_state, tile_grid_html, status_text,
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
            try:
                idx = int(idx_str)
            except (ValueError, TypeError):
                idx = -1

            if not tiles or idx < 0 or idx >= len(tiles):
                grid_html = _rebuild_grid_html(tiles, full_b64, -1, orig_img)
                return -1, None, None, "", grid_html

            tile = tiles[idx]
            prompt = tile.get("prompt", "")

            orig_tile_pil = _tile_pil(tile)
            if orig_img is not None:
                img_w, img_h = orig_img.size
                tile_info: TileInfo = tile["info"]
                overlap_px = max(1, int(overlap))
                orig_tile = generate_mask_blur_preview(
                    orig_tile_pil, tile_info, overlap_px, img_w, img_h
                )
            else:
                orig_tile = orig_tile_pil

            proc_tile = _tile_pil(tile, use_processed=True) if tile.get("processed_bytes") else None
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
        def on_prompt_edit(new_prompt: str, selected_idx: int, tiles: List[Dict]):
            if selected_idx < 0 or selected_idx >= len(tiles):
                return tiles
            tiles[selected_idx]["prompt"] = new_prompt
            return tiles

        tile_prompt_box.change(
            fn=on_prompt_edit,
            inputs=[tile_prompt_box, selected_idx_state, tiles_state],
            outputs=[tiles_state],
        )

        # ----------------------------------------------------------------
        # LoRA list helpers
        # ----------------------------------------------------------------

        def _build_lora_controls_html(loras: List[Dict], selections: List[Dict]) -> str:
            """
            Render interactive LoRA controls as HTML.

            Each LoRA gets a checkbox (enabled) and a weight display.
            Selections are stored in *lora_selections_state* via the hidden
            textbox using JS.
            """
            import json as _json

            if not loras:
                return (
                    "<p style='color:#888;font-style:italic;font-size:0.9em;'>"
                    "No LoRAs installed. Download from CivitAI or upload a file.</p>"
                )

            # Build a lookup for current selections
            sel_map: Dict[str, Dict] = {s["name"]: s for s in selections}

            rows: List[str] = []
            rows.append(
                "<style>"
                ".lora-row{display:flex;align-items:center;gap:10px;padding:6px 0;"
                "border-bottom:1px solid #333;}"
                ".lora-row:last-child{border-bottom:none;}"
                ".lora-name{flex:1;font-size:0.95em;color:#ddd;}"
                ".lora-triggers{font-size:0.78em;color:#888;margin-top:2px;}"
                ".lora-weight-label{font-size:0.82em;color:#aaa;min-width:60px;text-align:right;}"
                "</style>"
            )
            rows.append('<div style="border:1px solid #444;border-radius:8px;padding:8px 12px;">')

            for lora in loras:
                name = lora.get("name") or lora.get("filename", "Unknown")
                triggers = lora.get("trigger_words") or []
                trigger_str = ", ".join(triggers) if triggers else "(no triggers)"
                default_weight = lora.get("default_weight", 1.0)
                sel = sel_map.get(name, {})
                enabled = sel.get("enabled", True)
                weight = sel.get("weight", default_weight)
                checked = "checked" if enabled else ""
                # JS: update the hidden textbox with the full selections JSON
                js_update = (
                    "function updateLoraSelections(){"
                    "  var rows = document.querySelectorAll('.lora-row-item');"
                    "  var sels = [];"
                    "  rows.forEach(function(row){"
                    "    var cb = row.querySelector('input[type=checkbox]');"
                    "    var sl = row.querySelector('input[type=range]');"
                    "    var lbl = row.querySelector('.lora-weight-label');"
                    "    if(cb && sl){"
                    "      sels.push({name:cb.dataset.name,enabled:cb.checked,"
                    "                 weight:parseFloat(sl.value)});"
                    "      if(lbl) lbl.textContent = parseFloat(sl.value).toFixed(1);"
                    "    }"
                    "  });"
                    "  var tb = document.getElementById('upscale-lora-selection');"
                    "  if(tb){"
                    "    var ta = tb.querySelector('textarea');"
                    "    if(ta){ta.value=JSON.stringify(sels);"
                    "      ta.dispatchEvent(new Event('input'));}"
                    "  }"
                    "}"
                )
                rows.append(
                    f'<div class="lora-row lora-row-item">'
                    f'  <input type="checkbox" data-name="{name}" {checked}'
                    f'    onchange="({js_update})();">'
                    f'  <div class="lora-name">'
                    f'    <div>{name}</div>'
                    f'    <div class="lora-triggers">Triggers: {trigger_str}</div>'
                    f'  </div>'
                    f'  <input type="range" min="0" max="2" step="0.1" value="{weight:.1f}"'
                    f'    style="width:100px;" oninput="({js_update})();">'
                    f'  <span class="lora-weight-label">{weight:.1f}</span>'
                    f'</div>'
                )
            rows.append("</div>")
            return "".join(rows)

        def _parse_lora_selections(
            selection_json: str,
            available_loras: List[Dict],
        ) -> List[Dict]:
            """
            Parse the JSON from the hidden lora_selection_tb textbox.

            Falls back to all-enabled with default weights if parsing fails.
            """
            import json as _json

            try:
                parsed = _json.loads(selection_json or "[]")
                if isinstance(parsed, list) and parsed:
                    return parsed
            except (ValueError, TypeError):
                pass
            # Default: all enabled with default weights
            return [
                {
                    "name": m.get("name") or m.get("filename", ""),
                    "enabled": True,
                    "weight": m.get("default_weight", 1.0),
                }
                for m in available_loras
            ]

        # ----------------------------------------------------------------
        # LoRA Refresh handler
        # ----------------------------------------------------------------
        def on_lora_refresh():
            try:
                from src.services.download import DownloadService  # noqa: PLC0415
                loras: List[Dict] = DownloadService().list_user_models.remote(model_type="lora")
            except Exception as exc:  # noqa: BLE001
                logger.exception("list_user_models (lora) error")
                loras = []
            default_sels = [
                {
                    "name": m.get("name") or m.get("filename", ""),
                    "enabled": True,
                    "weight": m.get("default_weight", 1.0),
                }
                for m in loras
            ]
            html = _build_lora_controls_html(loras, default_sels)
            return html, loras, default_sels

        lora_refresh_btn.click(
            fn=on_lora_refresh,
            inputs=[],
            outputs=[lora_controls_html, available_loras_state, lora_selections_state],
        )

        # ----------------------------------------------------------------
        # Hidden textbox → update lora_selections_state
        # ----------------------------------------------------------------
        def on_lora_selection_change(selection_json: str, available_loras: List[Dict]):
            return _parse_lora_selections(selection_json, available_loras)

        lora_selection_tb.change(
            fn=on_lora_selection_change,
            inputs=[lora_selection_tb, available_loras_state],
            outputs=[lora_selections_state],
        )

        # ----------------------------------------------------------------
        # Upscale All Tiles
        # ----------------------------------------------------------------
        _GEN_RES_MAP = {
            "Same as grid": 0,
            "1536×1536 ✨ Recommended": 1536,
            "2048×2048 (Max detail)": 2048,
        }

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
            lora_states,
        ):
            tile_size = int(ts)
            overlap   = int(ov)
            gen_res_val = _GEN_RES_MAP.get(gen_res_label, 0)

            # Pass 1 — standard tile grid
            updated, result, msg = _upscale_tiles_batch(
                tiles, None, orig_img,
                g_prompt, neg_prompt,
                tile_size, overlap,
                strength, int(steps), cfg, int(seed),
                cn_enabled, cond_scale,
                lora_states=lora_states,
                ip_adapter_enabled=ip_enabled,
                ip_adapter_image=ip_img,
                ip_adapter_scale=ip_scale,
                gen_res=gen_res_val,
            )

            if result is None:
                grid_html = _rebuild_grid_html(updated, full_b64, -1, orig_img)
                return updated, grid_html, result, msg

            # Pass 2 — offset grid seam fix (optional)
            if seam_fix_enabled:
                try:
                    offset_tile_infos = calculate_offset_tiles(
                        result.size, tile_size=tile_size, overlap=overlap
                    )
                    if offset_tile_infos:
                        effective_gen_res = gen_res_val if gen_res_val and gen_res_val > tile_size else tile_size

                        # Extract offset tiles from the Pass-1 assembled result
                        offset_tile_imgs = [extract_tile(result, ti) for ti in offset_tile_infos]

                        # Build payload for the offset pass
                        offset_payload = []
                        for ti, tile_img in zip(offset_tile_infos, offset_tile_imgs):
                            if effective_gen_res != tile_size:
                                tile_img = tile_img.resize(
                                    (effective_gen_res, effective_gen_res), Image.LANCZOS
                                )
                            offset_payload.append({
                                "tile_id": f"seam_{ti.row}_{ti.col}",
                                "image_bytes": _pil_to_bytes(tile_img),
                                "prompt_override": None,
                            })

                        active_loras = [
                            {"name": s["name"], "weight": s.get("weight", 1.0)}
                            for s in lora_states
                            if s.get("enabled", True)
                        ]
                        seam_model_config = {
                            "loras": active_loras,
                            "controlnet": {"enabled": cn_enabled, "conditioning_scale": cond_scale},
                        }
                        seam_gen_params = {
                            "steps": int(steps),
                            "cfg_scale": cfg,
                            "denoising_strength": float(seam_fix_strength),
                            "seed": int(seed) if int(seed) >= 0 else None,
                        }

                        from src.gpu.upscale import UpscaleService

                        seam_results: List[Dict] = UpscaleService().upscale_tiles.remote(
                            tiles=offset_payload,
                            model_config=seam_model_config,
                            gen_params=seam_gen_params,
                            global_prompt=g_prompt,
                            negative_prompt=neg_prompt,
                            ip_adapter_enabled=False,
                            target_width=effective_gen_res,
                            target_height=effective_gen_res,
                        )

                        result_map = {r["tile_id"]: r["image_bytes"] for r in seam_results}
                        offset_processed: List[Image.Image] = []
                        for ti in offset_tile_infos:
                            tid = f"seam_{ti.row}_{ti.col}"
                            if tid in result_map:
                                proc_img = _bytes_to_pil(result_map[tid])
                                if effective_gen_res != tile_size:
                                    proc_img = proc_img.resize((tile_size, tile_size), Image.LANCZOS)
                                offset_processed.append(proc_img)
                            else:
                                offset_processed.append(extract_tile(result, ti))

                        result = blend_offset_pass(
                            result,
                            offset_tile_infos,
                            offset_processed,
                            feather_size=int(seam_fix_feather),
                        )
                        msg = msg.rstrip(".") + f" | Pass 2/2: Seam fix applied ({len(offset_tile_infos)} offset tiles)."
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Seam fix error")
                    msg = msg.rstrip(".") + f" | ⚠️ Seam fix error: {exc}"

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
                lora_selections_state,
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
            lora_states,
        ):
            if selected_idx < 0:
                grid_html = _rebuild_grid_html(tiles, full_b64, selected_idx, orig_img)
                return tiles, grid_html, None, None, "⚠️ No tile selected. Click a tile in the grid first."

            gen_res_val = _GEN_RES_MAP.get(gen_res_label, 0)

            updated, result, msg = _upscale_tiles_batch(
                tiles, [selected_idx], orig_img,
                g_prompt, neg_prompt,
                int(ts), int(ov),
                strength, int(steps), cfg, int(seed),
                cn_enabled, cond_scale,
                lora_states=lora_states,
                ip_adapter_enabled=ip_enabled,
                ip_adapter_image=ip_img,
                ip_adapter_scale=ip_scale,
                gen_res=gen_res_val,
            )
            grid_html = _rebuild_grid_html(updated, full_b64, selected_idx, orig_img)
            tile = updated[selected_idx]
            proc_tile = _tile_pil(tile, use_processed=True) if tile.get("processed_bytes") else None
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
                lora_selections_state,
            ],
            outputs=[tiles_state, tile_grid_html, tile_proc_preview, result_image, status_text],
        )

        # ----------------------------------------------------------------
        # Auto-caption all tiles
        # ----------------------------------------------------------------
        def on_caption_all(tiles, sys_prompt: str, full_b64: str, orig_img):
            updated, msg = _caption_all_tiles(tiles, sys_prompt or None)
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
            updated, caption_or_msg = _caption_selected_tile(selected_idx, tiles, sys_prompt or None)
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
        copy_to_input_btn.click(
            fn=lambda img: img,
            inputs=[result_image],
            outputs=[image_input],
        )


# ---------------------------------------------------------------------------
# Model Manager tab
# ---------------------------------------------------------------------------

# CSS injected once for the model manager delete buttons
_MODEL_MANAGER_CSS = """
<style>
.mm-model-card {
  border: 1px solid #444;
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 8px;
  background: #1a1a1a;
}
.mm-model-card:hover { border-color: #666; }
.mm-model-name { font-weight: bold; font-size: 1.05em; color: #e0e0e0; }
.mm-model-meta { font-size: 0.85em; color: #999; margin-top: 4px; }
.mm-model-triggers { font-size: 0.82em; color: #aaa; margin-top: 2px; }
.mm-delete-btn {
  float: right;
  background: #c0392b;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 4px 10px;
  cursor: pointer;
  font-size: 0.85em;
}
.mm-delete-btn:hover { background: #e74c3c; }
.mm-empty { color: #888; font-style: italic; padding: 16px; text-align: center; }
</style>
"""


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes <= 0:
        return "? MB"
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.1f} GB"
    return f"{size_bytes / 1_000_000:.0f} MB"


def _build_model_list_html(models: List[Dict], filter_type: str = "All") -> str:
    """
    Render the installed-models list as an HTML string.

    Each model card has a delete button that uses JS to write the model
    filename into a hidden textbox (same pattern as tile selection).
    """
    filtered = [
        m for m in models
        if filter_type == "All"
        or (filter_type == "LoRAs" and m.get("model_type", "lora") == "lora")
        or (filter_type == "Checkpoints" and m.get("model_type") == "checkpoint")
    ]

    if not filtered:
        return (
            _MODEL_MANAGER_CSS
            + '<div class="mm-empty">No models found. Download from CivitAI or upload a file.</div>'
        )

    cards: List[str] = [_MODEL_MANAGER_CSS]
    for m in filtered:
        name = m.get("name") or m.get("filename", "Unknown")
        filename = m.get("filename", "")
        model_type_raw = m.get("model_type", "lora")
        mtype = model_type_raw.capitalize()
        base = m.get("base_model") or "—"
        size_str = _format_size(m.get("size_bytes", 0))
        triggers = m.get("trigger_words") or []
        trigger_str = ", ".join(triggers) if triggers else "(none)"
        # JS onclick writes filename + model_type into the hidden textbox
        # so the Python delete handler can pick it up.
        delete_target_val = f"{filename}|{model_type_raw}"
        delete_js = (
            f"document.getElementById('mm-delete-target').querySelector('textarea').value"
            f" = '{delete_target_val}'; "
            f"document.getElementById('mm-delete-target').querySelector('textarea').dispatchEvent(new Event('input'));"
        )
        cards.append(
            f'<div class="mm-model-card">'
            f'  <button class="mm-delete-btn" onclick="{delete_js}">🗑️ Delete</button>'
            f'  <div class="mm-model-name">{name}</div>'
            f'  <div class="mm-model-meta">{mtype} · {base} · {size_str}</div>'
            f'  <div class="mm-model-triggers">Triggers: {trigger_str}</div>'
            f'</div>'
        )
    return "".join(cards)


def _build_preview_html(info: Dict) -> str:
    """Render a CivitAI model preview card as HTML."""
    name = info.get("name", "Unknown")
    version = info.get("version_name", "")
    mtype = info.get("model_type", "lora").capitalize()
    base = info.get("base_model") or "—"
    triggers = info.get("trigger_words") or []
    trigger_str = ", ".join(triggers) if triggers else "(none)"
    size_str = _format_size(info.get("size_bytes", 0))
    weight = info.get("recommended_weight")
    weight_str = f"{weight:.2f}" if weight is not None else "1.00"
    version_span = (
        f'  <span style="color:#aaa;font-size:0.85em;">{version}</span>'
        if version else ""
    )
    return (
        '<div style="border:1px solid #555;border-radius:8px;padding:14px 18px;background:#1e2a1e;">'
        f'<div style="font-size:1.1em;font-weight:bold;color:#7ec87e;margin-bottom:6px;">'
        f'{name}{version_span}'
        f'</div>'
        f'<div style="font-size:0.9em;color:#ccc;margin-bottom:4px;">'
        f'  <b>Type:</b> {mtype} &nbsp;|&nbsp; <b>Base:</b> {base}'
        f'</div>'
        f'<div style="font-size:0.9em;color:#ccc;margin-bottom:4px;">'
        f'  <b>Trigger words:</b> {trigger_str}'
        f'</div>'
        f'<div style="font-size:0.9em;color:#ccc;margin-bottom:4px;">'
        f'  <b>Size:</b> {size_str} &nbsp;|&nbsp; <b>Recommended weight:</b> {weight_str}'
        f'</div>'
        f'</div>'
    )


def _build_model_manager_tab() -> None:
    """Render the Model Manager tab inside a Gradio Blocks context."""

    with gr.Tab("🎛️ Model Manager"):
        gr.Markdown("## 🎛️ Model Manager")
        gr.Markdown(
            "Download LoRAs and checkpoints from CivitAI, manage installed models, "
            "and upload local files.  Metadata (trigger words, weights) is stored "
            "automatically."
        )

        # ----------------------------------------------------------------
        # Section A: CivitAI Download
        # ----------------------------------------------------------------
        with gr.Accordion("📥 Download from CivitAI", open=True):
            # Re-use the same BrowserState key as the setup wizard so the
            # token is shared between both UIs.
            civitai_token_state = gr.BrowserState("", storage_key="civitai_token")

            with gr.Row():
                civitai_url_input = gr.Textbox(
                    label="CivitAI URL or Model ID",
                    placeholder="https://civitai.com/models/929497  or  929497",
                    scale=4,
                )
                preview_btn = gr.Button("🔍 Preview", variant="secondary", scale=1)

            civitai_token_input = gr.Textbox(
                label="CivitAI API Token (optional — required for NSFW / early-access models)",
                placeholder="Paste your CivitAI API token here",
                type="password",
            )

            preview_html = gr.HTML(
                value="",
                label="Model Preview",
                sanitize_html=False,
            )

            download_civitai_btn = gr.Button(
                "⬇️ Download",
                variant="primary",
                visible=False,
            )
            download_status = gr.Textbox(
                label="Download Status",
                interactive=False,
                lines=1,
            )

            # State: stores the last-previewed model info dict so the
            # Download button can use it without re-fetching.
            _preview_info_state: gr.State = gr.State(value=None)

            # ------------------------------------------------------------------
            # Restore token from BrowserState on page load
            # ------------------------------------------------------------------
            # (wired via demo.load in create_gradio_app)

            # ------------------------------------------------------------------
            # Preview handler
            # ------------------------------------------------------------------
            def on_preview(url_or_id: str, token: str):
                if not url_or_id.strip():
                    return (
                        '<p style="color:#f88;">⚠️ Please enter a CivitAI URL or model ID.</p>',
                        gr.update(visible=False),
                        None,
                    )
                try:
                    from src.services.download import DownloadService  # noqa: PLC0415
                    result = DownloadService().fetch_civitai_info.remote(
                        url_or_id.strip(), token.strip() or None
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("fetch_civitai_info error")
                    return (
                        f'<p style="color:#f88;">❌ Error: {exc}</p>',
                        gr.update(visible=False),
                        None,
                    )
                if not result.get("success"):
                    err = result.get("error", "Unknown error")
                    return (
                        f'<p style="color:#f88;">❌ {err}</p>',
                        gr.update(visible=False),
                        None,
                    )
                info = result["model_info"]
                return (
                    _build_preview_html(info),
                    gr.update(visible=True),
                    info,
                )

            preview_btn.click(
                fn=on_preview,
                inputs=[civitai_url_input, civitai_token_input],
                outputs=[preview_html, download_civitai_btn, _preview_info_state],
            )

            # ------------------------------------------------------------------
            # Download handler
            # ------------------------------------------------------------------
            def on_download_civitai(url_or_id: str, token: str):
                if not url_or_id.strip():
                    return "⚠️ No URL/ID provided."
                try:
                    from src.services.download import DownloadService  # noqa: PLC0415
                    result = DownloadService().download_civitai_model.remote(
                        url_or_id.strip(), token.strip() or None
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("download_civitai_model error")
                    return f"❌ Download error: {exc}"
                if not result.get("success"):
                    return f"❌ {result.get('error', 'Unknown error')}"
                info = result.get("model_info") or {}
                name = info.get("name", "model")
                already = result.get("already_existed", False)
                if already:
                    return f"ℹ️ '{name}' was already downloaded — metadata updated."
                return f"✅ Downloaded: {name}"

            download_civitai_btn.click(
                fn=on_download_civitai,
                inputs=[civitai_url_input, civitai_token_input],
                outputs=[download_status],
            )

        # ----------------------------------------------------------------
        # Section B: Installed Models
        # ----------------------------------------------------------------
        with gr.Accordion("📦 Installed Models", open=True):
            with gr.Row():
                filter_radio = gr.Radio(
                    choices=["All", "LoRAs", "Checkpoints"],
                    value="All",
                    label="Filter",
                    scale=3,
                )
                mm_refresh_btn = gr.Button("🔄 Refresh", variant="secondary", scale=1)

            # Hidden textbox: JS delete buttons write "filename|model_type" here.
            # elem_id must match the JS in _build_model_list_html.
            mm_delete_target_tb = gr.Textbox(
                value="",
                visible=True,
                elem_id="mm-delete-target",
            )
            gr.HTML(
                value=(
                    "<style>"
                    "#mm-delete-target {"
                    "  position: absolute !important;"
                    "  width: 0 !important;"
                    "  height: 0 !important;"
                    "  overflow: hidden !important;"
                    "  opacity: 0 !important;"
                    "  pointer-events: none !important;"
                    "  margin: 0 !important;"
                    "  padding: 0 !important;"
                    "  border: none !important;"
                    "}"
                    "</style>"
                ),
                visible=True,
            )

            model_list_html = gr.HTML(
                value="<p style='color:#888;font-style:italic;'>Click 🔄 Refresh to load installed models.</p>",
                sanitize_html=False,
            )
            mm_status_text = gr.Textbox(label="Status", interactive=False, lines=1)

            # Internal state: full list of models (for filter re-render)
            _all_models_state: gr.State = gr.State(value=[])

            # ------------------------------------------------------------------
            # Refresh handler
            # ------------------------------------------------------------------
            def on_mm_refresh(filter_val: str):
                try:
                    from src.services.download import DownloadService  # noqa: PLC0415
                    models: List[Dict] = DownloadService().list_user_models.remote()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("list_user_models error")
                    return (
                        f'<p style="color:#f88;">❌ Error: {exc}</p>',
                        f"❌ Error: {exc}",
                        [],
                    )
                html = _build_model_list_html(models, filter_val)
                return html, f"✅ Found {len(models)} model(s).", models

            mm_refresh_btn.click(
                fn=on_mm_refresh,
                inputs=[filter_radio],
                outputs=[model_list_html, mm_status_text, _all_models_state],
            )

            # ------------------------------------------------------------------
            # Filter change handler (re-renders from cached state)
            # ------------------------------------------------------------------
            def on_filter_change(filter_val: str, all_models: List[Dict]):
                return _build_model_list_html(all_models, filter_val)

            filter_radio.change(
                fn=on_filter_change,
                inputs=[filter_radio, _all_models_state],
                outputs=[model_list_html],
            )

            # ------------------------------------------------------------------
            # Delete handler (triggered by JS onclick → hidden textbox)
            # ------------------------------------------------------------------
            def on_mm_delete(delete_target: str, filter_val: str):
                if not delete_target.strip():
                    return gr.update(), "⚠️ No model selected for deletion.", gr.update()
                parts = delete_target.strip().split("|", 1)
                filename = parts[0]
                model_type = parts[1] if len(parts) > 1 else "lora"
                try:
                    from src.services.download import DownloadService  # noqa: PLC0415
                    result = DownloadService().delete_user_model.remote(filename, model_type)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("delete_user_model error")
                    return gr.update(), f"❌ Delete error: {exc}", gr.update()
                if not result.get("success"):
                    return gr.update(), f"❌ {result.get('error', 'Unknown error')}", gr.update()
                # Refresh the list after deletion
                try:
                    from src.services.download import DownloadService  # noqa: PLC0415
                    models: List[Dict] = DownloadService().list_user_models.remote()
                except Exception:  # noqa: BLE001
                    models = []
                html = _build_model_list_html(models, filter_val)
                return html, f"✅ Deleted: {filename}", models

            mm_delete_target_tb.change(
                fn=on_mm_delete,
                inputs=[mm_delete_target_tb, filter_radio],
                outputs=[model_list_html, mm_status_text, _all_models_state],
            )

        # ----------------------------------------------------------------
        # Section C: Local Upload
        # ----------------------------------------------------------------
        with gr.Accordion("📤 Upload Local File", open=False):
            gr.Markdown(
                "_Supported formats: `.safetensors`, `.pt`, `.bin`. "
                "Files must be SDXL-compatible._"
            )
            with gr.Row():
                upload_file = gr.File(
                    label="Choose file (.safetensors, .pt, .bin)",
                    file_types=[".safetensors", ".pt", ".bin"],
                    scale=3,
                )
                upload_name_input = gr.Textbox(
                    label="Display Name",
                    placeholder="My Custom LoRA",
                    scale=2,
                )
            with gr.Row():
                upload_type_dd = gr.Dropdown(
                    choices=["lora", "checkpoint"],
                    value="lora",
                    label="Model Type",
                    scale=1,
                )
                upload_btn = gr.Button("⬆️ Upload", variant="secondary", scale=1)

            upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=1)

            # ------------------------------------------------------------------
            # Upload handler
            # ------------------------------------------------------------------
            def on_upload(file_obj, display_name: str, model_type: str):
                if file_obj is None:
                    return "⚠️ No file selected."
                try:
                    with open(file_obj.name, "rb") as fh:
                        file_data = fh.read()
                    filename = os.path.basename(file_obj.name)
                    from src.gpu.upscale import UpscaleService  # noqa: PLC0415
                    result: Dict = UpscaleService().upload_model.remote(filename, file_data)
                    name_used = display_name.strip() or filename.rsplit(".", 1)[0]
                    size_mb = result.get("size_mb", "?")
                    return (
                        f"✅ Uploaded '{name_used}' ({size_mb} MB) → "
                        f"{result.get('path', filename)}"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("upload_model error")
                    return f"❌ Upload failed: {exc}"

            upload_btn.click(
                fn=on_upload,
                inputs=[upload_file, upload_name_input, upload_type_dd],
                outputs=[upload_status],
            )
