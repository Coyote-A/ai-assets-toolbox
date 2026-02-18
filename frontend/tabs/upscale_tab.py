"""
Tile Upscale tab — the main upscaling workflow for the AI Assets Toolbox.

Layout
------
Left column  : controls (image upload, model settings, generation params)
Right column : preview (tile grid overlay, tile gallery, per-tile prompt)
Bottom       : progress / status, final result image

State
-----
tiles_state        : list[dict]  — per-tile data (info, prompt, processed image)
selected_idx_state : int         — currently selected tile index (-1 = none)
original_img_state : PIL.Image   — the uploaded (and upscaled) image
result_img_state   : PIL.Image   — the current assembled result
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
    calculate_tiles,
    extract_tile,
    extract_all_tiles,
    extract_region_image,
    blend_tiles,
    draw_tile_grid,
    upscale_image,
    draw_region_overlay,
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
MODEL_CHOICES = ["z-image-xl", "pony-v6", "illustrious-xl", "flux-dev"]

# Resolution preset values
RESOLUTION_VALUES = {
    "8K × 8K (8192×8192)": (8192, 8192),
    "8K × 4K (8192×4096)": (8192, 4096),
    "4K × 4K (4096×4096)": (4096, 4096),
    "4K × 2K (4096×2048)": (4096, 2048),
}


def _compute_target_size(
    original_size: Tuple[int, int],
    resolution_choice: str,
    custom_w: int,
    custom_h: int,
) -> Tuple[int, int]:
    w, h = original_size
    if resolution_choice in RESOLUTION_VALUES:
        return RESOLUTION_VALUES[resolution_choice]
    # Custom
    return int(custom_w), int(custom_h)


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


def _on_image_upload(
    image: Optional[Image.Image],
    resolution_choice: str,
    custom_w: int,
    custom_h: int,
    tile_size: int,
    overlap: int,
) -> Tuple[
    List[Dict],   # tiles_state
    int,          # selected_idx_state (-1)
    Optional[Image.Image],  # original_img_state
    Optional[Image.Image],  # grid_preview
    List[Image.Image],      # tile_gallery
    str,          # status
]:
    """Handle image upload: upscale to target resolution, calculate tiles, show grid."""
    if image is None:
        return [], -1, None, None, [], "No image uploaded."

    target_w, target_h = _compute_target_size(image.size, resolution_choice, custom_w, custom_h)
    target_w = min(target_w, config.MAX_RESOLUTION)
    target_h = min(target_h, config.MAX_RESOLUTION)

    upscaled = upscale_image(image, target_w, target_h)
    tile_pairs = extract_all_tiles(upscaled, tile_size=tile_size, overlap=overlap)

    tiles_state = [_build_tile_state(ti, img) for ti, img in tile_pairs]
    tile_infos = [t["info"] for t in tiles_state]

    grid_preview = draw_tile_grid(upscaled, tile_infos)
    gallery_images = [_b64_to_pil(t["original_b64"]) for t in tiles_state]

    status = (
        f"✅ Image upscaled to {target_w}×{target_h}. "
        f"Grid: {len(tiles_state)} tiles ({tile_size}px, overlap {overlap}px)."
    )
    return tiles_state, -1, upscaled, grid_preview, gallery_images, status


def _on_tile_select(
    evt: gr.SelectData,
    tiles_state: List[Dict],
    original_img: Optional[Image.Image],
    regions: List[Dict],
    pending_region: Optional[Dict],
    pending_padding: int,
) -> Tuple[int, str, Optional[Image.Image], Optional[Image.Image], Optional[Image.Image]]:
    """Handle gallery tile selection."""
    idx = evt.index
    if not tiles_state or idx >= len(tiles_state):
        return -1, "", None, None, None

    tile = tiles_state[idx]
    tile_info: TileInfo = tile["info"]
    prompt = tile.get("prompt", "")

    # Rebuild grid with selection highlighted
    grid_img = _build_preview(
        original_img,
        tiles_state,
        idx,
        regions,
        pending_region,
        int(pending_padding),
    )

    original_tile = _b64_to_pil(tile["original_b64"])
    processed_tile = _b64_to_pil(tile["processed_b64"]) if tile.get("processed_b64") else None

    return idx, prompt, grid_img, original_tile, processed_tile


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


def _build_preview(
    original_img: Optional[Image.Image],
    tiles_state: List[Dict],
    selected_idx: int,
    regions: List[Dict],
    pending_region: Optional[Dict],
    pending_padding: int,
) -> Optional[Image.Image]:
    """Build the preview image with tile grid and region overlays."""
    if original_img is None:
        return None

    tile_infos = [t["info"] for t in tiles_state] if tiles_state else []
    processed_indices = [i for i, t in enumerate(tiles_state) if t.get("processed_b64")] if tiles_state else []
    selected_indices = [selected_idx] if selected_idx is not None and selected_idx >= 0 else None

    if tile_infos:
        preview = draw_tile_grid(
            original_img,
            tile_infos,
            selected_indices=selected_indices,
            processed_indices=processed_indices,
        )
    else:
        preview = original_img.copy()

    overlay_regions = list(regions) if regions else []
    if pending_region:
        pending = dict(pending_region)
        pending["padding"] = pending_padding
        overlay_regions.append(pending)

    if overlay_regions:
        preview = draw_region_overlay(preview, overlay_regions)

    return preview


def _parse_region_bbox(evt: gr.SelectData) -> Optional[Tuple[int, int, int, int]]:
    """Extract a bbox (x, y, w, h) from a Gradio select event."""
    data = getattr(evt, "value", None)
    if isinstance(data, dict):
        if all(k in data for k in ("x", "y", "w", "h")):
            return int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"])
        if all(k in data for k in ("x", "y", "width", "height")):
            return int(data["x"]), int(data["y"]), int(data["width"]), int(data["height"])

    idx = getattr(evt, "index", None)
    if isinstance(idx, (list, tuple)):
        if len(idx) == 4:
            x1, y1, x2, y2 = [int(v) for v in idx]
            return x1, y1, max(0, x2 - x1), max(0, y2 - y1)

    return None


def _apply_region_results(
    original_img: Image.Image,
    results: List[Dict[str, Any]],
) -> Image.Image:
    """Paste processed regions back into a copy of the original image."""
    composed = original_img.convert("RGB").copy()

    for result in results:
        image_b64 = result.get("image_b64")
        if not image_b64:
            continue

        orig_bbox = result.get("original_bbox") or {}
        padded_bbox = result.get("padded_bbox") or {}
        ox = int(orig_bbox.get("x", 0))
        oy = int(orig_bbox.get("y", 0))
        ow = int(orig_bbox.get("w", 0))
        oh = int(orig_bbox.get("h", 0))
        px = int(padded_bbox.get("x", ox))
        py = int(padded_bbox.get("y", oy))

        if ow <= 0 or oh <= 0:
            continue

        region_img = _b64_to_pil(image_b64)

        # Crop out the padding from the processed image so we paste only the
        # original-region pixels at the original origin.
        pad_offset_x = ox - px
        pad_offset_y = oy - py
        region_img = region_img.crop(
            (pad_offset_x, pad_offset_y, pad_offset_x + ow, pad_offset_y + oh)
        )
        if region_img.size != (ow, oh):
            region_img = region_img.resize((ow, oh), Image.BICUBIC)

        composed.paste(region_img, (ox, oy))

    return composed


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


def _caption_regions(
    regions: List[Dict],
    original_img: Optional[Image.Image],
    system_prompt: Optional[str],
    client: RunPodClient,
) -> Tuple[List[Dict], str]:
    """Send all regions to RunPod for captioning and store results in state."""
    if not regions:
        return regions, "⚠️ No regions to caption."
    if original_img is None:
        return regions, "⚠️ No image uploaded."

    # Extract region images from the original image
    regions_b64 = []
    for i, region in enumerate(regions):
        x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
        padding = region.get("padding", 64)
        
        # Extract with padding
        padded_x = max(0, x - padding)
        padded_y = max(0, y - padding)
        padded_w = w + 2 * padding
        padded_h = h + 2 * padding
        
        # Clamp to image bounds
        padded_x = min(padded_x, original_img.width - 1)
        padded_y = min(padded_y, original_img.height - 1)
        padded_w = min(padded_w, original_img.width - padded_x)
        padded_h = min(padded_h, original_img.height - padded_y)
        
        region_img = original_img.crop((padded_x, padded_y, padded_x + padded_w, padded_y + padded_h))
        regions_b64.append({
            "region_id": f"region_{i}",
            "image_b64": _pil_to_b64(region_img),
        })

    try:
        captions = client.caption_regions(regions_b64, system_prompt=system_prompt or None)
    except RunPodError as exc:
        return regions, f"❌ Caption error: {exc}"

    caption_map = {c["region_id"]: c["caption"] for c in captions}
    for i, region in enumerate(regions):
        region_id = f"region_{i}"
        if region_id in caption_map:
            region["prompt"] = caption_map[region_id]

    return regions, f"✅ Captioned {len(captions)} region(s)."


def _upscale_tiles_batch(
    tiles_state: List[Dict],
    indices: Optional[List[int]],
    original_img: Optional[Image.Image],
    # generation params
    model: str,
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
) -> Tuple[List[Dict], Optional[Image.Image], str]:
    """
    Process a batch of tiles (all or selected) through RunPod upscaling.

    Returns updated tiles_state, assembled result image, and status text.
    """
    if not tiles_state or original_img is None:
        return tiles_state, None, "⚠️ No tiles to process."

    target_indices = indices if indices is not None else list(range(len(tiles_state)))

    model_type = "flux" if model == "flux-dev" else "sdxl"

    tiles_payload = []
    for idx in target_indices:
        tile = tiles_state[idx]
        tile_prompt = tile.get("prompt", "") or ""
        tiles_payload.append({
            "tile_id": tile["tile_id"],
            "image_b64": tile["original_b64"],
            "prompt_override": tile_prompt if tile_prompt else None,
            "model": model,
            "model_type": model_type,
            "loras": loras if loras else [],
            "global_prompt": global_prompt,
            "negative_prompt": negative_prompt,
            "controlnet_enabled": controlnet_enabled,
            "conditioning_scale": conditioning_scale,
            "strength": strength,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
        })

    try:
        results = client.upscale_tiles(tiles_payload)
    except RunPodError as exc:
        return tiles_state, None, f"❌ Upscale error: {exc}"

    result_map = {r["tile_id"]: r["image_b64"] for r in results}
    for tile in tiles_state:
        if tile["tile_id"] in result_map:
            tile["processed_b64"] = result_map[tile["tile_id"]]

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
    status = f"✅ Processed {len(results)} tile(s). Total processed: {processed_count}/{len(tiles_state)}."
    return tiles_state, result_img, status


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def create_upscale_tab(client: RunPodClient) -> None:
    """Render the Tile Upscale tab inside a Gradio Blocks context."""

    with gr.Tab("🖼️ Tile Upscale"):
        gr.Markdown("## Tile-Based AI Upscaling")
        gr.Markdown(
            "Upload an image, configure the model and generation settings, "
            "then caption and upscale tiles individually or all at once."
        )

        # ----------------------------------------------------------------
        # State
        # ----------------------------------------------------------------
        tiles_state = gr.State(value=[])
        selected_idx_state = gr.State(value=-1)
        original_img_state = gr.State(value=None)
        regions_state = gr.State(value=[])
        pending_region_state = gr.State(value=None)
        region_drawing_mode = gr.State(value=False)
        # Multi-LoRA state: list of {"name": str, "weight": float}
        loras_state = gr.State(value=[])

        # ----------------------------------------------------------------
        # Main layout: left controls | right preview
        # ----------------------------------------------------------------
        with gr.Row():

            # ---- Left column: controls ----
            with gr.Column(scale=1, min_width=340):
                gr.Markdown("### 📁 Input Image")
                image_input = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=220,
                )

                with gr.Row():
                    resolution_dd = gr.Dropdown(
                        choices=RESOLUTION_PRESETS,
                        value="8K × 8K (8192×8192)",
                        label="Target Resolution",
                        scale=2,
                    )
                    custom_w = gr.Number(value=2048, label="Width (custom)", precision=0, scale=1, visible=False)
                    custom_h = gr.Number(value=2048, label="Height (custom)", precision=0, scale=1, visible=False)

                gr.Markdown("### 🤖 Model Settings")
                model_dd = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value="z-image-xl",
                    label="Base Model",
                )

                # ---- Multi-LoRA UI ----
                gr.Markdown("### 🎨 LoRAs")
                with gr.Row():
                    lora_add_dd = gr.Dropdown(
                        choices=["None"],
                        value="None",
                        label="Select LoRA to Add",
                        scale=3,
                    )
                    lora_add_weight = gr.Slider(
                        minimum=0.0, maximum=2.0, value=0.7, step=0.05,
                        label="Weight",
                        scale=2,
                    )
                    lora_add_btn = gr.Button("➕ Add", scale=1, variant="secondary")

                loras_display = gr.HTML(
                    "<p style='color: #888; font-size: 0.85em;'>No LoRAs added. Select a LoRA above and click ➕ Add.</p>"
                )
                lora_remove_btn = gr.Button("➖ Remove Last LoRA", variant="secondary", size="sm")

                gr.Markdown("### ✍️ Prompts")
                global_prompt = gr.Textbox(
                    label="Global Style Prompt",
                    placeholder="masterpiece, best quality, highly detailed",
                    lines=2,
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="blurry, low quality, artifacts, watermark",
                    lines=2,
                )

                gr.Markdown("### ⚙️ Tile Settings")
                with gr.Row():
                    tile_size_num = gr.Number(value=config.TILE_SIZE, label="Tile Size (px)", precision=0, scale=1)
                    overlap_num = gr.Number(value=config.DEFAULT_OVERLAP, label="Overlap (px)", precision=0, scale=1)

                gr.Markdown("### 🎛️ Generation Settings")
                with gr.Row():
                    strength_sl = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Strength", scale=1)
                    steps_sl = gr.Slider(1, 100, value=30, step=1, label="Steps", scale=1)
                with gr.Row():
                    cfg_sl = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="CFG Scale", scale=1)
                    seed_num = gr.Number(value=-1, label="Seed (-1 = random)", precision=0, scale=1)

                with gr.Row():
                    controlnet_cb = gr.Checkbox(value=True, label="ControlNet (Tile)")
                    cond_scale_sl = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Conditioning Scale")

                gr.Markdown("### 🚀 Actions")
                with gr.Row():
                    caption_btn = gr.Button("🏷️ Auto-Caption All Tiles", variant="secondary")
                    upscale_all_btn = gr.Button("⬆️ Upscale All Tiles", variant="primary")
                    upscale_sel_btn = gr.Button("🎯 Upscale Selected Tile", variant="secondary")

                gr.Markdown("### 🌍 Region Selection")
                with gr.Row():
                    select_region_btn = gr.Button("🔲 Select Region", variant="secondary")
                    cancel_region_btn = gr.Button("✕ Cancel Selection", variant="secondary", visible=False)
                    add_region_btn = gr.Button("➕ Add Region", variant="secondary")
                    clear_regions_btn = gr.Button("🗑️ Clear Regions", variant="secondary")
                    caption_regions_btn = gr.Button("🔎 Auto-Generate Prompts", variant="secondary")

                with gr.Row():
                    region_padding = gr.Number(value=64, label="Region Padding (px)", precision=0, scale=1)
                    region_prompt = gr.Textbox(
                        label="Region Prompt",
                        placeholder="Prompt for this region (optional)",
                        lines=1,
                        scale=2,
                    )
                    region_negative_prompt = gr.Textbox(
                        label="Region Negative Prompt",
                        placeholder="Negative prompt for this region (optional)",
                        lines=1,
                        scale=2,
                    )

                regions_list = gr.HTML(
                    "<p style='color: #888;'>No regions added yet. Click 'Select Region' to draw a region on the image.</p>"
                )

                gr.Markdown("### 🗺️ Tile Grid Preview")
                grid_preview = gr.Image(
                    label="Grid Overlay (click gallery below to select tile)",
                    type="pil",
                    interactive=True,
                    height=320,
                )

                gr.Markdown("### 🖼️ Tile Gallery — click to select")
                tile_gallery = gr.Gallery(
                    label="Tiles",
                    columns=4,
                    height=200,
                    object_fit="contain",
                    allow_preview=False,
                )

                gr.Markdown("### 📝 Selected Tile")
                tile_info_text = gr.Textbox(
                    label="Tile Info",
                    interactive=False,
                    lines=1,
                    placeholder="Select a tile from the gallery above",
                )
                tile_prompt_box = gr.Textbox(
                    label="Tile Prompt (editable)",
                    lines=3,
                    placeholder="Auto-generated or manual prompt for this tile",
                )

                with gr.Row():
                    tile_orig_preview = gr.Image(
                        label="Original Tile",
                        type="pil",
                        interactive=False,
                        height=160,
                    )
                    tile_proc_preview = gr.Image(
                        label="Processed Tile",
                        type="pil",
                        interactive=False,
                        height=160,
                    )

        # ----------------------------------------------------------------
        # Bottom: status + result
        # ----------------------------------------------------------------
        status_text = gr.Textbox(label="Status", interactive=False, lines=1)
        result_image = gr.Image(
            label="Final Upscaled Result",
            type="pil",
            interactive=False,
            height=480,
        )

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
        # Image upload → calculate tiles
        # ----------------------------------------------------------------
        def on_upload(img, res, cw, ch, ts, ov):
            return _on_image_upload(img, res, cw, ch, int(ts), int(ov))

        image_input.upload(
            fn=on_upload,
            inputs=[image_input, resolution_dd, custom_w, custom_h, tile_size_num, overlap_num],
            outputs=[tiles_state, selected_idx_state, original_img_state, grid_preview, tile_gallery, status_text],
        )

        # Also trigger on change (e.g. drag-and-drop)
        image_input.change(
            fn=on_upload,
            inputs=[image_input, resolution_dd, custom_w, custom_h, tile_size_num, overlap_num],
            outputs=[tiles_state, selected_idx_state, original_img_state, grid_preview, tile_gallery, status_text],
        )

        # ----------------------------------------------------------------
        # Gallery tile selection
        # ----------------------------------------------------------------
        def on_gallery_select(evt: gr.SelectData, tiles: List[Dict], orig_img, regions, pending_region, padding):
            idx, prompt, grid_img, orig_tile, proc_tile = _on_tile_select(
                evt,
                tiles,
                orig_img,
                regions,
                pending_region,
                int(padding),
            )
            if idx < 0:
                info = "No tile selected."
            else:
                ti: TileInfo = tiles[idx]["info"]
                info = f"Tile #{idx} | ID: {ti.tile_id} | x={ti.x}, y={ti.y}, w={ti.w}, h={ti.h}"
            return idx, prompt, grid_img, info, orig_tile, proc_tile

        tile_gallery.select(
            fn=on_gallery_select,
            inputs=[tiles_state, original_img_state, regions_state, pending_region_state, region_padding],
            outputs=[selected_idx_state, tile_prompt_box, grid_preview, tile_info_text, tile_orig_preview, tile_proc_preview],
        )

        # ----------------------------------------------------------------
        # Prompt editing
        # ----------------------------------------------------------------
        def _render_regions_html(regions: List[dict]) -> str:
            """Render the regions list as HTML."""
            if not regions:
                return "<p style='color: #888;'>No regions added yet. Click 'Select Region' to draw a region on the image.</p>"

            html = "<div style='max-height: 200px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>"
            for idx, region in enumerate(regions):
                x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
                prompt = region.get("prompt", "")
                html += f"""
                <div style='display: flex; justify-content: space-between; align-items: center; padding: 8px; border-bottom: 1px solid #eee; background: {"#f0f4ff" if idx % 2 == 0 else "#fff"};'>
                    <div>
                        <strong>Region #{idx}</strong>: ({x}, {y}) {w}×{h}<br/>
                        <small style='color: #666;'>Prompt: {prompt[:50] + "..." if len(prompt) > 50 else prompt}</small>
                    </div>
                </div>
                """
            html += "</div>"
            return html

        def _on_region_select(
            evt: gr.SelectData,
            regions: List[dict],
            pending_region: Optional[dict],
            original_img,
            tiles_state: List[Dict],
            selected_idx: int,
            padding: int,
            drawing_mode: bool,
        ):
            """Capture region bbox from image drag and store as pending selection."""
            if original_img is None:
                preview = _build_preview(original_img, tiles_state, selected_idx, regions, pending_region, int(padding))
                return pending_region, regions, _render_regions_html(regions), preview, "⚠️ No image uploaded."

            if not drawing_mode:
                preview = _build_preview(original_img, tiles_state, selected_idx, regions, pending_region, int(padding))
                return pending_region, regions, _render_regions_html(regions), preview, "⚠️ Enable 'Select Region' to draw."

            bbox = _parse_region_bbox(evt)
            if not bbox:
                preview = _build_preview(original_img, tiles_state, selected_idx, regions, pending_region, int(padding))
                return pending_region, regions, _render_regions_html(regions), preview, "⚠️ Drag to select a valid region."

            x, y, w, h = bbox
            x = max(0, min(int(x), original_img.width - 1))
            y = max(0, min(int(y), original_img.height - 1))
            w = max(1, min(int(w), original_img.width - x))
            h = max(1, min(int(h), original_img.height - y))

            pending_region = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }

            preview = _build_preview(original_img, tiles_state, selected_idx, regions, pending_region, int(padding))
            msg = f"✅ Region selected: ({x}, {y}) {w}×{h}. Click 'Add Region' to store."
            return pending_region, regions, _render_regions_html(regions), preview, msg

        grid_preview.select(
            fn=_on_region_select,
            inputs=[
                regions_state,
                pending_region_state,
                original_img_state,
                tiles_state,
                selected_idx_state,
                region_padding,
                region_drawing_mode,
            ],
            outputs=[pending_region_state, regions_state, regions_list, grid_preview, status_text],
        )

        # ----------------------------------------------------------------
        # Populate LoRA dropdown from RunPod on tab load
        # ----------------------------------------------------------------
        def _load_loras():
            try:
                models = client.list_models("lora")
                names = ["None"] + [m.get("name", "") for m in models if m.get("name")]
                return gr.update(choices=names, value="None")
            except Exception:  # noqa: BLE001
                return gr.update(choices=["None"], value="None")

        # Trigger LoRA refresh when the model dropdown changes
        model_dd.change(
            fn=_load_loras,
            inputs=None,
            outputs=[lora_add_dd],
        )

        # ----------------------------------------------------------------
        # Multi-LoRA helpers
        # ----------------------------------------------------------------
        def _render_loras_html(loras: List[Dict]) -> str:
            if not loras:
                return "<p style='color: #888; font-size: 0.85em;'>No LoRAs added. Select a LoRA above and click ➕ Add.</p>"
            rows = ""
            for i, entry in enumerate(loras):
                name = entry.get("name", "")
                weight = entry.get("weight", 1.0)
                bg = "#f5f7ff" if i % 2 == 0 else "#fff"
                rows += (
                    f"<div style='display:flex;align-items:center;gap:8px;"
                    f"padding:4px 6px;border-bottom:1px solid #eee;background:{bg};'>"
                    f"<span style='flex:1;font-size:0.85em;overflow:hidden;text-overflow:ellipsis;"
                    f"white-space:nowrap;' title='{name}'>{name}</span>"
                    f"<span style='font-size:0.8em;color:#555;min-width:40px;text-align:right;'>"
                    f"&times;{weight:.2f}</span>"
                    f"</div>"
                )
            return (
                "<div style='border:1px solid #ddd;border-radius:4px;max-height:160px;overflow-y:auto;'>"
                + rows
                + "</div>"
                + f"<p style='font-size:0.75em;color:#888;margin:2px 0 0;'>{len(loras)} LoRA(s) active</p>"
            )

        def _add_lora_fn(loras: List[Dict], name: str, weight: float) -> Tuple[List[Dict], str]:
            if not name or name == "None":
                return loras, _render_loras_html(loras)
            # Avoid duplicates — update weight if already present
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

        # ----------------------------------------------------------------
        # Region selection handlers
        # ----------------------------------------------------------------
        def _toggle_region_drawing_mode(current_mode: bool):
            """Toggle region drawing mode on/off."""
            new_mode = not current_mode
            return (
                gr.update(variant="primary" if new_mode else "secondary", value="✅ Drawing Region" if new_mode else "🔲 Select Region"),
                gr.update(visible=new_mode),
                gr.update(visible=not new_mode),  # region_padding: gr.Number doesn't support interactive toggle
                gr.update(interactive=not new_mode),
                gr.update(interactive=not new_mode),
                gr.update(interactive=not new_mode),
                new_mode,
            )

        def _on_select_region_click(current_mode: bool):
            """Handle select region button click."""
            return _toggle_region_drawing_mode(current_mode)

        select_region_btn.click(
            fn=_on_select_region_click,
            inputs=[region_drawing_mode],
            outputs=[select_region_btn, cancel_region_btn, region_padding, region_prompt, region_negative_prompt, add_region_btn, region_drawing_mode],
        )

        cancel_region_btn.click(
            fn=lambda: _toggle_region_drawing_mode(True),  # Pass True to toggle to False
            inputs=[],
            outputs=[select_region_btn, cancel_region_btn, region_padding, region_prompt, region_negative_prompt, add_region_btn, region_drawing_mode],
        )

        def _add_region(
            regions: List[dict],
            pending_region: Optional[dict],
            padding: int,
            prompt: str,
            neg_prompt: str,
            original_img,
            tiles: List[Dict],
            selected_idx: int,
        ):
            """Add a new region from the pending selection."""
            if original_img is None:
                preview = _build_preview(original_img, tiles, selected_idx, regions, pending_region, int(padding))
                return regions, pending_region, _render_regions_html(regions), preview, "⚠️ Please upload an image first."

            if not pending_region:
                preview = _build_preview(original_img, tiles, selected_idx, regions, pending_region, int(padding))
                return regions, pending_region, _render_regions_html(regions), preview, "⚠️ Draw a region first."

            new_region = {
                "x": pending_region["x"],
                "y": pending_region["y"],
                "w": pending_region["w"],
                "h": pending_region["h"],
                "padding": padding,
                "prompt": prompt,
                "negative_prompt": neg_prompt,
            }
            regions = list(regions) + [new_region]
            pending_region = None
            preview = _build_preview(original_img, tiles, selected_idx, regions, None, int(padding))
            return regions, pending_region, _render_regions_html(regions), preview, f"✅ Added region at ({new_region['x']}, {new_region['y']}) {new_region['w']}×{new_region['h']}"

        add_region_btn.click(
            fn=_add_region,
            inputs=[regions_state, pending_region_state, region_padding, region_prompt, region_negative_prompt, original_img_state, tiles_state, selected_idx_state],
            outputs=[regions_state, pending_region_state, regions_list, grid_preview, status_text],
        )

        def _clear_regions(regions: List[dict], original_img, tiles: List[Dict], selected_idx: int):
            """Clear all regions."""
            preview = _build_preview(original_img, tiles, selected_idx, [], None, 0)
            return [], None, "<p style='color: #888;'>No regions added yet. Click 'Select Region' to draw a region on the image.</p>", preview, "🗑️ Cleared all regions."

        clear_regions_btn.click(
            fn=_clear_regions,
            inputs=[regions_state, original_img_state, tiles_state, selected_idx_state],
            outputs=[regions_state, pending_region_state, regions_list, grid_preview, status_text],
        )

        # ----------------------------------------------------------------
        # Auto-caption regions handler
        # ----------------------------------------------------------------
        def on_caption_regions(regions: List[dict], orig_img, sys_prompt: str):
            """Handle captioning all regions."""
            updated_regions, msg = _caption_regions(regions, orig_img, sys_prompt, client)
            # Update the regions list HTML with new prompts
            return updated_regions, _render_regions_html(updated_regions), msg

        caption_regions_btn.click(
            fn=on_caption_regions,
            inputs=[regions_state, original_img_state, global_prompt],
            outputs=[regions_state, regions_list, status_text],
        )

        # ----------------------------------------------------------------
        # Upscale regions handler
        # ----------------------------------------------------------------
        def on_upscale_regions(
            regions: List[dict],
            orig_img,
            model: str,
            loras: List[Dict],
            g_prompt: str,
            neg_prompt: str,
            strength: float,
            steps: int,
            cfg: float,
            seed: int,
            cn_enabled: bool,
            cond_scale: float,
        ):
            """Handle upscaling regions."""
            if not regions:
                return None, "⚠️ No regions to upscale. Add regions first."
            if orig_img is None:
                return None, "⚠️ No image uploaded."

            # Convert original image to base64
            orig_b64 = _pil_to_b64(orig_img)

            # Prepare regions data for API
            regions_data = [
                {
                    "x": r["x"],
                    "y": r["y"],
                    "w": r["w"],
                    "h": r["h"],
                    "padding": r["padding"],
                    "prompt": r["prompt"],
                    "negative_prompt": r["negative_prompt"],
                }
                for r in regions
            ]

            model_type = "flux" if model == "flux-dev" else "sdxl"

            try:
                results = client.upscale_regions(
                    regions_data=regions_data,
                    original_image_b64=orig_b64,
                    model=model,
                    model_type=model_type,
                    loras=loras if loras else [],
                    global_prompt=g_prompt,
                    negative_prompt=neg_prompt,
                    strength=strength,
                    steps=int(steps),
                    cfg_scale=cfg,
                    seed=int(seed),
                    controlnet_enabled=cn_enabled,
                    conditioning_scale=cond_scale,
                )
            except RunPodError as exc:
                return None, f"❌ Region upscale error: {exc}"

            result_img = _apply_region_results(orig_img, results)
            return result_img, f"✅ Processed {len(results)} region(s)."

        def on_upscale_all(
            tiles, orig_img,
            model, loras,
            g_prompt, neg_prompt,
            ts, ov,
            strength, steps, cfg, seed,
            cn_enabled, cond_scale,
        ):
            """Handle tile-based upscaling for all tiles."""
            updated, result, msg = _upscale_tiles_batch(
                tiles, None, orig_img,
                model, loras,
                g_prompt, neg_prompt,
                int(ts), int(ov),
                strength, int(steps), cfg, int(seed),
                cn_enabled, cond_scale,
                client,
            )
            return updated, result, msg

        # Add upscale regions button click handler (use upscale_all_btn when regions exist)
        def on_upscale_all_with_regions(
            tiles, regions, orig_img,
            model, loras,
            g_prompt, neg_prompt,
            ts, ov,
            strength, steps, cfg, seed,
            cn_enabled, cond_scale,
        ):
            """Handle upscale all - use regions if available, otherwise use tiles."""
            if regions and len(regions) > 0:
                # Use region-based upscaling
                result_img, msg = on_upscale_regions(
                    regions, orig_img,
                    model, loras,
                    g_prompt, neg_prompt,
                    strength, steps, cfg, seed,
                    cn_enabled, cond_scale,
                )
                # Return tiles state unchanged, region-composited image
                return tiles, result_img, f"🌍 Upscaling {len(regions)} region(s)... " + msg
            else:
                # Use tile-based upscaling
                return on_upscale_all(
                    tiles, orig_img,
                    model, loras,
                    g_prompt, neg_prompt,
                    ts, ov,
                    strength, steps, cfg, seed,
                    cn_enabled, cond_scale,
                )

        upscale_all_btn.click(
            fn=on_upscale_all_with_regions,
            inputs=[
                tiles_state, regions_state, original_img_state,
                model_dd, loras_state,
                global_prompt, negative_prompt,
                tile_size_num, overlap_num,
                strength_sl, steps_sl, cfg_sl, seed_num,
                controlnet_cb, cond_scale_sl,
            ],
            outputs=[tiles_state, result_image, status_text],
        )

        # ----------------------------------------------------------------
        # Upscale selected tile handler
        # ----------------------------------------------------------------
        def on_upscale_selected(
            tiles, selected_idx, orig_img,
            model, loras,
            g_prompt, neg_prompt,
            ts, ov,
            strength, steps, cfg, seed,
            cn_enabled, cond_scale,
        ):
            """Handle tile-based upscaling for the selected tile only."""
            if selected_idx < 0:
                return tiles, None, "⚠️ No tile selected. Click a tile in the gallery first."
            updated, result, msg = _upscale_tiles_batch(
                tiles, [selected_idx], orig_img,
                model, loras,
                g_prompt, neg_prompt,
                int(ts), int(ov),
                strength, int(steps), cfg, int(seed),
                cn_enabled, cond_scale,
                client,
            )
            return updated, result, msg

        upscale_sel_btn.click(
            fn=on_upscale_selected,
            inputs=[
                tiles_state, selected_idx_state, original_img_state,
                model_dd, loras_state,
                global_prompt, negative_prompt,
                tile_size_num, overlap_num,
                strength_sl, steps_sl, cfg_sl, seed_num,
                controlnet_cb, cond_scale_sl,
            ],
            outputs=[tiles_state, result_image, status_text],
        )

        # ----------------------------------------------------------------
        # Tile prompt editing
        # ----------------------------------------------------------------
        tile_prompt_box.change(
            fn=_on_prompt_edit,
            inputs=[tile_prompt_box, selected_idx_state, tiles_state],
            outputs=[tiles_state],
        )

        # ----------------------------------------------------------------
        # Auto-caption all tiles handler
        # ----------------------------------------------------------------
        def on_caption_all(tiles, sys_prompt: str):
            updated, msg = _caption_all_tiles(tiles, sys_prompt, client)
            # Refresh gallery with updated prompts (images unchanged)
            gallery_images = [_b64_to_pil(t["original_b64"]) for t in updated]
            return updated, gallery_images, msg

        caption_btn.click(
            fn=on_caption_all,
            inputs=[tiles_state, global_prompt],
            outputs=[tiles_state, tile_gallery, status_text],
        )
