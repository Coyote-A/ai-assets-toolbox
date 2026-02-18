"""
Tile Upscale tab — the main upscaling workflow for the AI Assets Toolbox.

Layout
------
Full-width image upload at the top.
Two-column layout below:
  Left column  (~65%): Unified tile grid gallery + selected tile detail + action buttons
  Right column (~35%): Settings organized in collapsible accordions

State
-----
tiles_state        : list[dict]  — per-tile data (info, prompt, processed image)
selected_idx_state : int         — currently selected tile index (-1 = none)
original_img_state : PIL.Image   — the uploaded (and upscaled) image
grid_cols_state    : int         — number of columns in the tile grid (for gallery layout)
loras_state        : list[dict]  — active LoRAs [{name, weight}, ...]
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
from PIL import Image, ImageDraw

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
# Tile thumbnail generation (annotated gallery images)
# ---------------------------------------------------------------------------

_THUMB_SIZE = 256  # thumbnail size for gallery display


def _make_tile_thumbnail(
    tile_img: Image.Image,
    label: str,
    is_selected: bool = False,
    is_processed: bool = False,
) -> Image.Image:
    """
    Generate an annotated thumbnail for the tile gallery.

    - Processed tiles get a green border/tint overlay.
    - Selected tile gets a blue border.
    - Label (e.g. "0,0" or "0,0 ✓") is drawn in the corner.
    """
    thumb = tile_img.convert("RGB").copy()
    thumb.thumbnail((_THUMB_SIZE, _THUMB_SIZE), Image.LANCZOS)

    # Pad to square so gallery looks uniform
    sq = Image.new("RGB", (_THUMB_SIZE, _THUMB_SIZE), (30, 30, 30))
    offset_x = (_THUMB_SIZE - thumb.width) // 2
    offset_y = (_THUMB_SIZE - thumb.height) // 2
    sq.paste(thumb, (offset_x, offset_y))

    draw = ImageDraw.Draw(sq, "RGBA")

    # Green tint overlay for processed tiles
    if is_processed:
        draw.rectangle([0, 0, _THUMB_SIZE - 1, _THUMB_SIZE - 1], fill=(0, 200, 80, 40))

    # Border colour: blue for selected, green for processed, grey otherwise
    if is_selected:
        border_color = (30, 120, 255, 255)
        border_width = 4
    elif is_processed:
        border_color = (0, 200, 80, 255)
        border_width = 3
    else:
        border_color = (80, 80, 80, 200)
        border_width = 1

    for i in range(border_width):
        draw.rectangle(
            [i, i, _THUMB_SIZE - 1 - i, _THUMB_SIZE - 1 - i],
            outline=border_color,
        )

    # Label badge in top-left corner
    badge_text = label
    badge_w = len(badge_text) * 7 + 8
    badge_h = 18
    draw.rectangle([2, 2, 2 + badge_w, 2 + badge_h], fill=(0, 0, 0, 180))
    draw.text((6, 4), badge_text, fill=(255, 255, 255, 255))

    return sq


def _build_gallery_items(
    tiles_state: List[Dict],
    selected_idx: int,
) -> List[Tuple[Image.Image, str]]:
    """Build the list of (annotated_image, caption) tuples for gr.Gallery."""
    items = []
    for i, tile in enumerate(tiles_state):
        ti: TileInfo = tile["info"]
        is_processed = bool(tile.get("processed_b64"))
        is_selected = (i == selected_idx)
        label = f"{ti.row},{ti.col}"
        if is_processed:
            label += " ✓"
        tile_img = _b64_to_pil(tile["original_b64"])
        thumb = _make_tile_thumbnail(tile_img, label, is_selected, is_processed)
        caption = f"Tile {ti.row},{ti.col}"
        if is_processed:
            caption += " ✓"
        items.append((thumb, caption))
    return items


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
    List[Dict],             # tiles_state
    int,                    # selected_idx_state (-1)
    Optional[Image.Image],  # original_img_state
    int,                    # grid_cols_state
    List[Tuple],            # gallery items
    str,                    # status
]:
    """Handle image upload: upscale to target resolution, calculate tiles, populate gallery."""
    if image is None:
        return [], -1, None, 1, [], "No image uploaded."

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

    gallery_items = _build_gallery_items(tiles_state, -1)

    status = (
        f"✅ Image upscaled to {target_w}×{target_h}. "
        f"Grid: {len(tiles_state)} tiles ({tile_size}px, overlap {overlap}px). "
        f"Click a tile to select it."
    )
    return tiles_state, -1, upscaled, grid_cols, gallery_items, status


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
) -> Tuple[List[Dict], Optional[Image.Image], str]:
    """
    Process a batch of tiles (all or selected) through RunPod upscaling.

    Returns updated tiles_state, assembled result image, and status text.
    """
    if not tiles_state or original_img is None:
        return tiles_state, None, "⚠️ No tiles to process."

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
        entry: Dict[str, Any] = {
            "tile_id": tile["tile_id"],
            "image_b64": tile["original_b64"],
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

                # ---- Unified tile grid gallery ----
                gr.Markdown("### 🗂️ Tile Grid — click to select")
                tile_gallery = gr.Gallery(
                    label="Tile Grid",
                    columns=4,
                    height=380,
                    object_fit="contain",
                    allow_preview=False,
                    elem_classes=["tile-grid-gallery"],
                    show_label=False,
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
        # Image upload → calculate tiles → populate gallery
        # ----------------------------------------------------------------
        def on_upload(img, res, cw, ch, ts, ov):
            return _on_image_upload(img, res, cw, ch, int(ts), int(ov))

        def _update_gallery_cols(grid_cols: int):
            """Return a gallery update with the correct column count."""
            return gr.update(columns=max(1, grid_cols))

        image_input.upload(
            fn=on_upload,
            inputs=[image_input, resolution_dd, custom_w, custom_h, tile_size_num, overlap_num],
            outputs=[tiles_state, selected_idx_state, original_img_state, grid_cols_state, tile_gallery, status_text],
        ).then(
            fn=_update_gallery_cols,
            inputs=[grid_cols_state],
            outputs=[tile_gallery],
        )

        image_input.change(
            fn=on_upload,
            inputs=[image_input, resolution_dd, custom_w, custom_h, tile_size_num, overlap_num],
            outputs=[tiles_state, selected_idx_state, original_img_state, grid_cols_state, tile_gallery, status_text],
        ).then(
            fn=_update_gallery_cols,
            inputs=[grid_cols_state],
            outputs=[tile_gallery],
        )

        # Regenerate grid button (re-runs the same upload logic)
        regen_grid_btn.click(
            fn=on_upload,
            inputs=[image_input, resolution_dd, custom_w, custom_h, tile_size_num, overlap_num],
            outputs=[tiles_state, selected_idx_state, original_img_state, grid_cols_state, tile_gallery, status_text],
        ).then(
            fn=_update_gallery_cols,
            inputs=[grid_cols_state],
            outputs=[tile_gallery],
        )

        # ----------------------------------------------------------------
        # Gallery tile selection → update selected tile detail panel
        # ----------------------------------------------------------------
        def on_gallery_select(
            evt: gr.SelectData,
            tiles: List[Dict],
            current_selected: int,
        ):
            """Handle gallery tile click: update selected index and detail panel."""
            idx = evt.index
            if not tiles or idx >= len(tiles):
                return current_selected, None, None, "", _build_gallery_items(tiles, current_selected)

            tile = tiles[idx]
            ti: TileInfo = tile["info"]
            prompt = tile.get("prompt", "")

            # Show original tile
            orig_tile = _b64_to_pil(tile["original_b64"])
            # Show processed tile if available
            proc_tile = _b64_to_pil(tile["processed_b64"]) if tile.get("processed_b64") else None

            # Rebuild gallery with new selection highlighted
            gallery_items = _build_gallery_items(tiles, idx)

            return idx, orig_tile, proc_tile, prompt, gallery_items

        tile_gallery.select(
            fn=on_gallery_select,
            inputs=[tiles_state, selected_idx_state],
            outputs=[selected_idx_state, tile_orig_preview, tile_proc_preview, tile_prompt_box, tile_gallery],
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
        ):
            """Upscale all tiles and return updated gallery + assembled result.

            When seam_fix_enabled is True, a second pass is run with the tile
            grid shifted by half a stride in both X and Y.  The offset tiles
            are extracted from the Pass-1 result, processed at the lower
            seam_fix_strength, then blended back using a feathered mask.
            """
            tile_size = int(ts)
            overlap   = int(ov)

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
            )

            if result is None:
                # Pass 1 failed — return early
                gallery_items = _build_gallery_items(updated, -1)
                return updated, gallery_items, result, msg

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

                        # Build payload for the offset pass
                        offset_payload = []
                        for i, (ti, tile_img) in enumerate(
                            zip(offset_tile_infos, offset_tile_imgs)
                        ):
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
                                offset_processed.append(_b64_to_pil(result_map[tid]))
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

            # Rebuild gallery to show processed checkmarks
            gallery_items = _build_gallery_items(updated, -1)
            return updated, gallery_items, result, msg

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
            ],
            outputs=[tiles_state, tile_gallery, result_image, status_text],
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
        ):
            """Upscale only the currently selected tile."""
            if selected_idx < 0:
                return tiles, None, None, None, "⚠️ No tile selected. Click a tile in the grid first."
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
            )
            # Refresh gallery and selected tile preview
            gallery_items = _build_gallery_items(updated, selected_idx)
            tile = updated[selected_idx]
            proc_tile = _b64_to_pil(tile["processed_b64"]) if tile.get("processed_b64") else None
            return updated, gallery_items, proc_tile, result, msg

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
            ],
            outputs=[tiles_state, tile_gallery, tile_proc_preview, result_image, status_text],
        )

        # ----------------------------------------------------------------
        # Auto-caption all tiles
        # ----------------------------------------------------------------
        def on_caption_all(tiles, sys_prompt: str):
            updated, msg = _caption_all_tiles(tiles, sys_prompt, client)
            gallery_items = _build_gallery_items(updated, -1)
            return updated, gallery_items, msg

        caption_btn.click(
            fn=on_caption_all,
            inputs=[tiles_state, global_prompt],
            outputs=[tiles_state, tile_gallery, status_text],
        )
