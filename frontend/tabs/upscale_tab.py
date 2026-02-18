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
    blend_tiles,
    draw_tile_grid,
    upscale_image,
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

RESOLUTION_PRESETS = ["2x", "4x", "Custom W×H"]
MODEL_CHOICES = ["z-image-xl", "pony-v6", "illustrious-xl", "flux-dev"]


def _compute_target_size(
    original_size: Tuple[int, int],
    resolution_choice: str,
    custom_w: int,
    custom_h: int,
) -> Tuple[int, int]:
    w, h = original_size
    if resolution_choice == "2x":
        return w * 2, h * 2
    if resolution_choice == "4x":
        return w * 4, h * 4
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
) -> Tuple[int, str, Optional[Image.Image], Optional[Image.Image], Optional[Image.Image]]:
    """Handle gallery tile selection."""
    idx = evt.index
    if not tiles_state or idx >= len(tiles_state):
        return -1, "", None, None, None

    tile = tiles_state[idx]
    tile_info: TileInfo = tile["info"]
    prompt = tile.get("prompt", "")

    # Rebuild grid with selection highlighted
    tile_infos = [t["info"] for t in tiles_state]
    processed_indices = [i for i, t in enumerate(tiles_state) if t.get("processed_b64")]
    grid_img = draw_tile_grid(original_img, tile_infos, selected_indices=[idx], processed_indices=processed_indices)

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
    model: str,
    lora_name: str,
    lora_weight: float,
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
            "lora_name": lora_name if lora_name and lora_name != "None" else None,
            "lora_weight": lora_weight,
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
                        value="2x",
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
                with gr.Row():
                    lora_dd = gr.Dropdown(
                        choices=["None"],
                        value="None",
                        label="LoRA",
                        scale=3,
                    )
                    lora_weight = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                        label="LoRA Weight",
                        scale=2,
                    )

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
                with gr.Row():
                    upscale_sel_btn = gr.Button("🎯 Upscale Selected Tile", variant="secondary")

            # ---- Right column: preview ----
            with gr.Column(scale=1, min_width=400):
                gr.Markdown("### 🗺️ Tile Grid Preview")
                grid_preview = gr.Image(
                    label="Grid Overlay (click gallery below to select tile)",
                    type="pil",
                    interactive=False,
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
        def on_gallery_select(evt: gr.SelectData, tiles: List[Dict], orig_img):
            idx, prompt, grid_img, orig_tile, proc_tile = _on_tile_select(evt, tiles, orig_img)
            if idx < 0:
                info = "No tile selected."
            else:
                ti: TileInfo = tiles[idx]["info"]
                info = f"Tile #{idx} | ID: {ti.tile_id} | x={ti.x}, y={ti.y}, w={ti.w}, h={ti.h}"
            return idx, prompt, grid_img, info, orig_tile, proc_tile

        tile_gallery.select(
            fn=on_gallery_select,
            inputs=[tiles_state, original_img_state],
            outputs=[selected_idx_state, tile_prompt_box, grid_preview, tile_info_text, tile_orig_preview, tile_proc_preview],
        )

        # ----------------------------------------------------------------
        # Prompt editing
        # ----------------------------------------------------------------
        tile_prompt_box.change(
            fn=_on_prompt_edit,
            inputs=[tile_prompt_box, selected_idx_state, tiles_state],
            outputs=[tiles_state],
        )

        # ----------------------------------------------------------------
        # Auto-caption all tiles
        # ----------------------------------------------------------------
        def on_caption(tiles: List[Dict], sys_prompt: str):
            updated, msg = _caption_all_tiles(tiles, sys_prompt, client)
            # Refresh prompt box for currently selected tile (if any)
            return updated, msg

        caption_btn.click(
            fn=on_caption,
            inputs=[tiles_state, global_prompt],
            outputs=[tiles_state, status_text],
        )

        # ----------------------------------------------------------------
        # Upscale all tiles
        # ----------------------------------------------------------------
        def on_upscale_all(
            tiles, orig_img,
            model, lora, lora_w,
            g_prompt, neg_prompt,
            ts, ov,
            strength, steps, cfg, seed,
            cn_enabled, cond_scale,
        ):
            updated, result, msg = _upscale_tiles_batch(
                tiles, None, orig_img,
                model, lora, lora_w,
                g_prompt, neg_prompt,
                int(ts), int(ov),
                strength, int(steps), cfg, int(seed),
                cn_enabled, cond_scale,
                client,
            )
            return updated, result, msg

        upscale_all_btn.click(
            fn=on_upscale_all,
            inputs=[
                tiles_state, original_img_state,
                model_dd, lora_dd, lora_weight,
                global_prompt, negative_prompt,
                tile_size_num, overlap_num,
                strength_sl, steps_sl, cfg_sl, seed_num,
                controlnet_cb, cond_scale_sl,
            ],
            outputs=[tiles_state, result_image, status_text],
        )

        # ----------------------------------------------------------------
        # Upscale selected tile only
        # ----------------------------------------------------------------
        def on_upscale_selected(
            tiles, sel_idx, orig_img,
            model, lora, lora_w,
            g_prompt, neg_prompt,
            ts, ov,
            strength, steps, cfg, seed,
            cn_enabled, cond_scale,
        ):
            if sel_idx < 0:
                return tiles, None, "⚠️ No tile selected."
            updated, result, msg = _upscale_tiles_batch(
                tiles, [sel_idx], orig_img,
                model, lora, lora_w,
                g_prompt, neg_prompt,
                int(ts), int(ov),
                strength, int(steps), cfg, int(seed),
                cn_enabled, cond_scale,
                client,
            )
            # Update processed tile preview
            proc_tile = None
            if updated and sel_idx < len(updated) and updated[sel_idx].get("processed_b64"):
                proc_tile = _b64_to_pil(updated[sel_idx]["processed_b64"])
            return updated, result, msg, proc_tile

        upscale_sel_btn.click(
            fn=on_upscale_selected,
            inputs=[
                tiles_state, selected_idx_state, original_img_state,
                model_dd, lora_dd, lora_weight,
                global_prompt, negative_prompt,
                tile_size_num, overlap_num,
                strength_sl, steps_sl, cfg_sl, seed_num,
                controlnet_cb, cond_scale_sl,
            ],
            outputs=[tiles_state, result_image, status_text, tile_proc_preview],
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
            outputs=[lora_dd],
        )
