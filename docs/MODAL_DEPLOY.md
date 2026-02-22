# Modal Deployment Guide

## Overview

The AI Assets Toolbox runs entirely on Modal.com as a single unified app. One command deploys all three components:

| Component | Type | GPU | Idle Timeout |
|-----------|------|-----|-------------|
| `web_ui` | Gradio ASGI | CPU | 10 min |
| `CaptionService` | Modal class | T4 | 5 min |
| `UpscaleService` | Modal class | A10G | 5 min |

## Deployment Scripts

### Production deploy — permanent URL
```powershell
.\deploy.ps1
```

Modal prints the Gradio UI URL after deployment, e.g.:
```
✓ Created web function web_ui => https://your-org--ai-toolbox-web-ui.modal.run
```

### Development — hot-reload, temporary URL
```powershell
.\serve.ps1
```

Changes to any file in `src/` are picked up automatically while `modal serve` is running. Press `Ctrl+C` to stop.

## Auto-Setup (no manual prerequisites)

Both scripts call `Ensure-Setup` from `scripts/common.ps1` before launching Modal. On every run this automatically:

1. **Checks Python 3.11+** — installs via `winget` if missing
2. **Checks pip** — bootstraps via `ensurepip` if missing
3. **Installs / upgrades Modal CLI** — `pip install --upgrade modal`
4. **Checks Modal authentication** — opens browser login if not authenticated
5. **Creates `ai-toolbox-secrets`** — prompts for `CIVITAI_API_TOKEN` if missing
6. **Creates `ai-toolbox-loras` volume** — creates the Modal volume if missing

All steps are idempotent — already-done steps are skipped.

## How It Works

`src/app.py` is the single entrypoint. It:

1. Imports `app` from `src/app_config.py` — the shared `modal.App("ai-toolbox")` instance.
2. Imports `CaptionService` and `UpscaleService` from `src/gpu/` — this registers them with the app via their `@app.cls()` decorators.
3. Defines `web_ui()` — a `@modal.asgi_app()` function that returns the Gradio `Blocks` instance.

All three components share the same app, so `modal deploy src/app.py` deploys everything at once.

## Container Images

Images are built once and cached. Rebuilds only happen when `src/app_config.py` changes.

| Image | Contents | Build time |
|-------|----------|-----------|
| `caption_image` | PyTorch + Transformers + Qwen3-VL-2B weights (~4 GB) | ~10 min |
| `upscale_image` | PyTorch + Diffusers + all model weights (~15 GB) | ~20 min |
| `gradio_image` | Gradio + Pillow + NumPy (no GPU libs) | ~2 min |

## LoRA Management

LoRA `.safetensors` files are stored in the `ai-toolbox-loras` Modal volume, mounted at `/vol/loras` inside the upscale container. The Gradio UI provides a Model Manager tab for uploading and managing LoRAs.

To list LoRAs from the CLI:
```bash
modal volume ls ai-toolbox-loras
```

To upload a LoRA manually:
```bash
modal volume put ai-toolbox-loras my-lora.safetensors /loras/my-lora.safetensors
```

## Secrets

The app uses the `ai-toolbox-secrets` Modal secret for:
- `CIVITAI_API_TOKEN` — required for downloading LoRAs from CivitAI

## Logs

```bash
# Stream logs from all functions
modal app logs ai-toolbox

# Logs for a specific function
modal app logs ai-toolbox --function web_ui
```

## Stopping / Deleting

```bash
# Stop the app (containers will drain and shut down)
modal app stop ai-toolbox

# Delete the app entirely
modal app delete ai-toolbox
```
