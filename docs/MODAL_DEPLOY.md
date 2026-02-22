# Modal Deployment Guide

## Overview

The AI Assets Toolbox runs entirely on Modal.com as a single unified app. One command deploys all components:

| Component | Type | GPU | Idle Timeout |
|-----------|------|-----|-------------|
| `web_ui` | Gradio ASGI | CPU | 10 min |
| `CaptionService` | Modal class | T4 | 5 min |
| `UpscaleService` | Modal class | A10G | 5 min |
| `DownloadService` | Modal class | CPU | 5 min |

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
5. **Creates Modal volumes** — creates `ai-toolbox-models` and `ai-toolbox-loras` if missing

All steps are idempotent — already-done steps are skipped.

> **No secrets needed at deploy time.** API keys (CivitAI, HuggingFace) are entered through the
> setup wizard in the browser UI and stored in browser `localStorage` via Gradio `BrowserState`.

## First-Time Setup Wizard

On first visit to the deployed URL, the app shows a **Setup Wizard** before the main UI:

1. **API Keys** — Enter your CivitAI and/or HuggingFace tokens (stored in browser only, never sent to a server)
2. **Model Downloads** — Select which models to download; the wizard calls `DownloadService` to fetch them from HuggingFace into the `ai-toolbox-models` Modal Volume
3. **Progress** — Real-time download progress is streamed back to the browser

On subsequent visits the wizard is skipped automatically if models are already present in the Volume.

## How It Works

`src/app.py` is the single entrypoint. It:

1. Imports `app` from `src/app_config.py` — the shared `modal.App("ai-toolbox")` instance.
2. Imports `CaptionService` and `UpscaleService` from `src/gpu/` — registers them via `@app.cls()`.
3. Imports `DownloadService` from `src/services/download` — registers the CPU download worker.
4. Defines `web_ui()` — a `@modal.asgi_app()` function that returns the Gradio `Blocks` instance.

All components share the same app, so `modal deploy src/app.py` deploys everything at once.

## Container Images

Images are built once and cached. Rebuilds only happen when `src/app_config.py` changes.

| Image | Contents | Build time |
|-------|----------|-----------|
| `caption_image` | PyTorch + Transformers (no weights baked in) | ~3 min |
| `upscale_image` | PyTorch + Diffusers (no weights baked in) | ~5 min |
| `gradio_image` | Gradio + Pillow + NumPy (no GPU libs) | ~2 min |

> **Fast deploys.** Model weights are stored in the `ai-toolbox-models` Modal Volume, not baked
> into Docker images. Deploy is now pip-only — no 17 GB model downloads on every image rebuild.

## Volume-Based Model Storage

Models are stored in the `ai-toolbox-models` Modal Volume, mounted at `/vol/models` inside GPU containers.

GPU services include a **readiness guard**: if the required model files are not yet present in the
Volume, the service returns a clear error message prompting the user to run the setup wizard first.

To inspect the models volume from the CLI:
```bash
modal volume ls ai-toolbox-models
```

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
