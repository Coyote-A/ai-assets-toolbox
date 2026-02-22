# AI Assets Toolbox

Tile-based AI image upscaler for game assets, powered by Modal.com.

## Features
- SDXL-based tile upscaling with ControlNet guidance
- AI-powered tile captioning (Qwen3-VL-2B)
- LoRA support (custom + CivitAI)
- IP-Adapter style transfer
- Automatic tile splitting, processing, and merging
- Web UI (Gradio) served on Modal
- **Setup wizard** — first-time model download through the browser UI (no CLI setup needed)

## Architecture
Everything runs on Modal.com:
- **Gradio UI** — CPU container, serves the web interface + setup wizard
- **Caption Service** — T4 GPU, Qwen3-VL-2B-Instruct
- **Upscale Service** — A10G GPU, Illustrious-XL + ControlNet Tile + LoRAs
- **Download Service** — CPU, downloads models from HuggingFace into a Modal Volume

Models are stored in the `ai-toolbox-models` Modal Volume (not baked into Docker images), so deploys are fast and model updates don't require image rebuilds.

## Quick Start

### Prerequisites
- Python 3.11+
- Modal account (https://modal.com)

> **No manual setup needed.** Both scripts below auto-check and configure everything:
> Python, Modal CLI, authentication, and volumes.
> API keys and model downloads are handled through the **setup wizard** in the browser.

### Deploy (production — permanent URL)
```powershell
.\deploy.ps1
```

### Dev mode (hot-reload — temporary URL)
```powershell
.\serve.ps1
```

Both scripts run `Ensure-Setup` from [`scripts/common.ps1`](scripts/common.ps1) before launching Modal, so they are safe to run on a fresh machine.

## Project Structure
```
ai-assets-toolbox/
├── deploy.ps1           # Production deploy (permanent URL)
├── serve.ps1            # Dev server with hot-reload (temporary URL)
├── scripts/
│   └── common.ps1       # Shared auto-setup logic (Python, Modal, auth, volumes)
└── src/
    ├── app.py           # Main entrypoint (modal deploy src/app.py)
    ├── app_config.py    # Shared Modal app, images, volumes
    ├── tiling.py        # Tile splitting/merging utilities
    ├── gpu/
    │   ├── caption.py   # CaptionService (Qwen3-VL-2B on T4)
    │   └── upscale.py   # UpscaleService (SDXL on A10G)
    ├── services/
    │   ├── download.py        # DownloadService (CPU, HuggingFace → Volume)
    │   └── model_registry.py  # Model definitions and Volume paths
    └── ui/
        ├── gradio_app.py  # Gradio web interface
        └── setup_wizard.py # First-time setup wizard (API keys + model downloads)
```

## Cost
| Component | GPU | Cost/hr | Idle Timeout |
|-----------|-----|---------|-------------|
| Caption | T4 | ~$0.59 | 5 min |
| Upscale | A10G | ~$1.10 | 5 min |
| UI | CPU | ~$0.04 | 10 min |

You only pay for actual compute time. Containers shut down automatically after idle timeout.

## License
MIT
