# AI Assets Toolbox

A toolkit for AI-powered image upscaling and asset generation, with a RunPod serverless backend and Gradio frontend.

---

## Features

- **Tile-based image upscaling (up to 8K)** — Slices large images into 1024×1024 tiles, processes each through a diffusion img2img pipeline, and reassembles them with automatic seam blending using linear gradient feathering.
- **Multiple model support** — SDXL checkpoints (Z-Image, Pony, Illustrious) and Flux.1-dev, selectable per job.
- **Auto-captioning tiles via Qwen2.5-VL-7B** — Each tile is automatically described by a vision-language model to generate accurate per-tile diffusion prompts.
- **Per-tile prompt editing and global style prompts** — Review and edit auto-generated captions in the UI; prepend a global style prompt to all tiles.
- **LoRA support** — Load up to 2 stacked LoRA adapters per job (SDXL or Flux).
- **Tiled ControlNet for upscale quality** — Tile ControlNet preserves structural composition during diffusion, preventing hallucination at higher denoising strengths.
- **Model management** — Upload and delete LoRAs and model checkpoints directly to/from RunPod network storage via the Gradio UI.
- **Spritesheet animation generation** *(coming soon)* — Generate animated spritesheets for game assets.

---

## Architecture

The system consists of two components:

- **Gradio frontend** — Runs locally on the user's machine. Handles tile slicing, seam blending, and all UI interactions.
- **RunPod serverless backend** — Runs on a GPU worker (recommended: NVIDIA A100 80GB). Handles diffusion inference, captioning, and model storage.

Communication is via HTTPS REST API; images are transferred as base64-encoded strings within JSON payloads.

For full details see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Quick Start

### Prerequisites

- Python 3.11+
- [RunPod](https://runpod.io) account with API key and a Network Volume
- Docker (for building the backend image)

### Backend Deployment

1. **Build and push the Docker image:**

   ```bash
   # Linux / macOS
   ./scripts/deploy.sh <your_dockerhub_username>

   # Windows (PowerShell)
   .\scripts\deploy.ps1 -DockerHubUsername <your_dockerhub_username>
   ```

2. **Create a RunPod Serverless Endpoint:**
   - Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
   - Click **New Endpoint**
   - Set the Docker image to `<your_dockerhub_username>/ai-assets-toolbox-backend:latest`
   - Attach your Network Volume (mounted at `/runpod-volume/`)
   - Select GPU: **A100 80GB SXM** (recommended)
   - Save and note the **Endpoint ID**

3. **Download model weights to the network volume** (first-time setup):
   - SSH into a RunPod pod with the volume attached, or use the RunPod file browser
   - Place model files under `/runpod-volume/models/` following the directory layout in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

### Frontend Setup

1. **Install dependencies:**

   ```bash
   pip install -r frontend/requirements.txt
   ```

2. **Configure environment variables:**

   ```bash
   cp .env.example frontend/.env
   # Edit frontend/.env and fill in your RunPod credentials
   ```

3. **Run the app:**

   ```bash
   python frontend/app.py
   ```

   The Gradio UI will be available at `http://localhost:7860`.

---

## Project Structure

```
ai-assets-toolbox/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example                    # Template for environment variables
├── docs/
│   └── ARCHITECTURE.md             # Full architecture document
│
├── backend/                        # RunPod serverless worker
│   ├── Dockerfile                  # Container build for RunPod
│   ├── requirements.txt            # Python dependencies
│   ├── handler.py                  # RunPod handler entry point
│   ├── model_manager.py            # Dynamic model loading/unloading
│   ├── start.sh                    # Container startup script
│   ├── pipelines/
│   │   ├── sdxl_pipeline.py        # SDXL img2img + ControlNet
│   │   ├── flux_pipeline.py        # Flux img2img + ControlNet
│   │   └── qwen_pipeline.py        # Qwen2.5-VL captioning
│   ├── actions/
│   │   ├── upscale.py              # Upscale action handler
│   │   ├── caption.py              # Caption action handler
│   │   └── models.py               # List/upload/delete model actions
│   └── utils/
│       ├── image_utils.py          # Base64 encode/decode, image transforms
│       └── storage.py              # Network volume file operations
│
├── frontend/                       # Gradio application
│   ├── requirements.txt            # Python dependencies
│   ├── app.py                      # Gradio app entry point
│   ├── api_client.py               # RunPod API client wrapper
│   ├── config.py                   # Configuration loader
│   ├── tiling.py                   # Tile slicing, grid calculation, reassembly
│   └── tabs/
│       ├── upscale_tab.py          # Tab 1: Tile-based upscaling UI
│       ├── spritesheet_tab.py      # Tab 2: Spritesheet placeholder
│       └── model_manager_tab.py    # Tab 3: Model management UI
│
└── scripts/
    ├── deploy.sh                   # Deploy backend to RunPod (Linux/macOS)
    └── deploy.ps1                  # Deploy backend to RunPod (Windows)
```

---

## Configuration

Copy `.env.example` to `frontend/.env` and fill in your values:

| Variable | Description |
|---|---|
| `RUNPOD_API_KEY` | Your RunPod API key (from [RunPod Settings](https://www.runpod.io/console/user/settings)) |
| `RUNPOD_ENDPOINT_ID` | The serverless endpoint ID created during backend deployment |

---

## License

MIT — see [`LICENSE`](LICENSE).
