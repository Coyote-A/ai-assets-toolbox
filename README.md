# AI Assets Toolbox

A toolkit for AI-powered tile-based game asset upscaling. Slices large images into 1024×1024 tiles, processes each through a diffusion img2img pipeline on RunPod serverless GPU workers, and reassembles them with automatic seam blending.

---

## Architecture

The system consists of three components:

- **Upscale Worker** — RunPod serverless worker running SDXL img2img with ControlNet Tile, IP-Adapter, and LoRA support. Handles tile upscaling and model management (list/upload/delete LoRAs).
- **Caption Worker** — RunPod serverless worker running Qwen3-VL-8B. Auto-generates per-tile descriptions for use as diffusion prompts.
- **Gradio Frontend** — Runs locally on the user's machine. Handles tile slicing, seam blending, and all UI interactions. Communicates with both workers via separate RunPod endpoints.

Communication is via HTTPS REST API; images are transferred as base64-encoded strings within JSON payloads.

For full details see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Features

- **Tile-based image upscaling (up to 8K)** — Slices large images into 1024×1024 tiles, processes each through a diffusion img2img pipeline, and reassembles them with automatic seam blending using linear gradient feathering.
- **Illustrious-XL model** — Pre-cached in the Docker image via `wget` for fast cold starts; no manual model download required.
- **Auto-captioning tiles via Qwen3-VL-8B** — Each tile is automatically described by a vision-language model to generate accurate per-tile diffusion prompts. Runs on a dedicated caption worker.
- **Per-tile prompt editing and global style prompts** — Review and edit auto-generated captions in the UI; prepend a global style prompt to all tiles.
- **LoRA support** — Load multiple stacked LoRA adapters per job (SDXL-compatible).
- **Tiled ControlNet for upscale quality** — Tile ControlNet preserves structural composition during diffusion, preventing hallucination at higher denoising strengths.
- **IP-Adapter style transfer** — Upload a style reference image to guide all tile generations for consistent visual style across the output.
- **Seam Fix (Grid Offset Pass)** — Optional second upscale pass with the tile grid shifted by half a stride, blended back with feathered masks to eliminate visible tile seams.
- **Custom HTML/JS tile grid UI** — Interactive tile grid rendered with custom HTML and JavaScript; selected tile highlighted with a blue border; processed tiles show a green indicator.
- **LoRA Manager** — Upload and delete LoRA adapters directly to/from RunPod network storage via the Gradio UI.
- **Spritesheet animation generation** *(coming soon)* — Generate animated spritesheets using Qwen-Image-Edit on a dedicated RunPod public endpoint.

---

## Project Structure

```
ai-assets-toolbox/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example                        # Template for environment variables
├── docs/
│   ├── ARCHITECTURE.md                 # Full architecture document
│   └── REDESIGN_PLAN.md                # Design decisions and implementation notes
│
├── workers/
│   ├── upscale/                        # Upscale RunPod serverless worker
│   │   ├── Dockerfile                  # Container build — pre-caches Illustrious-XL + aux models
│   │   ├── requirements.txt            # Python dependencies
│   │   ├── handler.py                  # RunPod handler entry point
│   │   ├── model_manager.py            # Dynamic model loading/unloading + IP-Adapter support
│   │   ├── start.sh                    # Container startup script
│   │   ├── pipelines/
│   │   │   └── sdxl_pipeline.py        # SDXL img2img + ControlNet + IP-Adapter
│   │   ├── actions/
│   │   │   ├── upscale.py              # Tile upscale action handler
│   │   │   ├── upscale_regions.py      # Region upscale action handler
│   │   │   └── models.py               # List/upload/delete LoRA actions
│   │   └── utils/
│   │       ├── image_utils.py          # Base64 encode/decode, image transforms
│   │       └── storage.py              # Network volume file operations
│   │
│   └── caption/                        # Caption RunPod serverless worker
│       ├── Dockerfile                  # Container build — pre-caches Qwen3-VL-8B
│       ├── requirements.txt            # Python dependencies
│       ├── handler.py                  # RunPod handler entry point
│       ├── qwen_pipeline.py            # Qwen3-VL-8B captioning pipeline
│       └── start.sh                    # Container startup script
│
├── frontend/                           # Gradio application
│   ├── requirements.txt                # Python dependencies
│   ├── app.py                          # Gradio app entry point + CSS theme
│   ├── api_client.py                   # RunPod API client (multi-endpoint)
│   ├── config.py                       # Configuration loader
│   ├── tiling.py                       # Tile slicing, grid calculation, seam blending, offset pass
│   └── tabs/
│       ├── upscale_tab.py              # Tab 1: Tile-based upscaling UI (main workflow)
│       ├── spritesheet_tab.py          # Tab 2: Spritesheet animation (coming soon)
│       └── model_manager_tab.py        # Tab 3: LoRA Manager
│
└── scripts/
    ├── deploy.sh                       # Deploy workers to RunPod (Linux/macOS)
    └── deploy.ps1                      # Deploy workers to RunPod (Windows)
```

---

## Setup

### Prerequisites

- Python 3.11+
- [RunPod](https://runpod.io) account with API key and a Network Volume (for the upscale worker)
- Docker (for building worker images)

### Environment Variables

Copy `.env.example` to `frontend/.env` and fill in your values:

| Variable | Description |
|---|---|
| `RUNPOD_API_KEY` | Your RunPod API key (from [RunPod Settings](https://www.runpod.io/console/user/settings)) |
| `RUNPOD_UPSCALE_ENDPOINT_ID` | Endpoint ID for the upscale worker |
| `RUNPOD_CAPTION_ENDPOINT_ID` | Endpoint ID for the caption worker |

### Frontend

```bash
pip install -r frontend/requirements.txt
cd frontend && python app.py
```

The Gradio UI will be available at `http://localhost:7860`.

---

## 🚀 RunPod Serverless Deployment Guide

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed locally
- [Docker Hub](https://hub.docker.com/) account (free tier works)
- [RunPod](https://www.runpod.io/) account with credits

### Step 1: Build & Push Docker Images

#### Upscale Worker (~15 GB image)

```bash
cd workers/upscale
docker build -t <your-dockerhub-username>/ai-assets-upscale:latest .
docker push <your-dockerhub-username>/ai-assets-upscale:latest
```

#### Caption Worker (~8 GB image)

```bash
cd workers/caption
docker build -t <your-dockerhub-username>/ai-assets-caption:latest .
docker push <your-dockerhub-username>/ai-assets-caption:latest
```

Or use the deploy scripts:
```bash
# Linux/macOS
./scripts/deploy.sh <your-dockerhub-username> all

# Windows PowerShell
.\scripts\deploy.ps1 -DockerUser <your-dockerhub-username> -Worker all
```

### Step 2: Create RunPod Serverless Endpoints

#### Upscale Endpoint

1. Go to [RunPod Console](https://www.runpod.io/console/serverless) → **Serverless** → **New Endpoint**
2. **Endpoint Name**: `ai-assets-upscale`
3. **Container Image**: `<your-dockerhub-username>/ai-assets-upscale:latest`
4. **GPU**: Select **A100 80GB** (recommended) or **A40 48GB** (minimum)
   - The upscale worker needs ~20-25 GB VRAM for SDXL + ControlNet + IP-Adapter
5. **Min Workers**: `0` (scales to zero when idle — no cost)
6. **Max Workers**: `1-3` (depending on your budget)
7. **Idle Timeout**: `60` seconds (worker stays warm for 60s after last request)
8. **Network Volume** (optional): Attach a network volume for persistent LoRA storage
   - Mount path: `/runpod-volume`
   - This allows uploading/managing LoRA models that persist across worker restarts
9. Click **Create Endpoint**
10. Copy the **Endpoint ID** (e.g., `abc123def456`)

#### Caption Endpoint

1. **New Endpoint** → **Endpoint Name**: `ai-assets-caption`
2. **Container Image**: `<your-dockerhub-username>/ai-assets-caption:latest`
3. **GPU**: Select **RTX 4090** or **A40** (Qwen3-VL-2B needs only ~5 GB VRAM)
   - This is a much lighter worker — cheaper GPU is fine
4. **Min Workers**: `0`
5. **Max Workers**: `1`
6. **Idle Timeout**: `60` seconds
7. **No network volume needed**
8. Click **Create Endpoint**
9. Copy the **Endpoint ID**

### Step 3: Get Your API Key

1. Go to [RunPod Console](https://www.runpod.io/console/user/settings) → **Settings** → **API Keys**
2. Create a new API key or copy your existing one

### Step 4: Configure the Frontend

Create a `.env` file in the `frontend/` directory:

```bash
cp frontend/.env.example frontend/.env
```

Edit `frontend/.env`:
```env
RUNPOD_API_KEY=your_runpod_api_key_here
RUNPOD_UPSCALE_ENDPOINT_ID=your_upscale_endpoint_id
RUNPOD_CAPTION_ENDPOINT_ID=your_caption_endpoint_id
```

### Step 5: Run the Frontend

```bash
cd frontend
pip install -r requirements.txt
python app.py
```

The Gradio app will open at `http://localhost:7860`.

### Step 6: Test the Connection

1. Open the app in your browser
2. Go to the **LoRA Manager** tab → click **Refresh Models** to test the upscale endpoint connection
3. Upload an image in the **Tile Upscale** tab → click **Caption All** to test the caption endpoint

### Cost Estimates

| Worker | GPU | Cost/hr | Typical Usage |
|--------|-----|---------|---------------|
| Upscale | A100 80GB | ~$1.50/hr | ~30s per tile upscale |
| Upscale | A40 48GB | ~$0.75/hr | ~45s per tile upscale |
| Caption | RTX 4090 | ~$0.45/hr | ~5s per tile caption |
| Caption | A40 48GB | ~$0.75/hr | ~3s per tile caption |

With **Min Workers = 0**, you only pay when actively processing. A typical 4×4 tile upscale session costs ~$0.10-0.20.

### Troubleshooting

- **Cold start delay**: First request after idle takes 30-120s while the worker boots and loads models
- **Timeout errors**: Increase the RunPod endpoint timeout to 300s for upscale operations
- **VRAM errors**: Make sure you selected a GPU with enough VRAM (A100 80GB recommended for upscale)
- **Connection refused**: Check that your API key and endpoint IDs are correct in `.env`

---

## Future Plans

- **Spritesheet animation tab** — Generate animated spritesheets for game assets using Qwen-Image-Edit on a dedicated RunPod public endpoint.

---

## License

MIT — see [`LICENSE`](LICENSE).
