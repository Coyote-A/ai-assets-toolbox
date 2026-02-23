# Ask Mode Rules (Non-Obvious Only)

## Documentation Context
- Main docs: `docs/MODEL_MANAGER_DESIGN.md`, `docs/SETUP_WIZARD_DESIGN.md`, `docs/TILE_GRID_DESIGN.md`
- Modal docs: `docs/MODAL_DEPLOY.md` + modal.com/docs for platform reference.

## Architecture Clarifications
- **"src/" is the entire codebase** — no separate frontend/backend. UI is Gradio in `src/ui/`.
- **Models NOT in repo** — weights stored in Modal Volumes, downloaded at runtime via setup wizard.
- **No HTTP API** — services communicate via Modal `.remote()` calls, not REST endpoints.

## Service Responsibilities
- `CaptionService` (T4): Qwen2.5-VL-3B for tile captioning
- `UpscaleService` (A10G): SDXL + ControlNet Tile for upscaling
- `DownloadService` (CPU): HuggingFace/CivitAI downloads to volumes
- `web_ui` (CPU): Gradio interface, dispatches work to GPU services

## Key Files for Common Questions
- Model definitions: `src/services/model_registry.py` (`ALL_MODELS` list)
- Volume paths: `MODELS_MOUNT_PATH = "/vol/models"`, `LORAS_MOUNT_PATH = "/vol/loras"`
- GPU configs: `src/gpu/caption.py`, `src/gpu/upscale.py`
