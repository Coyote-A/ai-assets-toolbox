# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## External Documentation
- Modal docs: modal.com/docs, modal.com/llms.txt (for LLMs), modal.com/llms-full.txt (full reference)

## Commands
```powershell
.\serve.ps1    # Dev mode (hot-reload, temporary URL) — USE THIS for development
.\deploy.ps1   # Production deploy (permanent URL) — DO NOT USE unless explicitly asked
```
Both scripts auto-setup Python 3.11+, Modal CLI, auth, and volumes via `scripts/common.ps1`.

**Important:** Always use `.\serve.ps1` for testing. It allows seeing logs directly and saves tokens by shutting down when done. Never deploy without explicit user request.

## Debugging Modal Volumes
```powershell
modal volume ls ai-toolbox-models              # List root contents
modal volume ls ai-toolbox-models ip-adapter   # List specific model directory
```
Use this to verify model files exist before assuming download issues.

## Architecture (Non-Obvious)
- **Single Modal app** — All services register against `app` from `src.app_config` via `@app.cls()`. Import order matters: GPU services must be imported in `src/app.py` to register.
- **Models in Volumes** — Weights stored in Modal Volumes (`ai-toolbox-models`, `ai-toolbox-loras`), NOT baked into Docker images. Mount paths: `/vol/models`, `/vol/loras`.
- **No HTTP between services** — GPU work dispatched via `.remote()` calls from Gradio UI. Images passed as raw `bytes` (Modal's cloudpickle handles serialization).
- **Gradio 6.6 ASGI** — Must use `gr.mount_gradio_app(fastapi_app, blocks, path="/")` — Blocks is not directly callable as ASGI app.

## Critical Patterns
- **Image imports** — All Modal images need `.add_local_python_source("src")` for `from src.xxx` imports to work inside containers.
- **Metadata storage** — Model manifest and download progress stored in `modal.Dict.from_name("ai-toolbox-model-metadata")` via `MetadataStore` class. No file I/O needed; Dict is always current across containers.
- **Token persistence** — Uses `modal.Dict.from_name("ai-toolbox-tokens")` because Gradio's BrowserState rotates encryption keys on restart.
- **Model validation** — `is_model_downloaded()` checks both the manifest entry in Dict AND file existence on volume (defense in depth).
- **LoRA paths** — Two paths exist: `/vol/loras` (mount point) and `/vol/loras/loras` (subdirectory for actual files). Check `LORAS_DIR` vs `LORAS_SUBDIR` in `src/gpu/upscale.py`.
- **Model file locations** — Single-file models (like IP-Adapter) may be in flattened location (`ip-adapter/file.safetensors`) OR still in subfolder (`ip-adapter/sdxl_models/file.safetensors`). Always use `get_model_file_path()` to find the actual file location, not hardcoded paths.

## Service GPU Requirements
| Service | GPU | Timeout | Scaledown |
|---------|-----|---------|-----------|
| CaptionService | T4 | 10 min | 5 min |
| UpscaleService | A10G | 10 min | 5 min |
| DownloadService | CPU | 30 min | N/A |
| web_ui | CPU | 1 hour | 10 min |
