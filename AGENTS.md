# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## External Documentation
- Modal docs: modal.com/docs, modal.com/llms.txt (for LLMs), modal.com/llms-full.txt (full reference)

## Commands
```powershell
.\serve.ps1    # Dev mode (hot-reload, temporary URL)
.\deploy.ps1   # Production deploy (permanent URL)
```
Both scripts auto-setup Python 3.11+, Modal CLI, auth, and volumes via `scripts/common.ps1`.

## Architecture (Non-Obvious)
- **Single Modal app** — All services register against `app` from `src.app_config` via `@app.cls()`. Import order matters: GPU services must be imported in `src/app.py` to register.
- **Models in Volumes** — Weights stored in Modal Volumes (`ai-toolbox-models`, `ai-toolbox-loras`), NOT baked into Docker images. Mount paths: `/vol/models`, `/vol/loras`.
- **No HTTP between services** — GPU work dispatched via `.remote()` calls from Gradio UI. Images passed as raw `bytes` (Modal's cloudpickle handles serialization).
- **Gradio 6.0 ASGI** — Must use `gr.mount_gradio_app(fastapi_app, blocks, path="/")` — Blocks is not directly callable as ASGI app.

## Critical Patterns
- **Image imports** — All Modal images need `.add_local_python_source("src")` for `from src.xxx` imports to work inside containers.
- **Token persistence** — Uses `modal.Dict.from_name("ai-toolbox-tokens")` because Gradio's BrowserState rotates encryption keys on restart.
- **Model manifest** — Download progress tracked in `.progress.json`, completion in `.manifest.json` at volume root. Use `is_model_downloaded()` to check before assuming weights exist.
- **LoRA paths** — Two paths exist: `/vol/loras` (mount point) and `/vol/loras/loras` (subdirectory for actual files). Check `LORAS_DIR` vs `LORAS_SUBDIR` in `src/gpu/upscale.py`.

## Service GPU Requirements
| Service | GPU | Timeout | Scaledown |
|---------|-----|---------|-----------|
| CaptionService | T4 | 10 min | 5 min |
| UpscaleService | A10G | 10 min | 5 min |
| DownloadService | CPU | 30 min | N/A |
| web_ui | CPU | 1 hour | 10 min |
