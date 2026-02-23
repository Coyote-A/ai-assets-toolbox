# Code Mode Rules (Non-Obvious Only)

## Workflow
- **Always use `.\serve.ps1`** for development — never deploy without explicit user request. Serve mode shows logs directly and saves tokens by shutting down when done.
- **Always commit changes** after completing a task.
- **Update AGENTS.md** when discovering important non-obvious patterns.

## Debugging Modal Volumes
```powershell
modal volume ls ai-toolbox-models              # List root contents
modal volume ls ai-toolbox-models ip-adapter   # List specific model directory
```
Use this to verify model files exist before assuming download issues.

## Modal Service Registration
- GPU services MUST be imported in `src/app.py` to register with the shared `app`. Import order matters — services register via `@app.cls()` decorator at import time.
- All Modal images need `.add_local_python_source("src")` for `from src.xxx` imports to work inside containers.

## Image Handling
- Images passed as raw `bytes` between services (not base64). Modal's cloudpickle handles serialization.
- Gradio 6.0: Use `gr.mount_gradio_app(fastapi_app, blocks, path="/")` — Blocks is not directly callable as ASGI app.

## Volume Paths
- Models: `/vol/models` mount point, manifest at `/vol/models/.manifest.json`
- LoRAs: Two paths exist — `/vol/loras` (mount point) vs `/vol/loras/loras` (subdirectory for files). Check `LORAS_DIR` vs `LORAS_SUBDIR` in `src/gpu/upscale.py`.

## Model File Locations
- Single-file models (like IP-Adapter) may be in **flattened location** (`ip-adapter/file.safetensors`) OR still in **subfolder** (`ip-adapter/sdxl_models/file.safetensors`).
- Always use `get_model_file_path()` from `src/services/model_registry.py` to find the actual file location — never hardcode paths.
- The download code attempts to flatten subfolders but this may not always happen.

## Token Storage
- Use `modal.Dict.from_name("ai-toolbox-tokens")` for persistence. Gradio's BrowserState rotates encryption keys on restart, losing data.
