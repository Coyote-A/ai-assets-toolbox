# Code Mode Rules (Non-Obvious Only)

## Modal Service Registration
- GPU services MUST be imported in `src/app.py` to register with the shared `app`. Import order matters — services register via `@app.cls()` decorator at import time.
- All Modal images need `.add_local_python_source("src")` for `from src.xxx` imports to work inside containers.

## Image Handling
- Images passed as raw `bytes` between services (not base64). Modal's cloudpickle handles serialization.
- Gradio 6.0: Use `gr.mount_gradio_app(fastapi_app, blocks, path="/")` — Blocks is not directly callable as ASGI app.

## Volume Paths
- Models: `/vol/models` mount point, manifest at `/vol/models/.manifest.json`
- LoRAs: Two paths exist — `/vol/loras` (mount point) vs `/vol/loras/loras` (subdirectory for files). Check `LORAS_DIR` vs `LORAS_SUBDIR` in `src/gpu/upscale.py`.

## Token Storage
- Use `modal.Dict.from_name("ai-toolbox-tokens")` for persistence. Gradio's BrowserState rotates encryption keys on restart, losing data.
