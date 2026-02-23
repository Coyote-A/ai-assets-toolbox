# Debug Mode Rules (Non-Obvious Only)

## Modal-Specific Debugging
- Check model download status via `is_model_downloaded(key)` from `src/services/model_registry.py` — don't assume weights exist.
- Manifest files at `/vol/models/.manifest.json` track downloaded models; `.progress.json` tracks active downloads.
- GPU services have 10-min timeout, 5-min scaledown — long operations may timeout.

## Common Failure Points
- Missing `add_local_python_source("src")` on images causes import errors inside containers.
- LoRA path confusion: `/vol/loras` vs `/vol/loras/loras` — check which one code expects.
- Token loss on restart: Gradio BrowserState is ephemeral; check `modal.Dict` for persisted tokens.

## Logs & Debugging Commands
- `modal app logs ai-toolbox` — Stream logs for the deployed app (Ctrl+C to stop).
- `modal app list` — List all deployed apps.
- `modal volume list` — List volumes (check `ai-toolbox-models`, `ai-toolbox-loras`).
- For deployment debugging: wrap `app.deploy()` in `with modal.enable_output():` for verbose output.

## Modal Execution Model
- Modal always executes in the cloud, even during development (`modal serve`).
- Global scope code runs in ALL environments — avoid heavy imports at module level.
