# Architect Mode Rules (Non-Obvious Only)

## Architectural Constraints
- **Single Modal App** — All services MUST register against shared `app` from `src/app_config`. No separate apps per service.
- **Volume-based models** — Weights stored in Modal Volumes, NOT in Docker images. Model updates don't require image rebuilds.
- **No HTTP between services** — Use `.remote()` calls. Images passed as raw `bytes` (cloudpickle serialization).

## Service Isolation
- Each service has its own Modal Image with specific dependencies (see `src/app_config.py`).
- GPU services are stateless — model loading happens in `@modal.enter()`, not `__init__`.
- Container lifecycle: `@modal.enter()` → method calls → `@modal.exit()` on scaledown.

## Scaling & Cost Considerations
- GPU containers scale to zero after 5-min idle (configurable via `scaledown_window`).
- Cold start: ~30-60s for GPU services (model loading from volume).
- UI container: 10-min idle timeout, handles multiple users via `@modal.concurrent(max_inputs=100)`.

## Data Persistence
- `ai-toolbox-models` Volume: Base model weights (read-only after download).
- `ai-toolbox-loras` Volume: User-uploaded LoRAs + metadata (read-write).
- `ai-toolbox-tokens` Dict: API tokens (survives restarts, unlike Gradio BrowserState).

## Adding New Services
1. Create service class with `@app.cls()` decorator in `src/gpu/` or `src/services/`.
2. Import in `src/app.py` to trigger registration.
3. Define Image in `src/app_config.py` with `.add_local_python_source("src")`.
4. Mount volumes as needed in `volumes={}` parameter.
