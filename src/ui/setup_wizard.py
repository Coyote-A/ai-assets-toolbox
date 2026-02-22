"""Setup wizard UI for first-time model download and API key configuration."""
from __future__ import annotations

import logging

import gradio as gr

from src.services.model_registry import ALL_MODELS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _build_progress_html(status: dict, progress: dict) -> str:
    """
    Render a styled HTML grid showing per-model download status.

    Args:
        status: Dict returned by ``DownloadService().check_status.remote()``.
                Keys are model keys; values have ``downloaded``, ``size_bytes``,
                ``description``.
        progress: Dict returned by ``DownloadService().get_progress.remote()``.
                  Keys are model keys; values have ``status`` and optionally
                  ``percentage`` / ``error``.

    Returns:
        An HTML string suitable for ``gr.HTML``.
    """
    rows: list[str] = []
    for key, info in status.items():
        if info["downloaded"]:
            icon = "✅"
            color = "#4CAF50"
            state = "Downloaded"
        elif key in progress and progress[key].get("status") == "downloading":
            pct = progress[key].get("percentage")
            pct_str = f" ({pct:.0f}%)" if pct is not None else ""
            icon = "⏳"
            color = "#FF9800"
            state = f"Downloading…{pct_str}"
        elif key in progress and progress[key].get("status") == "error":
            icon = "❌"
            color = "#f44336"
            state = f"Error: {progress[key].get('error', 'unknown')}"
        elif key in progress and progress[key].get("status") == "completed":
            icon = "✅"
            color = "#4CAF50"
            state = "Downloaded"
        else:
            icon = "⬜"
            color = "#9E9E9E"
            state = "Not downloaded"

        size_gb = info["size_bytes"] / 1_000_000_000
        rows.append(
            f"""
            <div style="display:flex;align-items:center;padding:10px 12px;
                        border-bottom:1px solid #333;">
                <span style="font-size:1.5em;margin-right:14px;">{icon}</span>
                <div style="flex:1;">
                    <div style="font-weight:bold;color:{color};">
                        {info["description"]}
                    </div>
                    <div style="font-size:0.85em;color:#888;">
                        {size_gb:.1f} GB — {state}
                    </div>
                </div>
            </div>"""
        )

    inner = "".join(rows)
    return (
        f'<div style="border:1px solid #444;border-radius:8px;overflow:hidden;">'
        f"{inner}"
        f"</div>"
    )


def _initial_status_html() -> str:
    """Build the initial (all-not-downloaded) HTML from ALL_MODELS."""
    status = {
        m.key: {
            "downloaded": False,
            "size_bytes": m.size_bytes,
            "description": m.description,
        }
        for m in ALL_MODELS
    }
    return _build_progress_html(status, {})


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def create_setup_wizard() -> tuple:
    """
    Create the setup wizard UI components inside the current ``gr.Blocks`` context.

    The wizard has three steps rendered as ``gr.Group`` blocks with toggled
    visibility:

    * **Step 1** — API keys (HuggingFace + CivitAI), persisted via
      ``gr.BrowserState``.
    * **Step 2** — Model downloads with live progress polling via
      ``gr.Timer``.
    * **Step 3** — "Ready" screen with a *Start* button.

    The caller is responsible for wiring the *Start* button and the
    token-restore handler to ``demo.load``.  All necessary references are
    returned in the tuple.

    Returns:
        A 6-tuple ``(wizard_group, check_fn, start_btn,
        restore_tokens_fn, restore_inputs, restore_outputs)`` where:

        * *wizard_group* — outer ``gr.Group`` wrapping all three steps.
        * *check_fn* — callable returning ``True`` when all models are
          already downloaded.
        * *start_btn* — ``gr.Button`` on step 3; wire its ``.click`` with
          ``outputs=[wizard_group, tool_group]``.
        * *restore_tokens_fn* — function to pass to ``demo.load`` to
          pre-fill token textboxes from BrowserState.
        * *restore_inputs* — list of ``gr.BrowserState`` components to
          pass as ``inputs`` to ``demo.load``.
        * *restore_outputs* — list of ``gr.Textbox`` components to pass
          as ``outputs`` to ``demo.load``.
    """

    # ------------------------------------------------------------------
    # BrowserState — persists tokens in the browser's localStorage
    # ------------------------------------------------------------------
    hf_token_state = gr.BrowserState("", storage_key="hf_token")
    civitai_token_state = gr.BrowserState("", storage_key="civitai_token")

    # ------------------------------------------------------------------
    # Outer wrapper — hidden/shown by the parent app
    # ------------------------------------------------------------------
    with gr.Group(visible=True) as wizard_group:

        gr.Markdown("# 🛠️ AI Assets Toolbox — First-Time Setup")
        gr.Markdown(
            "Welcome! Before you can use the toolbox, the required AI models "
            "need to be downloaded to the server. This only happens once."
        )

        # --------------------------------------------------------------
        # Step 1: API Keys
        # --------------------------------------------------------------
        with gr.Group(visible=True) as step1_group:
            gr.Markdown("## Step 1 of 3 — API Keys")
            gr.Markdown(
                "These tokens are stored **only in your browser** (localStorage) "
                "and are never sent to any third-party server.\n\n"
                "* **HuggingFace token** — required only for gated models.\n"
                "* **CivitAI token** — required only for LoRA downloads."
            )

            hf_token_input = gr.Textbox(
                label="HuggingFace Token (optional)",
                type="password",
                placeholder="hf_…",
            )
            civitai_token_input = gr.Textbox(
                label="CivitAI API Token (optional)",
                type="password",
                placeholder="Paste your CivitAI token here…",
            )

            step1_next_btn = gr.Button("Save & Continue →", variant="primary")

        # --------------------------------------------------------------
        # Step 2: Model Downloads
        # --------------------------------------------------------------
        with gr.Group(visible=False) as step2_group:
            gr.Markdown("## Step 2 of 3 — Download Models")
            gr.Markdown(
                "Click **Download All** to start downloading the required models "
                "to the server. Progress updates every 3 seconds."
            )

            download_status_html = gr.HTML(
                value=_initial_status_html(),
                label="Model Status",
                sanitize_html=False,
            )

            with gr.Row():
                download_btn = gr.Button("⬇️ Download All", variant="primary")
                step2_status_tb = gr.Textbox(
                    label="Status",
                    interactive=False,
                    scale=3,
                )

            # Timer polls every 3 s while downloads are in flight
            poll_timer = gr.Timer(value=3, active=False)

        # --------------------------------------------------------------
        # Step 3: Ready
        # --------------------------------------------------------------
        with gr.Group(visible=False) as step3_group:
            gr.Markdown("## Step 3 of 3 — Ready! 🎉")
            gr.Markdown(
                "All models have been downloaded successfully. "
                "Click **Start** to open the toolbox."
            )
            start_btn = gr.Button("🚀 Start", variant="primary", size="lg")

    # ------------------------------------------------------------------
    # Helper: check if all models are already downloaded
    # ------------------------------------------------------------------
    def check_models_downloaded() -> bool:
        """Return True if every model is already present on the volume."""
        try:
            from src.services.download import DownloadService  # noqa: PLC0415

            status = DownloadService().check_status.remote()
            return all(v["downloaded"] for v in status.values())
        except Exception:  # noqa: BLE001
            logger.warning("Could not check model status — assuming not ready.")
            return False

    # ------------------------------------------------------------------
    # Event: restore tokens from BrowserState on page load.
    # BrowserState values are available after the page loads; we wire
    # this via the parent's demo.load by returning the necessary
    # inputs/outputs so the caller can register the handler.
    # ------------------------------------------------------------------
    def _restore_tokens(hf_tok: str, civitai_tok: str):
        """Populate the token textboxes from BrowserState on page load."""
        return hf_tok or "", civitai_tok or ""

    # Expose the restore handler and its I/O so the parent can wire it
    # to demo.load (BrowserState values are only available after load).
    restore_token_inputs = [hf_token_state, civitai_token_state]
    restore_token_outputs = [hf_token_input, civitai_token_input]

    # ------------------------------------------------------------------
    # Event: Step 1 → Step 2 (save tokens, advance)
    # ------------------------------------------------------------------
    def _save_and_advance(hf_tok: str, civitai_tok: str):
        """Persist tokens to BrowserState and show step 2."""
        return (
            gr.Group(visible=False),   # step1_group
            gr.Group(visible=True),    # step2_group
            hf_tok or "",              # hf_token_state
            civitai_tok or "",         # civitai_token_state
        )

    step1_next_btn.click(
        fn=_save_and_advance,
        inputs=[hf_token_input, civitai_token_input],
        outputs=[step1_group, step2_group, hf_token_state, civitai_token_state],
    )

    # ------------------------------------------------------------------
    # Event: Start Downloads button
    # ------------------------------------------------------------------
    def _start_downloads(hf_tok: str):
        """Kick off the download_all job and activate the polling timer."""
        try:
            from src.services.download import DownloadService  # noqa: PLC0415

            # Fire-and-forget: download_all runs in a separate CPU container.
            # We don't await the result here — the timer will poll for progress.
            DownloadService().download_all.spawn(hf_tok or None)
            return "⏳ Downloads started — polling for progress…", gr.Timer(active=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to start downloads")
            return f"❌ Failed to start downloads: {exc}", gr.Timer(active=False)

    download_btn.click(
        fn=_start_downloads,
        inputs=[hf_token_state],
        outputs=[step2_status_tb, poll_timer],
    )

    # ------------------------------------------------------------------
    # Event: Timer tick — poll progress
    # ------------------------------------------------------------------
    def _poll_progress():
        """Poll DownloadService for status and progress; update HTML."""
        try:
            from src.services.download import DownloadService  # noqa: PLC0415

            status = DownloadService().check_status.remote()
            progress = DownloadService().get_progress.remote()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Progress poll failed: %s", exc)
            return (
                _initial_status_html(),
                f"⚠️ Poll error: {exc}",
                gr.Timer(active=True),   # keep trying
                gr.Group(visible=True),  # step2_group stays visible
                gr.Group(visible=False), # step3_group stays hidden
            )

        html = _build_progress_html(status, progress)
        all_done = all(v["downloaded"] for v in status.values())

        if all_done:
            return (
                html,
                "✅ All models downloaded!",
                gr.Timer(active=False),   # stop polling
                gr.Group(visible=False),  # hide step2
                gr.Group(visible=True),   # show step3
            )

        # Count how many are done
        done_count = sum(1 for v in status.values() if v["downloaded"])
        total = len(status)
        status_msg = f"⏳ Downloading… {done_count}/{total} models complete."
        return (
            html,
            status_msg,
            gr.Timer(active=True),   # keep polling
            gr.Group(visible=True),  # step2 stays visible
            gr.Group(visible=False), # step3 stays hidden
        )

    poll_timer.tick(
        fn=_poll_progress,
        outputs=[
            download_status_html,
            step2_status_tb,
            poll_timer,
            step2_group,
            step3_group,
        ],
    )

    # ------------------------------------------------------------------
    # Note: start_btn.click is NOT wired here.
    # The caller receives start_btn and wires it with the appropriate
    # outputs (wizard_group + tool_group) after both groups exist.
    #
    # Similarly, the caller must register the token-restore handler on
    # demo.load using the returned restore_* values:
    #
    #   demo.load(
    #       fn=restore_tokens_fn,
    #       inputs=restore_token_inputs,
    #       outputs=restore_token_outputs,
    #   )
    # ------------------------------------------------------------------

    return (
        wizard_group,
        check_models_downloaded,
        start_btn,
        _restore_tokens,
        restore_token_inputs,
        restore_token_outputs,
    )
