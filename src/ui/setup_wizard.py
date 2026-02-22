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


def _token_status_html(saved: bool) -> str:
    """Return a small inline HTML badge indicating whether a token is saved."""
    if saved:
        return (
            '<span style="color:#4CAF50;font-size:0.9em;font-weight:600;">'
            "✅ Saved"
            "</span>"
        )
    return (
        '<span style="color:#9E9E9E;font-size:0.9em;">'
        "❌ Not set"
        "</span>"
    )


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
    combined page-load handler to ``demo.load``.  All necessary references
    are returned in the tuple.

    Returns:
        A 6-tuple ``(wizard_group, start_btn,
        on_load_fn, on_load_inputs, on_load_outputs, step1_group)`` where:

        * *wizard_group* — outer ``gr.Group`` wrapping all three steps.
        * *start_btn* — ``gr.Button`` on step 3; wire its ``.click`` with
          ``outputs=[wizard_group, tool_group]``.
        * *on_load_fn* — combined page-load function; receives BrowserState
          tokens, checks model status, and returns visibility updates for
          the wizard, tool, step1, step2 groups plus the token textboxes.
          Pass to ``demo.load`` with *on_load_inputs* / *on_load_outputs*.
        * *on_load_inputs* — list of ``gr.BrowserState`` components to
          pass as ``inputs`` to ``demo.load``.
        * *on_load_outputs* — list of components to pass as ``outputs`` to
          ``demo.load`` (textboxes + status badges + group visibility).
        * *step1_group* — the Step-1 ``gr.Group``; exposed so the caller
          can include it in ``on_load_outputs``.
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

            with gr.Row():
                hf_token_input = gr.Textbox(
                    label="HuggingFace Token (optional)",
                    type="password",
                    placeholder="hf_…",
                    scale=4,
                )
                hf_status_html = gr.HTML(
                    value=_token_status_html(False),
                    label="",
                )

            with gr.Row():
                civitai_token_input = gr.Textbox(
                    label="CivitAI API Token (optional)",
                    type="password",
                    placeholder="Paste your CivitAI token here…",
                    scale=4,
                )
                civitai_status_html = gr.HTML(
                    value=_token_status_html(False),
                    label="",
                )

            with gr.Row():
                step1_next_btn = gr.Button("Save & Continue →", variant="primary")
                step1_save_status = gr.HTML(value="", label="")

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
    # Combined page-load handler.
    #
    # Receives BrowserState tokens (available only after page load) and
    # decides which step/group to show:
    #
    #   • All models downloaded          → hide wizard, show tool
    #   • Models missing, tokens saved   → show wizard at Step 2
    #   • Models missing, no tokens      → show wizard at Step 1
    #
    # Returns a 9-tuple that maps to on_load_outputs (see below).
    # ------------------------------------------------------------------
    def _on_load(hf_tok: str, civitai_tok: str):
        """Combined page-load handler: restore tokens + route to correct step."""
        hf_tok = hf_tok or ""
        civitai_tok = civitai_tok or ""

        hf_saved = bool(hf_tok)
        civitai_saved = bool(civitai_tok)

        # Check whether all models are already present.
        all_ready = False
        try:
            from src.services.download import DownloadService  # noqa: PLC0415

            status = DownloadService().check_status.remote()
            all_ready = all(v["downloaded"] for v in status.values())
        except Exception:  # noqa: BLE001
            logger.warning("Could not check model status on page load — showing wizard.")

        if all_ready:
            # Skip wizard entirely — show the main tool.
            return (
                hf_tok,                           # hf_token_input
                civitai_tok,                      # civitai_token_input
                _token_status_html(hf_saved),     # hf_status_html
                _token_status_html(civitai_saved),# civitai_status_html
                "",                               # step1_save_status (clear)
                gr.Group(visible=False),          # wizard_group
                gr.Group(visible=True),           # tool_group  (caller provides at index 6)
                gr.Group(visible=True),           # step1_group (doesn't matter, wizard hidden)
                gr.Group(visible=False),          # step2_group
            )

        has_tokens = bool(hf_tok or civitai_tok)
        if has_tokens:
            # Models missing but tokens already saved → skip Step 1, go to Step 2.
            return (
                hf_tok,
                civitai_tok,
                _token_status_html(hf_saved),
                _token_status_html(civitai_saved),
                "",                               # step1_save_status (clear)
                gr.Group(visible=True),           # wizard_group
                gr.Group(visible=False),          # tool_group
                gr.Group(visible=False),          # step1_group  ← hidden
                gr.Group(visible=True),           # step2_group  ← shown
            )

        # No tokens yet → show Step 1 as normal.
        return (
            hf_tok,
            civitai_tok,
            _token_status_html(False),
            _token_status_html(False),
            "",                                   # step1_save_status (clear)
            gr.Group(visible=True),               # wizard_group
            gr.Group(visible=False),              # tool_group
            gr.Group(visible=True),               # step1_group
            gr.Group(visible=False),              # step2_group
        )

    # Inputs: the two BrowserState components.
    on_load_inputs = [hf_token_state, civitai_token_state]
    # Outputs (wizard-side): textboxes + status badges + wizard_group + step1_group + step2_group.
    # The caller must insert tool_group at index 6 when wiring demo.load:
    #
    #   full_outputs = (
    #       on_load_outputs[:6]   # hf_input, civitai_input, hf_status,
    #                             # civitai_status, save_status, wizard_group
    #       + [tool_group]        # injected by caller  (index 6 in _on_load)
    #       + on_load_outputs[6:] # step1_group, step2_group
    #   )
    on_load_outputs_wizard = [
        hf_token_input,        # 0
        civitai_token_input,   # 1
        hf_status_html,        # 2
        civitai_status_html,   # 3
        step1_save_status,     # 4
        wizard_group,          # 5
        # tool_group at index 6 — injected by caller
        step1_group,           # 7 (after caller inserts tool_group)
        step2_group,           # 8
    ]

    # ------------------------------------------------------------------
    # Event: Step 1 → Step 2 (save tokens, advance)
    # ------------------------------------------------------------------
    def _save_and_advance(hf_tok: str, civitai_tok: str):
        """Persist tokens to BrowserState, show feedback, then advance to step 2."""
        hf_tok = hf_tok or ""
        civitai_tok = civitai_tok or ""
        hf_saved = bool(hf_tok)
        civitai_saved = bool(civitai_tok)
        save_feedback = (
            '<span style="color:#4CAF50;font-size:0.9em;font-weight:600;">'
            "✅ Keys saved!"
            "</span>"
        )
        return (
            gr.Group(visible=False),          # step1_group
            gr.Group(visible=True),           # step2_group
            _token_status_html(hf_saved),     # hf_status_html
            _token_status_html(civitai_saved),# civitai_status_html
            save_feedback,                    # step1_save_status
            hf_tok,                           # hf_token_state
            civitai_tok,                      # civitai_token_state
        )

    step1_next_btn.click(
        fn=_save_and_advance,
        inputs=[hf_token_input, civitai_token_input],
        outputs=[
            step1_group,
            step2_group,
            hf_status_html,
            civitai_status_html,
            step1_save_status,
            hf_token_state,
            civitai_token_state,
        ],
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
    # The caller must register the combined page-load handler on
    # demo.load, inserting tool_group into the outputs list at index 6:
    #
    #   full_outputs = (
    #       on_load_outputs[:5]          # hf_input, civitai_input,
    #                                    # hf_status, civitai_status, save_status
    #       + [tool_group]               # injected by caller  (index 6 in _on_load)
    #       + on_load_outputs[5:]        # wizard_group, step1_group, step2_group
    #   )
    #   demo.load(fn=on_load_fn, inputs=on_load_inputs, outputs=full_outputs)
    # ------------------------------------------------------------------

    return (
        wizard_group,
        start_btn,
        _on_load,
        on_load_inputs,
        on_load_outputs_wizard,
        step1_group,
    )
