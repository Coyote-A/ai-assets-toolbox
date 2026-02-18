"""
RunPod API client for the AI Assets Toolbox frontend.

Handles all communication with the RunPod serverless endpoint:
  - Synchronous jobs via /runsync
  - Asynchronous jobs via /run + polling
  - Caption, upscale, and model management actions
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

import config


class RunPodError(Exception):
    """Raised when the RunPod API returns an error or times out."""


class RunPodClient:
    """Thin wrapper around the RunPod serverless REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_id: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or config.RUNPOD_API_KEY
        self.endpoint_id = endpoint_id or config.RUNPOD_ENDPOINT_ID
        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, payload: Dict[str, Any], timeout: int = config.RUNSYNC_TIMEOUT) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            resp = requests.post(url, json=payload, headers=self._headers(), timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            raise RunPodError(f"Request to {path} timed out after {timeout}s")
        except requests.exceptions.HTTPError as exc:
            raise RunPodError(f"HTTP {exc.response.status_code} from RunPod: {exc.response.text}") from exc
        except requests.exceptions.RequestException as exc:
            raise RunPodError(f"Network error: {exc}") from exc

    def _get(self, path: str, timeout: int = 30) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            raise RunPodError(f"GET {path} timed out")
        except requests.exceptions.HTTPError as exc:
            raise RunPodError(f"HTTP {exc.response.status_code}: {exc.response.text}") from exc
        except requests.exceptions.RequestException as exc:
            raise RunPodError(f"Network error: {exc}") from exc

    @staticmethod
    def _extract_output(response: Dict[str, Any]) -> Any:
        """Pull the output from a completed RunPod response, raising on failure."""
        status = response.get("status", "")
        if status == "FAILED":
            raise RunPodError(f"RunPod job failed: {response.get('error', 'unknown error')}")
        output = response.get("output")
        if output is None:
            raise RunPodError(f"RunPod response missing 'output': {response}")
        return output

    # ------------------------------------------------------------------
    # Core job methods
    # ------------------------------------------------------------------

    def run_sync(self, action: str, payload: Dict[str, Any]) -> Any:
        """
        Submit a job to /runsync and return the output directly.

        RunPod will block until the job completes (up to RUNSYNC_TIMEOUT seconds).
        """
        body = {"input": {"action": action, **payload}}
        response = self._post("/runsync", body, timeout=config.RUNSYNC_TIMEOUT)
        return self._extract_output(response)

    def run_async(self, action: str, payload: Dict[str, Any]) -> str:
        """
        Submit a job to /run and return the job_id for later polling.
        """
        body = {"input": {"action": action, **payload}}
        response = self._post("/run", body, timeout=30)
        job_id = response.get("id")
        if not job_id:
            raise RunPodError(f"RunPod /run did not return a job id: {response}")
        return job_id

    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Return the raw status response for a job."""
        return self._get(f"/status/{job_id}")

    def wait_for_job(self, job_id: str) -> Any:
        """
        Poll get_status until the job is complete, then return the output.
        Raises RunPodError on failure or timeout.
        """
        deadline = time.time() + config.POLL_TIMEOUT
        while time.time() < deadline:
            response = self.get_status(job_id)
            status = response.get("status", "")
            if status in ("COMPLETED", "FAILED", "CANCELLED"):
                return self._extract_output(response)
            time.sleep(config.POLL_INTERVAL)
        raise RunPodError(f"Job {job_id} did not complete within {config.POLL_TIMEOUT}s")

    # ------------------------------------------------------------------
    # High-level action methods
    # ------------------------------------------------------------------

    def caption_tiles(
        self,
        tiles_b64: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Send tiles for captioning via Qwen2.5-VL.

        Args:
            tiles_b64: list of {"tile_id": str, "image_b64": str}
            system_prompt: optional override for the captioning system prompt

        Returns:
            list of {"tile_id": str, "caption": str}
        """
        caption_params: Dict[str, Any] = {"max_tokens": 200}
        if system_prompt:
            caption_params["system_prompt"] = system_prompt

        output = self.run_sync(
            "caption",
            {"tiles": tiles_b64, "caption_params": caption_params},
        )
        return output.get("captions", [])

    def upscale_tiles(self, tiles_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Send tiles for upscaling through the diffusion img2img pipeline.

        Each element of tiles_data should contain:
            tile_id, image_b64, prompt_override (optional),
            model, model_type, lora_name (optional), lora_weight,
            controlnet_enabled, conditioning_scale,
            strength, steps, cfg_scale, seed,
            global_prompt, negative_prompt

        Returns:
            list of {"tile_id": str, "image_b64": str, "seed_used": int}
        """
        if not tiles_data:
            return []

        # Build the request from the first tile's shared settings
        first = tiles_data[0]

        loras = []
        if first.get("lora_name"):
            loras = [{"name": first["lora_name"], "weight": first.get("lora_weight", 0.7)}]

        model_config: Dict[str, Any] = {
            "base_model": first.get("model", "z-image-xl"),
            "model_type": first.get("model_type", "sdxl"),
            "loras": loras,
            "controlnet": {
                "enabled": first.get("controlnet_enabled", True),
                "model": "sdxl-tile",
                "conditioning_scale": first.get("conditioning_scale", 0.7),
            },
        }

        generation_params: Dict[str, Any] = {
            "steps": first.get("steps", 30),
            "cfg_scale": first.get("cfg_scale", 7.0),
            "denoising_strength": first.get("strength", 0.5),
            "seed": first.get("seed", -1),
        }

        tiles_payload = [
            {
                "tile_id": t["tile_id"],
                "image_b64": t["image_b64"],
                "prompt_override": t.get("prompt_override"),
            }
            for t in tiles_data
        ]

        output = self.run_sync(
            "upscale",
            {
                "model_config": model_config,
                "generation_params": generation_params,
                "global_prompt": first.get("global_prompt", ""),
                "negative_prompt": first.get("negative_prompt", ""),
                "tiles": tiles_payload,
            },
        )
        return output.get("tiles", [])

    def list_models(self, model_type: str, base_model_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List models available on RunPod network storage.

        Args:
            model_type: "checkpoint", "lora", or "controlnet"
            base_model_filter: optional filter, e.g. "sdxl" or "flux"

        Returns:
            list of model dicts with keys: name, path, size_mb, base_model
        """
        payload: Dict[str, Any] = {"model_type": model_type}
        if base_model_filter:
            payload["base_model_filter"] = base_model_filter
        output = self.run_sync("list_models", payload)
        return output.get("models", [])

    def upload_model(
        self,
        name: str,
        model_type: str,
        file_data: bytes,
        base_model: str = "sdxl",
    ) -> Dict[str, Any]:
        """
        Upload a model file to RunPod network storage.

        For large files this sends the entire file as a single base64 chunk.
        Returns the response output dict with keys: status, path, size_mb.
        """
        import base64

        file_b64 = base64.b64encode(file_data).decode("utf-8")
        output = self.run_sync(
            "upload_model",
            {
                "filename": name,
                "model_type": model_type,
                "base_model": base_model,
                "file_b64": file_b64,
                "chunk_index": 0,
                "total_chunks": 1,
            },
        )
        return output

    def delete_model(self, path: str) -> Dict[str, Any]:
        """
        Delete a model from RunPod network storage by its storage path.

        Returns the response output dict.
        """
        output = self.run_sync("delete_model", {"path": path})
        return output

    def health_check(self) -> Dict[str, Any]:
        """Run a health check against the RunPod endpoint."""
        output = self.run_sync("health", {})
        return output
