"""
RunPod API client for the AI Assets Toolbox frontend.

Handles all communication with the RunPod serverless endpoints:
  - Synchronous jobs via /runsync
  - Asynchronous jobs via /run + polling
  - Caption, upscale, and model management actions

Multi-endpoint architecture
---------------------------
  - Upscale worker  → handles: upscale, upscale_regions, list_models,
                                upload_model, delete_model, health
  - Caption worker  → handles: caption, health

Public endpoint
---------------
  - QwenImageEditClient → RunPod public Qwen-Image-Edit endpoint
                          (for future spritesheet tab use)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

import config


class RunPodError(Exception):
    """Raised when the RunPod API returns an error or times out."""


# Actions that must be routed to the caption worker endpoint.
_CAPTION_ACTIONS = frozenset({"caption"})


class RunPodClient:
    """
    Wrapper around the RunPod serverless REST API supporting two separate
    worker endpoints: one for upscaling/model management and one for captioning.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        upscale_endpoint_id: Optional[str] = None,
        caption_endpoint_id: Optional[str] = None,
        # Legacy single-endpoint parameter kept for backward compatibility.
        endpoint_id: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or config.RUNPOD_API_KEY

        # Resolve upscale endpoint: explicit arg → config var → legacy fallback
        self.upscale_endpoint_id = (
            upscale_endpoint_id
            or config.RUNPOD_UPSCALE_ENDPOINT_ID
            or endpoint_id
            or config.RUNPOD_ENDPOINT_ID
        )

        # Resolve caption endpoint: explicit arg → config var → fall back to upscale
        self.caption_endpoint_id = (
            caption_endpoint_id
            or config.RUNPOD_CAPTION_ENDPOINT_ID
            or endpoint_id
            or self.upscale_endpoint_id
        )

        self.upscale_base_url = f"https://api.runpod.ai/v2/{self.upscale_endpoint_id}"
        self.caption_base_url = f"https://api.runpod.ai/v2/{self.caption_endpoint_id}"

        # Legacy attribute so any code that reads .base_url still works.
        self.base_url = self.upscale_base_url

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_base_url(self, action: str) -> str:
        """Route an action to the correct worker endpoint base URL."""
        if action in _CAPTION_ACTIONS:
            return self.caption_base_url
        return self.upscale_base_url

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(
        self,
        path: str,
        payload: Dict[str, Any],
        timeout: int = config.RUNSYNC_TIMEOUT,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = f"{base_url or self.upscale_base_url}{path}"
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

    def _get(
        self,
        path: str,
        timeout: int = 30,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = f"{base_url or self.upscale_base_url}{path}"
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

        The request is automatically routed to the correct worker endpoint
        based on the action name.  RunPod will block until the job completes
        (up to RUNSYNC_TIMEOUT seconds).
        """
        body = {"input": {"action": action, **payload}}
        base_url = self._get_base_url(action)
        response = self._post("/runsync", body, timeout=config.RUNSYNC_TIMEOUT, base_url=base_url)
        return self._extract_output(response)

    def run_async(self, action: str, payload: Dict[str, Any]) -> str:
        """
        Submit a job to /run and return the job_id for later polling.

        The request is automatically routed to the correct worker endpoint.
        """
        body = {"input": {"action": action, **payload}}
        base_url = self._get_base_url(action)
        response = self._post("/run", body, timeout=30, base_url=base_url)
        job_id = response.get("id")
        if not job_id:
            raise RunPodError(f"RunPod /run did not return a job id: {response}")
        return job_id

    def get_status(self, job_id: str, action: str = "") -> Dict[str, Any]:
        """
        Return the raw status response for a job.

        Pass the original *action* so the status poll is sent to the same
        endpoint that accepted the job.  Defaults to the upscale endpoint.
        """
        base_url = self._get_base_url(action) if action else self.upscale_base_url
        return self._get(f"/status/{job_id}", base_url=base_url)

    def wait_for_job(self, job_id: str, action: str = "") -> Any:
        """
        Poll get_status until the job is complete, then return the output.
        Raises RunPodError on failure or timeout.

        Pass the original *action* so polling targets the correct endpoint.
        """
        deadline = time.time() + config.POLL_TIMEOUT
        while time.time() < deadline:
            response = self.get_status(job_id, action=action)
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
        Send tiles for captioning via Qwen3-VL.

        Args:
            tiles_b64: list of {"tile_id": str, "image_b64": str}
            system_prompt: optional override for the.captioning system prompt

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

    def caption_regions(
        self,
        regions_b64: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Send region images for captioning via Qwen3-VL.

        Args:
            regions_b64: list of {"region_id": str, "image_b64": str}
            system_prompt: optional override for the captioning system prompt

        Returns:
            list of {"region_id": str, "caption": str}
        """
        caption_params: Dict[str, Any] = {"max_tokens": 200}
        if system_prompt:
            caption_params["system_prompt"] = system_prompt

        # Convert region_id to tile_id for the API (backend treats them the same)
        tiles_payload = [
            {"tile_id": r["region_id"], "image_b64": r["image_b64"]}
            for r in regions_b64
        ]

        output = self.run_sync(
            "caption",
            {"tiles": tiles_payload, "caption_params": caption_params},
        )
        # Convert tile_id back to region_id in the response
        captions = output.get("captions", [])
        return [
            {"region_id": c["tile_id"], "caption": c["caption"]}
            for c in captions
        ]

    def upscale_tiles(self, tiles_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Send tiles for upscaling through the diffusion img2img pipeline.

        Each element of tiles_data should contain:
            tile_id, image_b64, prompt_override (optional),
            model, loras (list of {name, weight}),
            controlnet_enabled, conditioning_scale,
            strength, steps, cfg_scale, seed,
            global_prompt, negative_prompt,
            ip_adapter_enabled (bool, optional),
            ip_adapter_image (base64 str, optional),
            ip_adapter_scale (float, optional)

        Returns:
            list of {"tile_id": str, "image_b64": str, "seed_used": int}
        """
        if not tiles_data:
            return []

        # Build the request from the first tile's shared settings
        first = tiles_data[0]

        # Accept loras as a list of {name, weight} dicts (new multi-LoRA format)
        loras: List[Dict[str, Any]] = first.get("loras", [])

        model_config: Dict[str, Any] = {
            "base_model": first.get("model", "illustrious-xl"),
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

        # Optional generation resolution override (target_width/target_height).
        # When provided, the backend will use these dimensions for the pipeline
        # width/height instead of deriving them from the input tile size.
        target_width: Optional[int] = first.get("target_width")
        target_height: Optional[int] = first.get("target_height")

        tiles_payload = [
            {
                "tile_id": t["tile_id"],
                "image_b64": t["image_b64"],
                "prompt_override": t.get("prompt_override"),
            }
            for t in tiles_data
        ]

        request_body: Dict[str, Any] = {
            "model_config": model_config,
            "generation_params": generation_params,
            "global_prompt": first.get("global_prompt", ""),
            "negative_prompt": first.get("negative_prompt", ""),
            "tiles": tiles_payload,
        }

        # Include target resolution when explicitly set
        if target_width is not None:
            request_body["target_width"] = int(target_width)
        if target_height is not None:
            request_body["target_height"] = int(target_height)

        # Include IP-Adapter params when enabled
        if first.get("ip_adapter_enabled"):
            request_body["ip_adapter_enabled"] = True
            request_body["ip_adapter_scale"] = first.get("ip_adapter_scale", 0.6)
            if first.get("ip_adapter_image"):
                request_body["ip_adapter_image"] = first["ip_adapter_image"]
        else:
            request_body["ip_adapter_enabled"] = False

        output = self.run_sync("upscale", request_body)
        return output.get("tiles", [])

    def upscale_regions(
        self,
        regions_data: List[Dict[str, Any]],
        original_image_b64: str,
        model: str,
        loras: Optional[List[Dict[str, Any]]],
        global_prompt: str,
        negative_prompt: str,
        strength: float,
        steps: int,
        cfg_scale: float,
        seed: int,
        controlnet_enabled: bool,
        conditioning_scale: float,
        ip_adapter_enabled: bool = False,
        ip_adapter_image: Optional[str] = None,
        ip_adapter_scale: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Send regions for upscaling through the diffusion img2img pipeline.

        Each element of regions_data should contain:
            x, y, w, h, padding, prompt, negative_prompt

        loras: list of {"name": str, "weight": float} dicts (multi-LoRA support).
        ip_adapter_image: base64-encoded style reference image (optional).

        Returns:
            list of {"region_id": str, "image_b64": str, "seed_used": int}
        """
        if not regions_data:
            return []

        model_config: Dict[str, Any] = {
            "base_model": model,
            "loras": loras if loras else [],
            "controlnet": {
                "enabled": controlnet_enabled,
                "model": "sdxl-tile",
                "conditioning_scale": conditioning_scale,
            },
        }

        generation_params: Dict[str, Any] = {
            "steps": steps,
            "cfg_scale": cfg_scale,
            "denoising_strength": strength,
            "seed": seed,
        }

        regions_payload = [
            {
                "region_id": f"region_{i}",
                "x": r.get("x", 0),
                "y": r.get("y", 0),
                "w": r.get("w", 0),
                "h": r.get("h", 0),
                "padding": r.get("padding", 64),
                "prompt": r.get("prompt", ""),
                "negative_prompt": r.get("negative_prompt", ""),
            }
            for i, r in enumerate(regions_data)
        ]

        request_body: Dict[str, Any] = {
            "model_config": model_config,
            "generation_params": generation_params,
            "global_prompt": global_prompt,
            "negative_prompt": negative_prompt,
            "regions": regions_payload,
            "source_image_b64": original_image_b64,
            "ip_adapter_enabled": ip_adapter_enabled,
        }

        if ip_adapter_enabled and ip_adapter_image:
            request_body["ip_adapter_image"] = ip_adapter_image
            request_body["ip_adapter_scale"] = ip_adapter_scale

        output = self.run_sync("upscale_regions", request_body)
        return output.get("regions", [])

    def list_models(self, model_type: str, base_model_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List models available on RunPod network storage.

        Args:
            model_type: "checkpoint", "lora", or "controlnet"
            base_model_filter: optional filter, e.g. "sdxl"

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
        """Run a health check against the upscale RunPod endpoint."""
        output = self.run_sync("health", {})
        return output

    def caption_health_check(self) -> Dict[str, Any]:
        """Run a health check against the caption RunPod endpoint."""
        output = self.run_sync("caption_health", {})
        return output


# ---------------------------------------------------------------------------
# Public endpoint client — Qwen-Image-Edit
# ---------------------------------------------------------------------------

class QwenImageEditClient:
    """
    Client for the RunPod public Qwen-Image-Edit endpoint.

    This endpoint is used for image editing tasks (e.g. background replacement,
    style transfer) and is intended for future use in the Spritesheet tab.

    Endpoint docs:
        https://docs.runpod.io/public-endpoints/models/qwen-image-edit-2511-lora

    IMPORTANT — Image URL requirement
    ----------------------------------
    The Qwen-Image-Edit endpoint requires images to be provided as **URLs**,
    not base64-encoded strings.  Before calling ``edit_image``, the caller
    must upload the source image to a publicly accessible location and pass
    the resulting URL.

    TODO (spritesheet tab): implement a helper that uploads a PIL image or
    base64 blob to a temporary hosting service (e.g. RunPod's own upload API,
    Cloudinary, or an S3 pre-signed URL) and returns a public URL, then wire
    that helper into the spritesheet workflow.
    """

    #: Default endpoint URL; can be overridden via RUNPOD_IMAGE_EDIT_ENDPOINT.
    DEFAULT_ENDPOINT = "https://api.runpod.ai/v2/qwen-image-edit-2511-lora"

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or config.RUNPOD_API_KEY
        self.endpoint_url = (
            endpoint_url
            or config.RUNPOD_IMAGE_EDIT_ENDPOINT
            or self.DEFAULT_ENDPOINT
        ).rstrip("/")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        url = f"{self.endpoint_url}{path}"
        try:
            resp = requests.post(url, json=payload, headers=self._headers(), timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            raise RunPodError(f"Request to {path} timed out after {timeout}s")
        except requests.exceptions.HTTPError as exc:
            raise RunPodError(f"HTTP {exc.response.status_code} from Qwen endpoint: {exc.response.text}") from exc
        except requests.exceptions.RequestException as exc:
            raise RunPodError(f"Network error: {exc}") from exc

    def _get(self, path: str, timeout: int = 30) -> Dict[str, Any]:
        url = f"{self.endpoint_url}{path}"
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_edit(
        self,
        image_url: str,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 28,
        seed: int = -1,
    ) -> str:
        """
        Submit an image-editing job and return the RunPod job ID.

        Args:
            image_url: Publicly accessible URL of the source image.
                       **Must be a URL — base64 is not accepted by this endpoint.**
            prompt: Text description of the desired edit.
            negative_prompt: Things to avoid in the output.
            guidance_scale: Classifier-free guidance scale (default 7.5).
            num_inference_steps: Diffusion steps (default 28).
            seed: Random seed; -1 for random.

        Returns:
            RunPod job ID string for use with ``poll_edit``.
        """
        payload = {
            "input": {
                "image": image_url,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
            }
        }
        response = self._post("/run", payload, timeout=30)
        job_id = response.get("id")
        if not job_id:
            raise RunPodError(f"Qwen endpoint /run did not return a job id: {response}")
        return job_id

    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Return the raw status response for a Qwen-Image-Edit job."""
        return self._get(f"/status/{job_id}")

    def wait_for_edit(self, job_id: str) -> str:
        """
        Poll until the edit job completes and return the output image URL.

        Returns:
            URL string pointing to the generated image.

        Raises:
            RunPodError: on job failure, cancellation, or poll timeout.
        """
        deadline = time.time() + config.POLL_TIMEOUT
        while time.time() < deadline:
            response = self.get_status(job_id)
            status = response.get("status", "")
            if status == "COMPLETED":
                output = response.get("output", {})
                image_url = output.get("image") if isinstance(output, dict) else None
                if not image_url:
                    raise RunPodError(f"Qwen job completed but output has no 'image': {response}")
                return image_url
            if status in ("FAILED", "CANCELLED"):
                raise RunPodError(
                    f"Qwen job {job_id} ended with status {status}: "
                    f"{response.get('error', 'unknown error')}"
                )
            time.sleep(config.POLL_INTERVAL)
        raise RunPodError(f"Qwen job {job_id} did not complete within {config.POLL_TIMEOUT}s")

    def edit_image(
        self,
        image_url: str,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 28,
        seed: int = -1,
    ) -> str:
        """
        Convenience wrapper: submit an edit job and block until it completes.

        Returns the output image URL.  See ``submit_edit`` for parameter docs.
        """
        job_id = self.submit_edit(
            image_url=image_url,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        return self.wait_for_edit(job_id)
