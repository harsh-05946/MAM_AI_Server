"""HTTP client for local Triton Inference Server.

Uses httpx for readiness and JSON infer (Phase 3 Emotion). Binary protocol can be
added later; JSON is enough for ViT logits.
"""
from __future__ import annotations

from typing import Any, Optional
import threading
import time

import httpx
import numpy as np

from runtime.triton_flags import triton_grpc_url, triton_http_url

_DEFAULT_TIMEOUT = 2.0
_INFER_TIMEOUT = 60.0


class TritonClient:
    """Readiness + HTTP v2 infer against Triton's REST API."""

    def __init__(
        self,
        http_url: Optional[str] = None,
        grpc_url: Optional[str] = None,
        timeout_sec: float = _DEFAULT_TIMEOUT,
        infer_timeout_sec: float = _INFER_TIMEOUT,
    ) -> None:
        self.http_url = (http_url or triton_http_url()).rstrip("/")
        self.grpc_url = grpc_url or triton_grpc_url()
        self.timeout_sec = timeout_sec
        self.infer_timeout_sec = infer_timeout_sec
        self._lock = threading.Lock()
        self._cached_ready: Optional[bool] = None
        self._cached_at: float = 0.0
        self._cache_ttl_sec = 2.0

    def _get(self, path: str) -> tuple[Optional[httpx.Response], Optional[str]]:
        url = f"{self.http_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                resp = client.get(url)
                return resp, None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def is_live(self) -> bool:
        resp, _ = self._get("/v2/health/live")
        return bool(resp is not None and resp.status_code == 200)

    def is_ready(self, *, use_cache: bool = True) -> bool:
        now = time.monotonic()
        with self._lock:
            if (
                use_cache
                and self._cached_ready is not None
                and (now - self._cached_at) < self._cache_ttl_sec
            ):
                return self._cached_ready
        resp, _ = self._get("/v2/health/ready")
        ready = bool(resp is not None and resp.status_code == 200)
        with self._lock:
            self._cached_ready = ready
            self._cached_at = now
        return ready

    def server_metadata(self) -> dict[str, Any]:
        resp, err = self._get("/v2")
        if err or resp is None:
            return {"ok": False, "error": err or "no_response"}
        if resp.status_code != 200:
            return {"ok": False, "status_code": resp.status_code, "body": resp.text[:200]}
        try:
            return {"ok": True, "metadata": resp.json()}
        except Exception as exc:
            return {"ok": False, "error": f"json:{exc}"}

    def model_ready(self, model_name: str) -> bool:
        resp, _ = self._get(f"/v2/models/{model_name}/ready")
        return bool(resp is not None and resp.status_code == 200)

    def status(self) -> dict[str, Any]:
        live = self.is_live()
        ready = self.is_ready(use_cache=False) if live else False
        return {
            "http_url": self.http_url,
            "grpc_url": self.grpc_url,
            "live": live,
            "ready": ready,
        }

    def infer_tensors(
        self,
        model_name: str,
        inputs: list[tuple[str, np.ndarray, str]],
        output_name: str,
        *,
        model_version: str = "",
        output_dtype: Any = np.float32,
    ) -> np.ndarray:
        """HTTP JSON infer with multiple named inputs.

        `inputs` entries are (name, array, datatype) where datatype is e.g. FP32, INT64.
        Arrays must include the batch dimension.
        """
        payload_inputs = []
        for name, array, datatype in inputs:
            if datatype == "FP32" and array.dtype != np.float32:
                array = array.astype(np.float32, copy=False)
            elif datatype == "INT64" and array.dtype != np.int64:
                array = array.astype(np.int64, copy=False)
            if not array.flags["C_CONTIGUOUS"]:
                array = np.ascontiguousarray(array)
            payload_inputs.append(
                {
                    "name": name,
                    "shape": list(array.shape),
                    "datatype": datatype,
                    "data": array.reshape(-1).tolist(),
                }
            )
        payload = {"inputs": payload_inputs, "outputs": [{"name": output_name}]}
        ver = f"/versions/{model_version}" if model_version else ""
        url = f"{self.http_url}/v2/models/{model_name}{ver}/infer"
        with httpx.Client(timeout=self.infer_timeout_sec) as client:
            resp = client.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Triton infer failed model={model_name} status={resp.status_code}: {resp.text[:500]}"
            )
        body = resp.json()
        outputs = body.get("outputs") or []
        if not outputs:
            raise RuntimeError(f"Triton infer returned no outputs: {body}")
        out = outputs[0]
        shape = out.get("shape") or []
        data = out.get("data")
        if data is None:
            raise RuntimeError(f"Triton output missing data: {out.keys()}")
        return np.asarray(data, dtype=output_dtype).reshape(shape)

    def infer_fp32(
        self,
        model_name: str,
        input_name: str,
        output_name: str,
        array: np.ndarray,
        *,
        model_version: str = "",
    ) -> np.ndarray:
        """Run a single FP32 tensor through Triton HTTP JSON infer."""
        return self.infer_tensors(
            model_name,
            [(input_name, array, "FP32")],
            output_name,
            model_version=model_version,
            output_dtype=np.float32,
        )


_CLIENT: Optional[TritonClient] = None
_CLIENT_LOCK = threading.Lock()


def get_triton_client() -> TritonClient:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = TritonClient()
        return _CLIENT


def reset_triton_client_for_tests() -> None:
    global _CLIENT
    with _CLIENT_LOCK:
        _CLIENT = None
