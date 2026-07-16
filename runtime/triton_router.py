"""Route visual model calls to native FastAPI or Triton.

Phase 3: Emotion is migrated. Flip USE_TRITON_EMOTION=true once Triton serves it.
Other models remain pending until their migration.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from runtime.triton_client import TritonClient, get_triton_client
from runtime.triton_flags import (
    any_triton_flag_enabled,
    triton_flag_for_model,
    triton_flags_snapshot,
    triton_grpc_url,
    triton_http_url,
)

Backend = Literal["native", "triton", "triton_pending"]

TRITON_MODEL_NAMES = {
    "emotion": "emotion",
    "ram_plus": "ram_plus",
    "scene": "scene",
    "embed": "embed",
    "insightface": "insightface",
}

# Models with wired FastAPI → Triton inference helpers that passed / are ready for parity.
_MIGRATED: set[str] = {"emotion", "embed"}


def is_migrated(model: str) -> bool:
    return model in _MIGRATED


def resolve_backend(model: str, client: Optional[TritonClient] = None) -> Backend:
    """Decide execution backend for a model key."""
    if not triton_flag_for_model(model):
        return "native"
    if not is_migrated(model):
        return "triton_pending"
    c = client or get_triton_client()
    if not c.is_ready() or not c.model_ready(TRITON_MODEL_NAMES[model]):
        return "triton_pending"
    return "triton"


def require_native_or_raise(model: str, client: Optional[TritonClient] = None) -> None:
    """Legacy guard for models not yet routed. Prefer resolve_backend + branch."""
    backend = resolve_backend(model, client=client)
    if backend == "native":
        return
    if backend == "triton_pending":
        raise RuntimeError(
            f"USE_TRITON enabled for '{model}' but Triton path is not ready "
            f"(server down, model not loaded, or not migrated). Check scripts/status_triton.sh."
        )
    raise RuntimeError(
        f"Triton backend selected for '{model}'; use the Triton-aware process path instead of require_native_or_raise."
    )


def triton_runtime_status(client: Optional[TritonClient] = None) -> dict[str, Any]:
    c = client or get_triton_client()
    probe = c.status()
    backends = {m: resolve_backend(m, client=c) for m in TRITON_MODEL_NAMES}
    emotion_ready = False
    ram_ready = False
    embed_ready = False
    try:
        emotion_ready = bool(probe["live"] and c.model_ready("emotion"))
    except Exception:
        emotion_ready = False
    try:
        ram_ready = bool(probe["live"] and c.model_ready("ram_plus"))
    except Exception:
        ram_ready = False
    try:
        embed_ready = bool(probe["live"] and c.model_ready("embed"))
    except Exception:
        embed_ready = False
    return {
        "phase": 3,
        "any_flag_enabled": any_triton_flag_enabled(),
        "flags": triton_flags_snapshot(),
        "http_url": triton_http_url(),
        "grpc_url": triton_grpc_url(),
        "live": probe["live"],
        "ready": probe["ready"],
        "emotion_model_ready": emotion_ready,
        "ram_plus_model_ready": ram_ready,
        "embed_model_ready": embed_ready,
        "backends": backends,
        "migrated_models": sorted(_MIGRATED),
        "note": (
            "Emotion + Embed Triton paths ready. USE_TRITON_* controls traffic; "
            "RAM++ ONNX deferred (tag parity)."
        ),
    }
