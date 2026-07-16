"""Per-model Triton feature flags (Phase 2: all default false)."""
from __future__ import annotations

import os
from typing import Any


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Visual models only. Qwen / Sarvam stay native until a later phase.
TRITON_FLAG_ENV = {
    "emotion": "USE_TRITON_EMOTION",
    "ram_plus": "USE_TRITON_RAM",
    "scene": "USE_TRITON_SCENE",
    "embed": "USE_TRITON_EMBED",
    "insightface": "USE_TRITON_FACE",
}


def triton_flag_for_model(model: str) -> bool:
    env_name = TRITON_FLAG_ENV.get(model)
    if not env_name:
        return False
    return _env_bool(env_name, False)


def any_triton_flag_enabled() -> bool:
    return any(triton_flag_for_model(m) for m in TRITON_FLAG_ENV)


def triton_flags_snapshot() -> dict[str, Any]:
    return {
        env_name: triton_flag_for_model(model)
        for model, env_name in TRITON_FLAG_ENV.items()
    }


def triton_http_url() -> str:
    return os.getenv("TRITON_HTTP_URL", "http://127.0.0.1:8001").rstrip("/")


def triton_grpc_url() -> str:
    return os.getenv("TRITON_GRPC_URL", "127.0.0.1:8002")
