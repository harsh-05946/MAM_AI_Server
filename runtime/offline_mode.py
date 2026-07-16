"""Shared offline-mode helpers.

`AI_OFFLINE_MODE` is the single operator-facing switch.
When enabled, we force local-only HF behavior by default.
"""

from __future__ import annotations

import os


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def is_offline_mode() -> bool:
    return (
        env_bool("AI_OFFLINE_MODE", False)
        or env_bool("HF_HUB_OFFLINE", False)
        or env_bool("TRANSFORMERS_OFFLINE", False)
    )


def apply_offline_env_defaults() -> None:
    """Apply derived defaults from AI_OFFLINE_MODE without clobbering explicit envs."""
    if not env_bool("AI_OFFLINE_MODE", False):
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("ALLOW_HF_FALLBACK", "0")
