"""Cached AI Server readiness state (Release 1).

/ready reads this cache — never runs Face/Scene/Qwen inference in the probe path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import os
import threading
import time

from runtime.event_logger import emit


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_model_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [p.strip() for p in raw.split(",") if p.strip()]


# Map config names → keys in MODELS / Triton flags.
_MODEL_KEY_ALIASES = {
    "face": "face",
    "face_detector": "face",
    "face_recognizer": "face",
    "emotion": "emotion",
    "scene": "scene",
    "ram_plus": "ram_plus",
    "ram": "ram_plus",
    "embeddings": "embed",
    "embed": "embed",
    "qwen": "qwen_vl",
    "qwen_vl": "qwen_vl",
    "sarvam": "sarvam",
    "sarvam_translation": "sarvam",
    "image_enhancement": "image_enhancement",
}


@dataclass
class ReadinessSnapshot:
    accepting_requests: bool = False
    draining: bool = False
    startup_complete: bool = False
    warmed: bool = False
    gpu_ready: bool = False
    models_ready: bool = False
    triton_ready: bool = True
    scheduler_ready: bool = True
    service_state: str = "STARTING"  # STARTING|READY|DEGRADED|DRAINING|NOT_READY
    reason: Optional[str] = "starting"
    required_missing: list[str] = field(default_factory=list)
    optional_missing: list[str] = field(default_factory=list)
    updated_at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        ready = (
            self.accepting_requests
            and not self.draining
            and self.startup_complete
            and self.warmed
            and self.gpu_ready
            and self.models_ready
            and self.triton_ready
            and self.scheduler_ready
        )
        return {
            "status": "ready" if ready else "not_ready",
            "accepting_requests": self.accepting_requests and not self.draining and ready,
            "gpu_ready": self.gpu_ready,
            "models_ready": self.models_ready,
            "triton_ready": self.triton_ready,
            "scheduler_ready": self.scheduler_ready,
            "draining": self.draining,
            "startup_complete": self.startup_complete,
            "warmed": self.warmed,
            "service_state": self.service_state,
            "reason": None if ready else (self.reason or "not_ready"),
            "required_missing": list(self.required_missing),
            "optional_missing": list(self.optional_missing),
            "updated_at": self.updated_at,
        }


class ReadinessManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snap = ReadinessSnapshot()
        self.required = _parse_model_list(
            "AI_REQUIRED_MODELS",
            "face,emotion,scene,ram_plus,embeddings",
        )
        self.optional = _parse_model_list(
            "AI_OPTIONAL_MODELS",
            "qwen,sarvam_translation",
        )
        self.canary_interval = _env_int("AI_MODEL_CANARY_INTERVAL_SECONDS", 300)
        self._canary_thread: Optional[threading.Thread] = None
        self._stop_canary = threading.Event()
        self._fatal_error: Optional[str] = None

    def snapshot(self) -> ReadinessSnapshot:
        with self._lock:
            return ReadinessSnapshot(**{**self._snap.__dict__})

    def begin_drain(self) -> None:
        with self._lock:
            self._snap.draining = True
            self._snap.accepting_requests = False
            self._snap.service_state = "DRAINING"
            self._snap.reason = "draining"
            self._snap.updated_at = time.time()
        emit("server_drain_started")

    def set_fatal(self, reason: str) -> None:
        with self._lock:
            self._fatal_error = reason
            self._snap.accepting_requests = False
            self._snap.service_state = "NOT_READY"
            self._snap.reason = reason
            self._snap.updated_at = time.time()

    def mark_startup_loaded(self) -> None:
        with self._lock:
            self._snap.startup_complete = True
            self._snap.updated_at = time.time()

    def mark_warmed(self) -> None:
        with self._lock:
            self._snap.warmed = True
            self._snap.updated_at = time.time()

    def open_for_traffic(self) -> None:
        with self._lock:
            if self._snap.draining or self._fatal_error:
                return
            self._snap.accepting_requests = True
            self._refresh_locked()
            # Do not claim READY if refresh closed traffic (e.g. Triton down).
            if (
                self._snap.accepting_requests
                and self._snap.models_ready
                and self._snap.gpu_ready
                and self._snap.warmed
                and self._snap.triton_ready
            ):
                self._snap.service_state = (
                    "DEGRADED" if self._snap.optional_missing else "READY"
                )
                self._snap.reason = None
            emit(
                "server_accepting_requests",
                service_state=self._snap.service_state,
                reason=self._snap.reason,
                accepting_requests=self._snap.accepting_requests,
                triton_ready=self._snap.triton_ready,
                required_missing=list(self._snap.required_missing),
                optional_missing=list(self._snap.optional_missing),
            )

    def refresh(self, loaded_models: Optional[set[str]] = None) -> ReadinessSnapshot:
        with self._lock:
            if loaded_models is not None:
                self._last_loaded = set(loaded_models)
            self._refresh_locked()
            return self.snapshot()

    def _resolve_key(self, name: str) -> str:
        return _MODEL_KEY_ALIASES.get(name.strip().lower(), name.strip().lower())

    def _refresh_locked(self) -> None:
        loaded = getattr(self, "_last_loaded", set())
        required_keys = [self._resolve_key(m) for m in self.required]
        optional_keys = [self._resolve_key(m) for m in self.optional]
        # image_enhancement is never loaded today — ignore if listed
        optional_keys = [k for k in optional_keys if k != "image_enhancement"]

        self._snap.required_missing = [k for k in required_keys if k not in loaded]
        self._snap.optional_missing = [k for k in optional_keys if k not in loaded]
        self._snap.models_ready = len(self._snap.required_missing) == 0

        try:
            import torch

            self._snap.gpu_ready = bool(torch.cuda.is_available())
        except Exception:
            self._snap.gpu_ready = False

        self._snap.triton_ready = self._check_triton_locked()
        self._snap.scheduler_ready = not self._snap.draining

        if self._fatal_error:
            self._snap.service_state = "NOT_READY"
            self._snap.reason = self._fatal_error
            self._snap.accepting_requests = False
        elif self._snap.draining:
            self._snap.service_state = "DRAINING"
            self._snap.reason = "draining"
            self._snap.accepting_requests = False
        elif not self._snap.startup_complete:
            self._snap.service_state = "STARTING"
            self._snap.reason = "starting"
            self._snap.accepting_requests = False
        elif not self._snap.warmed:
            self._snap.service_state = "STARTING"
            self._snap.reason = "warming"
            self._snap.accepting_requests = False
        elif not self._snap.gpu_ready:
            self._snap.service_state = "NOT_READY"
            self._snap.reason = "gpu_unavailable"
            self._snap.accepting_requests = False
        elif not self._snap.models_ready:
            self._snap.service_state = "NOT_READY"
            self._snap.reason = "required_models_missing"
            self._snap.accepting_requests = False
        elif not self._snap.triton_ready:
            self._snap.service_state = "NOT_READY"
            self._snap.reason = "triton_not_ready"
            self._snap.accepting_requests = False
        else:
            # Keep accepting if already opened
            if self._snap.accepting_requests:
                self._snap.service_state = (
                    "DEGRADED" if self._snap.optional_missing else "READY"
                )
                self._snap.reason = None
            else:
                self._snap.service_state = "STARTING"
                self._snap.reason = "not_opened"

        self._snap.updated_at = time.time()

    def _check_triton_locked(self) -> bool:
        # If no Triton flags enabled, Triton is not required for readiness.
        try:
            from runtime.triton_flags import any_triton_flag_enabled
            from runtime.triton_client import get_triton_client
            from runtime.triton_router import is_migrated, TRITON_MODEL_NAMES
            from runtime.triton_flags import triton_flag_for_model

            if not any_triton_flag_enabled():
                return True
            client = get_triton_client()
            if not client.is_live():
                return False
            # Every model with flag on and migrated must be ready.
            for model_key, triton_name in TRITON_MODEL_NAMES.items():
                if not triton_flag_for_model(model_key):
                    continue
                if not is_migrated(model_key):
                    continue
                if not client.model_ready(triton_name):
                    return False
            return True
        except Exception:
            return False

    def start_canary(self, loaded_models_fn) -> None:
        """Periodic lightweight readiness refresh (no heavy inference)."""
        if self._canary_thread and self._canary_thread.is_alive():
            return

        def _loop():
            while not self._stop_canary.wait(self.canary_interval):
                try:
                    loaded = set(loaded_models_fn() or [])
                    self.refresh(loaded)
                except Exception as exc:
                    emit("readiness_canary_failed", error=str(exc))

        self._canary_thread = threading.Thread(target=_loop, name="readiness-canary", daemon=True)
        self._canary_thread.start()

    def stop_canary(self) -> None:
        self._stop_canary.set()


_READY: Optional[ReadinessManager] = None
_READY_LOCK = threading.Lock()


def get_readiness_manager() -> ReadinessManager:
    global _READY
    with _READY_LOCK:
        if _READY is None:
            _READY = ReadinessManager()
        return _READY


def reset_readiness_manager_for_tests() -> None:
    global _READY
    with _READY_LOCK:
        _READY = None
