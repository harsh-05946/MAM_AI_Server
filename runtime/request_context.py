from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import contextvars
import os
import threading
import time
import uuid

from runtime.stage_timer import StageTimer

_request_ctx: contextvars.ContextVar[Optional["RequestContext"]] = contextvars.ContextVar(
    "request_ctx", default=None
)


@dataclass
class RequestContext:
    server_run_id: str
    run_id: str
    job_id: str
    media_id: str
    request_id: str
    endpoint: str
    http_method: str
    requested_items: Optional[int] = None
    content_length_bytes: Optional[int] = None
    model: Optional[str] = None
    execution_mode: Optional[str] = None
    timer: StageTimer = field(default_factory=StageTimer)
    wait_thresholds_emitted: set[str] = field(default_factory=set)
    # Release 1 correlation / deadline
    attempt_id: str = ""
    server_request_id: str = ""
    deadline_seconds: Optional[float] = None
    deadline_monotonic: Optional[float] = None

    def snapshot(self) -> "RequestContext":
        """Copy safe for executor threads."""
        return RequestContext(
            server_run_id=self.server_run_id,
            run_id=self.run_id,
            job_id=self.job_id,
            media_id=self.media_id,
            request_id=self.request_id,
            endpoint=self.endpoint,
            http_method=self.http_method,
            requested_items=self.requested_items,
            content_length_bytes=self.content_length_bytes,
            model=self.model,
            execution_mode=self.execution_mode,
            timer=self.timer,
            wait_thresholds_emitted=set(self.wait_thresholds_emitted),
            attempt_id=self.attempt_id,
            server_request_id=self.server_request_id,
            deadline_seconds=self.deadline_seconds,
            deadline_monotonic=self.deadline_monotonic,
        )

    def to_event_fields(self) -> dict[str, Any]:
        return {
            "server_run_id": self.server_run_id,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "media_id": self.media_id,
            "request_id": self.request_id,
            "attempt_id": self.attempt_id,
            "server_request_id": self.server_request_id,
            "endpoint": self.endpoint,
            "http_method": self.http_method,
            "requested_items": self.requested_items,
            "content_length_bytes": self.content_length_bytes,
            "model": self.model,
            "execution_mode": self.execution_mode,
            "process_id": os.getpid(),
            "thread_id": threading.get_ident(),
            "gpu_id": 0,
        }


def get_request_context() -> Optional[RequestContext]:
    return _request_ctx.get()


def set_request_context(ctx: Optional[RequestContext]):
    return _request_ctx.set(ctx)


def reset_request_context(token) -> None:
    _request_ctx.reset(token)


def new_request_id(header_value: Optional[str] = None) -> str:
    return (header_value or "").strip() or str(uuid.uuid4())


def new_attempt_id(header_value: Optional[str] = None) -> str:
    return (header_value or "").strip() or ""


def parse_deadline_headers(headers) -> tuple[Optional[float], Optional[float]]:
    """Return (deadline_seconds, deadline_monotonic) from request headers."""
    raw_remaining = (headers.get("x-deadline-seconds") or headers.get("x-request-timeout-seconds") or "").strip()
    raw_epoch = (headers.get("x-deadline-epoch") or "").strip()
    now = time.time()
    mono = time.perf_counter()
    candidates: list[float] = []
    if raw_remaining:
        try:
            sec = float(raw_remaining)
            if sec > 0:
                candidates.append(mono + sec)
        except ValueError:
            pass
    if raw_epoch:
        try:
            epoch = float(raw_epoch)
            remaining = epoch - now
            if remaining > 0:
                candidates.append(mono + remaining)
        except ValueError:
            pass
    if not candidates:
        return None, None
    deadline_mono = min(candidates)
    return max(0.0, deadline_mono - mono), deadline_mono
