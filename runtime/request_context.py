from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import contextvars
import os
import threading
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
        )

    def to_event_fields(self) -> dict[str, Any]:
        return {
            "server_run_id": self.server_run_id,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "media_id": self.media_id,
            "request_id": self.request_id,
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
