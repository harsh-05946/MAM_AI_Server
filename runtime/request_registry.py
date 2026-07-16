"""Track request lifecycle for cancellation / disconnect awareness (Release 1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
import threading
import time
import uuid


class RequestCancelled(Exception):
    def __init__(self, reason: str = "cancelled"):
        self.reason = reason or "cancelled"
        super().__init__(self.reason)


class RequestState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class RegisteredRequest:
    server_request_id: str
    logical_request_id: str
    attempt_id: str
    endpoint: str
    model: Optional[str]
    state: RequestState = RequestState.QUEUED
    created_at: float = field(default_factory=time.perf_counter)
    deadline_monotonic: Optional[float] = None
    client_disconnected: bool = False
    cancel_reason: Optional[str] = None


class RequestRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, RegisteredRequest] = {}

    def register(
        self,
        *,
        logical_request_id: str,
        attempt_id: str = "",
        endpoint: str = "",
        model: Optional[str] = None,
        deadline_monotonic: Optional[float] = None,
        server_request_id: Optional[str] = None,
    ) -> RegisteredRequest:
        sid = server_request_id or str(uuid.uuid4())
        item = RegisteredRequest(
            server_request_id=sid,
            logical_request_id=logical_request_id,
            attempt_id=attempt_id or "",
            endpoint=endpoint,
            model=model,
            deadline_monotonic=deadline_monotonic,
        )
        with self._lock:
            self._items[sid] = item
        return item

    def get(self, server_request_id: str) -> Optional[RegisteredRequest]:
        with self._lock:
            return self._items.get(server_request_id)

    def mark(self, server_request_id: str, state: RequestState, reason: Optional[str] = None) -> None:
        with self._lock:
            item = self._items.get(server_request_id)
            if not item:
                return
            if item.state in (RequestState.COMPLETED, RequestState.FAILED, RequestState.CANCELLED):
                return
            item.state = state
            if reason:
                item.cancel_reason = reason

    def mark_disconnected(self, server_request_id: str) -> None:
        with self._lock:
            item = self._items.get(server_request_id)
            if not item:
                return
            item.client_disconnected = True
            if item.state == RequestState.QUEUED:
                item.state = RequestState.CANCELLED
                item.cancel_reason = "client_disconnected"

    def should_skip_gpu(self, server_request_id: Optional[str]) -> tuple[bool, str]:
        if not server_request_id:
            return False, ""
        with self._lock:
            item = self._items.get(server_request_id)
            if not item:
                return False, ""
            if item.state == RequestState.CANCELLED:
                return True, item.cancel_reason or "cancelled"
            if item.client_disconnected and item.state == RequestState.QUEUED:
                item.state = RequestState.CANCELLED
                item.cancel_reason = "client_disconnected"
                return True, "client_disconnected"
            if item.deadline_monotonic is not None and time.perf_counter() >= item.deadline_monotonic:
                if item.state == RequestState.QUEUED:
                    item.state = RequestState.CANCELLED
                    item.cancel_reason = "deadline_exceeded"
                    return True, "deadline_exceeded"
            return False, ""

    def unregister(self, server_request_id: str) -> None:
        with self._lock:
            self._items.pop(server_request_id, None)

    def cancel_all_queued(self, reason: str = "server_drain") -> int:
        n = 0
        with self._lock:
            for item in self._items.values():
                if item.state == RequestState.QUEUED:
                    item.state = RequestState.CANCELLED
                    item.cancel_reason = reason
                    n += 1
        return n

    def counts(self) -> dict[str, int]:
        with self._lock:
            out: dict[str, int] = {}
            for item in self._items.values():
                out[item.state.value] = out.get(item.state.value, 0) + 1
            return out


_REG: Optional[RequestRegistry] = None
_REG_LOCK = threading.Lock()


def get_request_registry() -> RequestRegistry:
    global _REG
    with _REG_LOCK:
        if _REG is None:
            _REG = RequestRegistry()
        return _REG


def reset_request_registry_for_tests() -> None:
    global _REG
    with _REG_LOCK:
        _REG = None
