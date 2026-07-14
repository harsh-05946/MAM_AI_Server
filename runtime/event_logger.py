from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import json
import os
import queue
import threading
import time

from runtime.paths import CURRENT_DIR, EVENTS_JSONL, ensure_runtime_dirs
from runtime.request_context import get_request_context


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _run_ids() -> dict[str, Any]:
    try:
        from runtime.run_state import get_run_state

        state = get_run_state()
        return {
            "server_run_id": state.server_run_id,
            "run_id": state.active_run_id(),
        }
    except Exception:
        return {}


class EventLogger:
    def __init__(self, path: Path = EVENTS_JSONL):
        ensure_runtime_dirs()
        self.path = path
        self._q: queue.Queue[Optional[dict[str, Any]]] = queue.Queue(maxsize=10000)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer_loop, name="event-logger", daemon=True)
        self._started = False
        self._drop_count = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            ensure_runtime_dirs()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self.path.touch()
            self._thread.start()
            self._started = True

    def stop(self, timeout: float = 5.0) -> None:
        if not self._started:
            return
        self._q.put(None)
        self._thread.join(timeout=timeout)
        self._started = False

    def emit(self, event: str, **fields: Any) -> None:
        if not self._started:
            self.start()
        payload: dict[str, Any] = {
            "timestamp": _iso_now(),
            "event": event,
            "process_id": os.getpid(),
            "thread_id": threading.get_ident(),
            "gpu_id": 0,
            "instance": os.getenv("INSTANCE_NAME", "main"),
        }
        payload.update(_run_ids())
        ctx = get_request_context()
        if ctx is not None:
            payload.update(ctx.to_event_fields())
        for k, v in fields.items():
            if v is not None:
                payload[k] = v
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            self._drop_count += 1

    def flush(self, timeout: float = 2.0) -> None:
        deadline = time.time() + timeout
        while not self._q.empty() and time.time() < deadline:
            time.sleep(0.01)

    def _writer_loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
            except Exception:
                # Never break inference because logging failed.
                pass


_LOGGER: Optional[EventLogger] = None
_LOGGER_LOCK = threading.Lock()


def get_event_logger() -> EventLogger:
    global _LOGGER
    with _LOGGER_LOCK:
        if _LOGGER is None:
            _LOGGER = EventLogger()
            _LOGGER.start()
        return _LOGGER


def emit(event: str, **fields: Any) -> None:
    get_event_logger().emit(event, **fields)
