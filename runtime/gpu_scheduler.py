from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import os
import threading
import time

from runtime.event_logger import emit
from runtime.gpu_memory import gpu_memory_snapshot, reset_peak_memory_stats
from runtime.model_registry import queue_class_for_model
from runtime.request_context import RequestContext, get_request_context
from runtime.stage_timer import StageTimer


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


WAIT_THRESHOLDS = (
    (300.0, "warning"),
    (600.0, "critical"),
    (840.0, "timeout_risk"),
)


@dataclass
class SchedulerStats:
    active_requests: int = 0
    waiting_for_admission: int = 0
    waiting_for_gpu: int = 0
    gpu_running: int = 0
    occupied_seconds: float = 0.0
    occupied_started: Optional[float] = None
    campaign_started: Optional[float] = None


class GpuScheduler:
    """Single CUDA execution slot with separate admission and GPU waits."""

    def __init__(self) -> None:
        self.max_waiting = _env_int("MAX_WAITING_GPU_REQUESTS", 16)
        self.reject_when_full = _env_bool("GPU_QUEUE_REJECT_WHEN_FULL", False)
        # One active + N waiting.
        self._admission = threading.Semaphore(self.max_waiting + 1)
        self._gpu = threading.Semaphore(1)
        self._lock = threading.Lock()
        self.stats = SchedulerStats()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            occupied = self.stats.occupied_seconds
            if self.stats.occupied_started is not None:
                occupied += time.perf_counter() - self.stats.occupied_started
            return {
                "active_requests": self.stats.active_requests,
                "waiting_for_admission": self.stats.waiting_for_admission,
                "waiting_for_gpu": self.stats.waiting_for_gpu,
                "gpu_running": self.stats.gpu_running,
                "occupied_seconds": round(occupied, 3),
                "max_waiting_gpu_requests": self.max_waiting,
                "reject_when_full": self.reject_when_full,
            }

    def mark_campaign_start(self) -> None:
        with self._lock:
            self.stats.occupied_seconds = 0.0
            self.stats.occupied_started = None
            self.stats.campaign_started = time.perf_counter()

    def try_acquire_admission(self) -> bool:
        return self._admission.acquire(blocking=False)

    def acquire_admission_nowait(self, ctx: Optional[RequestContext] = None, timer: Optional[StageTimer] = None) -> bool:
        """Non-blocking admission for reject-when-full mode."""
        ctx = ctx or get_request_context()
        timer = timer or (ctx.timer if ctx else None)
        if timer and timer.admission_entered <= 0:
            timer.admission_entered = StageTimer.now()
        if not self._admission.acquire(blocking=False):
            return False
        with self._lock:
            self.stats.active_requests += 1
        if timer:
            timer.admission_acquired = StageTimer.now()
        return True

    def _emit_wait_threshold(self, ctx: Optional[RequestContext], waited: float) -> None:
        if ctx is None:
            return
        for threshold, severity in WAIT_THRESHOLDS:
            key = f"{severity}:{int(threshold)}"
            if waited >= threshold and key not in ctx.wait_thresholds_emitted:
                ctx.wait_thresholds_emitted.add(key)
                emit(
                    "request_wait_threshold_exceeded",
                    request_id=ctx.request_id,
                    endpoint=ctx.endpoint,
                    total_wait_seconds=round(waited, 3),
                    severity=severity,
                    threshold_seconds=threshold,
                )

    def acquire_admission(self, ctx: Optional[RequestContext] = None, timer: Optional[StageTimer] = None) -> None:
        ctx = ctx or get_request_context()
        timer = timer or (ctx.timer if ctx else None)
        if timer and timer.admission_entered <= 0:
            timer.admission_entered = StageTimer.now()
        with self._lock:
            self.stats.waiting_for_admission += 1
            self.stats.active_requests += 1
        start = time.perf_counter()
        try:
            while True:
                if self._admission.acquire(timeout=1.0):
                    break
                self._emit_wait_threshold(ctx, time.perf_counter() - start)
            if timer:
                timer.admission_acquired = StageTimer.now()
        finally:
            with self._lock:
                self.stats.waiting_for_admission = max(0, self.stats.waiting_for_admission - 1)

    def release_admission(self) -> None:
        self._admission.release()
        with self._lock:
            self.stats.active_requests = max(0, self.stats.active_requests - 1)

    @contextmanager
    def run(
        self,
        *,
        model: str,
        batch_size: int = 1,
        ctx: Optional[RequestContext] = None,
        use_cuda_events: bool = True,
    ):
        """Serialize CUDA work and emit queue/inference timing events."""
        import torch

        ctx = ctx or get_request_context()
        timer = ctx.timer if ctx else StageTimer()
        queue_class = queue_class_for_model(model)
        if timer.gpu_queue_entered <= 0:
            timer.gpu_queue_entered = StageTimer.now()

        with self._lock:
            self.stats.waiting_for_gpu += 1
        emit(
            "gpu_queue_entered",
            model=model,
            queue_depth=self.snapshot()["waiting_for_gpu"],
            queue_class=queue_class,
            batch_size=batch_size,
            **self.snapshot(),
        )

        start_wait = time.perf_counter()
        while True:
            if self._gpu.acquire(timeout=1.0):
                break
            self._emit_wait_threshold(ctx, time.perf_counter() - start_wait + (timer.admission_wait_seconds or 0))

        with self._lock:
            self.stats.waiting_for_gpu = max(0, self.stats.waiting_for_gpu - 1)
            self.stats.gpu_running = 1
            self.stats.occupied_started = time.perf_counter()

        timer.gpu_started = StageTimer.now()
        reset_peak_memory_stats()
        mem_before = gpu_memory_snapshot()
        emit(
            "gpu_inference_started",
            model=model,
            batch_size=batch_size,
            queue_wait_seconds=round(timer.gpu_queue_wait_seconds or 0.0, 6),
            queue_class=queue_class,
            **mem_before,
            **self.snapshot(),
        )

        start_event = end_event = None
        if use_cuda_events and torch.cuda.is_available() and model != "insightface":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        wall_start = time.perf_counter()
        try:
            yield
        finally:
            inference_seconds = time.perf_counter() - wall_start
            if end_event is not None and start_event is not None:
                end_event.record()
                end_event.synchronize()
                inference_seconds = start_event.elapsed_time(end_event) / 1000.0
            timer.gpu_finished = StageTimer.now()
            mem_after = gpu_memory_snapshot()
            emit(
                "gpu_inference_completed",
                model=model,
                batch_size=batch_size,
                inference_seconds=round(inference_seconds, 6),
                seconds_per_item=round(inference_seconds / max(batch_size, 1), 6),
                items_per_second=round(max(batch_size, 1) / max(inference_seconds, 1e-9), 6),
                queue_class=queue_class,
                **mem_after,
                **self.snapshot(),
            )
            with self._lock:
                if self.stats.occupied_started is not None:
                    self.stats.occupied_seconds += time.perf_counter() - self.stats.occupied_started
                    self.stats.occupied_started = None
                self.stats.gpu_running = 0
            self._gpu.release()


_SCHEDULER: Optional[GpuScheduler] = None
_SCHEDULER_LOCK = threading.Lock()


def get_gpu_scheduler() -> GpuScheduler:
    global _SCHEDULER
    with _SCHEDULER_LOCK:
        if _SCHEDULER is None:
            _SCHEDULER = GpuScheduler()
        return _SCHEDULER
