"""Model-aware admission + VISUAL/QWEN/SARVAM execution (l40s-six-lane-v1)."""
from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple
import os
import threading
import time

from runtime.capacity_profile import (
    PUBLIC_TO_INTERNAL,
    capacity_limits,
    exec_class_for_model,
    execution_bounds,
    model_fleet_concurrent,
    public_model_name,
)
from runtime.event_logger import emit
from runtime.gpu_memory import gpu_memory_snapshot, reset_peak_memory_stats
from runtime.model_registry import queue_class_for_model
from runtime.request_context import RequestContext, get_request_context
from runtime.request_registry import RequestCancelled, RequestState, get_request_registry
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


@dataclass
class _Waiter:
    event: threading.Event
    exec_class: str
    model: str
    enqueued_at: float = field(default_factory=time.perf_counter)


@dataclass
class LaneMetrics:
    queue_wait_sum: float = 0.0
    queue_wait_count: int = 0
    inference_sum: float = 0.0
    inference_count: int = 0
    items_sum: int = 0
    effective_batch_sum: float = 0.0
    rejected: int = 0
    overlap_qwen_sarvam_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        qw = self.queue_wait_sum / max(self.queue_wait_count, 1) if self.queue_wait_count else None
        inf = self.inference_sum / max(self.inference_count, 1) if self.inference_count else None
        items = self.items_sum
        ips = (items / self.inference_sum) if self.inference_sum > 0 else None
        eff = (
            self.effective_batch_sum / max(self.inference_count, 1)
            if self.inference_count
            else None
        )
        return {
            "rejected": self.rejected,
            "avg_queue_wait_seconds": round(qw, 6) if qw is not None else None,
            "avg_inference_seconds": round(inf, 6) if inf is not None else None,
            "items_processed": items,
            "items_per_second": round(ips, 6) if ips is not None else None,
            "avg_effective_batch": round(eff, 3) if eff is not None else None,
            "qwen_sarvam_overlap_seconds": round(self.overlap_qwen_sarvam_seconds, 3),
        }


class GpuScheduler:
    """Global + per-model admission; model-aware VISUAL/QWEN/SARVAM execution."""

    def __init__(self) -> None:
        limits = capacity_limits()
        exec_b = execution_bounds()
        # Accepted queue is independent of Processing fleet_global.
        self.max_accepted = int(exec_b["accepted_queue_limit"])
        self.max_waiting = self.max_accepted
        self.reject_when_full = True
        if os.getenv("AI_INTERNAL_REJECT_WHEN_FULL") is not None:
            self.reject_when_full = _env_bool("AI_INTERNAL_REJECT_WHEN_FULL", True)
        elif os.getenv("GPU_QUEUE_REJECT_WHEN_FULL") is not None:
            self.reject_when_full = _env_bool("GPU_QUEUE_REJECT_WHEN_FULL", True)

        self._global_admission = threading.Semaphore(self.max_accepted)
        self._model_sems: Dict[str, threading.Semaphore] = {}
        self._model_active: Dict[str, int] = defaultdict(int)
        self._model_waiting: Dict[str, int] = defaultdict(int)
        for public, internal in PUBLIC_TO_INTERNAL.items():
            # Scalable: fleet_model (= PER_MEDIA_MODEL * MAX_PARALLEL_MEDIA).
            # Six-lane: typically 1.
            n = model_fleet_concurrent(public)
            self._model_sems[internal] = threading.Semaphore(max(1, n))

        self.combined_generative = bool(exec_b["combined_generative_limit_enabled"])
        self.max_active_qwen = max(1, int(exec_b["max_active_qwen"]))
        self.max_active_sarvam = max(1, int(exec_b["max_active_sarvam"]))
        self.max_active_generative = max(1, _env_int("AI_MAX_ACTIVE_GENERATIVE_REQUESTS", 1))
        self._generative_sem = threading.Semaphore(self.max_active_generative)
        self._generative_active = 0

        self._slots = {
            "visual": max(1, int(exec_b["gpu_slots"]["visual"])),
            "qwen": max(1, int(exec_b["gpu_slots"]["qwen"])),
            "sarvam": max(1, int(exec_b["gpu_slots"]["sarvam"])),
        }
        self.visual_qwen_overlap = bool(exec_b["visual_qwen_overlap"])
        self.visual_sarvam_overlap = bool(exec_b["visual_sarvam_overlap"])
        self.qwen_sarvam_overlap = bool(exec_b["qwen_sarvam_overlap"])

        self._busy = {"visual": 0, "qwen": 0, "sarvam": 0}
        self._waiters: Dict[str, Deque[_Waiter]] = {
            "visual": deque(),
            "qwen": deque(),
            "sarvam": deque(),
        }
        self._overlap_started: Optional[float] = None

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self.stats = SchedulerStats()
        self._metrics: Dict[str, LaneMetrics] = defaultdict(LaneMetrics)
        self._rejected_by_model: Dict[str, int] = defaultdict(int)

        self.fairness_enabled = _env_bool("GPU_FAIRNESS", True)
        self.visual_share = 0.6

        overload = limits["overload"]
        self.model_retry_after = int(overload["model_retry_after_seconds"])
        self.server_retry_after = int(overload["server_retry_after_seconds"])

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            occupied = self.stats.occupied_seconds
            if self.stats.occupied_started is not None:
                occupied += time.perf_counter() - self.stats.occupied_started
            active_by_model = {
                public_model_name(k) or k: v for k, v in self._model_active.items() if v
            }
            waiting_by_model = {
                public_model_name(k) or k: v for k, v in self._model_waiting.items() if v
            }
            waiting_by_class = {k: len(v) for k, v in self._waiters.items()}
            lane_metrics = {
                (public_model_name(k) or k): m.as_dict() for k, m in self._metrics.items()
            }
            return {
                "profile": capacity_limits()["profile"],
                "active_requests": self.stats.active_requests,
                "waiting_for_admission": self.stats.waiting_for_admission,
                "waiting_for_gpu": self.stats.waiting_for_gpu,
                "gpu_running": self.stats.gpu_running,
                "occupied_seconds": round(occupied, 3),
                "max_waiting_gpu_requests": self.max_accepted,
                "ai_internal_max_waiting_requests": self.max_accepted,
                "ai_max_accepted_inference_requests": self.max_accepted,
                "accepted_queue_limit": self.max_accepted,
                "reject_when_full": self.reject_when_full,
                "max_active_generative_requests": self.max_active_generative,
                "active_generative_requests": self._generative_active,
                "combined_generative_limit_enabled": self.combined_generative,
                "max_active_qwen": self.max_active_qwen,
                "max_active_sarvam": self.max_active_sarvam,
                "gpu_fairness": self.fairness_enabled,
                "gpu_visual_share": self.visual_share,
                "gpu_waiting_by_lane": waiting_by_class,
                "gpu_served_seconds": {},
                "active_by_model": active_by_model,
                "waiting_by_model": waiting_by_model,
                "rejected_by_model": dict(self._rejected_by_model),
                "exec_busy": dict(self._busy),
                "exec_slots": dict(self._slots),
                "overlap": {
                    "visual_qwen": self.visual_qwen_overlap,
                    "visual_sarvam": self.visual_sarvam_overlap,
                    "qwen_sarvam": self.qwen_sarvam_overlap,
                },
                "lane_metrics": lane_metrics,
            }

    def mark_campaign_start(self) -> None:
        with self._lock:
            self.stats.occupied_seconds = 0.0
            self.stats.occupied_started = None
            self.stats.campaign_started = time.perf_counter()
            self._metrics.clear()
            self._rejected_by_model.clear()

    def _normalize_model(self, model: Optional[str]) -> str:
        if not model:
            return "unknown"
        if model in PUBLIC_TO_INTERNAL:
            return PUBLIC_TO_INTERNAL[model]
        return model

    def try_acquire_admission(self) -> bool:
        """Legacy global-only try (no model). Prefer try_admit_nowait."""
        return self._global_admission.acquire(blocking=False)

    def try_admit_nowait(
        self,
        model: Optional[str],
        ctx: Optional[RequestContext] = None,
        timer: Optional[StageTimer] = None,
    ) -> Tuple[bool, Optional[str], str]:
        """Try global + per-model admission.

        Returns (ok, public_model_or_None, code).
        code is '' on success, else 'MODEL_CAPACITY_FULL' or 'SERVER_CAPACITY_FULL'.
        """
        ctx = ctx or get_request_context()
        timer = timer or (ctx.timer if ctx else None)
        if timer and timer.admission_entered <= 0:
            timer.admission_entered = StageTimer.now()
        internal = self._normalize_model(model or (ctx.model if ctx else None))
        public = public_model_name(internal) or internal

        if not self._global_admission.acquire(blocking=False):
            with self._lock:
                self._rejected_by_model[public] = self._rejected_by_model.get(public, 0) + 1
                self._metrics[internal].rejected += 1
            return False, public, "SERVER_CAPACITY_FULL"

        model_sem = self._model_sems.get(internal)
        if model_sem is not None and not model_sem.acquire(blocking=False):
            self._global_admission.release()
            with self._lock:
                self._rejected_by_model[public] = self._rejected_by_model.get(public, 0) + 1
                self._metrics[internal].rejected += 1
            return False, public, "MODEL_CAPACITY_FULL"

        # Optional legacy combined generative gate.
        if self.combined_generative and internal in ("qwen_vl", "sarvam"):
            if not self._generative_sem.acquire(blocking=False):
                if model_sem is not None:
                    model_sem.release()
                self._global_admission.release()
                with self._lock:
                    self._rejected_by_model[public] = self._rejected_by_model.get(public, 0) + 1
                    self._metrics[internal].rejected += 1
                return False, public, "MODEL_CAPACITY_FULL"
            with self._lock:
                self._generative_active += 1

        with self._lock:
            self.stats.active_requests += 1
            self._model_active[internal] += 1
        if timer:
            timer.admission_acquired = StageTimer.now()
        if ctx is not None:
            ctx.execution_mode = f"admitted:{public}"
        return True, None, ""

    def acquire_admission_nowait(
        self, ctx: Optional[RequestContext] = None, timer: Optional[StageTimer] = None
    ) -> bool:
        ok, _, _ = self.try_admit_nowait(ctx.model if ctx else None, ctx=ctx, timer=timer)
        return ok

    def release_admission(self, model: Optional[str] = None, ctx: Optional[RequestContext] = None) -> None:
        ctx = ctx or get_request_context()
        internal = self._normalize_model(model or (ctx.model if ctx else None))
        model_sem = self._model_sems.get(internal)
        if model_sem is not None:
            model_sem.release()
        if self.combined_generative and internal in ("qwen_vl", "sarvam"):
            with self._lock:
                self._generative_active = max(0, self._generative_active - 1)
            self._generative_sem.release()
        self._global_admission.release()
        with self._lock:
            self.stats.active_requests = max(0, self.stats.active_requests - 1)
            if self._model_active[internal] > 0:
                self._model_active[internal] -= 1

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

    def _overlap_ok(self, a: str, b: str) -> bool:
        if a == b:
            return True
        pair = frozenset({a, b})
        if pair == frozenset({"qwen", "sarvam"}):
            return self.qwen_sarvam_overlap
        if pair == frozenset({"visual", "qwen"}):
            return self.visual_qwen_overlap
        if pair == frozenset({"visual", "sarvam"}):
            return self.visual_sarvam_overlap
        return False

    def _can_start_locked(self, exec_class: str) -> bool:
        if self._busy[exec_class] >= self._slots[exec_class]:
            return False
        for other, n in self._busy.items():
            if n <= 0 or other == exec_class:
                continue
            if not self._overlap_ok(exec_class, other):
                return False
        return True

    def _dispatch_locked(self) -> None:
        # Prefer starting any runnable waiter across classes.
        progressed = True
        while progressed:
            progressed = False
            for exec_class in ("visual", "qwen", "sarvam"):
                if not self._waiters[exec_class]:
                    continue
                if not self._can_start_locked(exec_class):
                    continue
                waiter = self._waiters[exec_class].popleft()
                self._busy[exec_class] += 1
                self._maybe_mark_overlap_locked()
                waiter.event.set()
                progressed = True

    def _maybe_mark_overlap_locked(self) -> None:
        both = self._busy["qwen"] > 0 and self._busy["sarvam"] > 0
        if both and self._overlap_started is None:
            self._overlap_started = time.perf_counter()
        if not both and self._overlap_started is not None:
            dt = time.perf_counter() - self._overlap_started
            self._metrics["qwen_vl"].overlap_qwen_sarvam_seconds += dt
            self._metrics["sarvam"].overlap_qwen_sarvam_seconds += dt
            self._overlap_started = None

    def _acquire_gpu(self, exec_class: str, model: str, ctx: Optional[RequestContext], timer: StageTimer) -> str:
        registry = get_request_registry()
        server_request_id = ctx.server_request_id if ctx else None
        skip, reason = registry.should_skip_gpu(server_request_id)
        if skip:
            raise RequestCancelled(reason)

        waiter = _Waiter(event=threading.Event(), exec_class=exec_class, model=model)
        with self._cv:
            self.stats.waiting_for_gpu += 1
            self._model_waiting[model] += 1
            self._waiters[exec_class].append(waiter)
            self._dispatch_locked()

        start_wait = time.perf_counter()
        granted = False
        try:
            while not waiter.event.wait(timeout=0.5):
                skip, reason = registry.should_skip_gpu(server_request_id)
                if skip:
                    with self._cv:
                        removed = False
                        q = self._waiters[exec_class]
                        if waiter in q:
                            q.remove(waiter)
                            removed = True
                        if not removed and waiter.event.is_set():
                            self._busy[exec_class] = max(0, self._busy[exec_class] - 1)
                            self._maybe_mark_overlap_locked()
                            self._dispatch_locked()
                            granted = True
                        self.stats.waiting_for_gpu = max(0, self.stats.waiting_for_gpu - 1)
                        self._model_waiting[model] = max(0, self._model_waiting[model] - 1)
                    raise RequestCancelled(reason)
                self._emit_wait_threshold(
                    ctx, time.perf_counter() - start_wait + (timer.admission_wait_seconds or 0)
                )
            granted = True
        finally:
            pass

        skip, reason = registry.should_skip_gpu(server_request_id)
        if skip:
            with self._cv:
                self._busy[exec_class] = max(0, self._busy[exec_class] - 1)
                self._maybe_mark_overlap_locked()
                self.stats.waiting_for_gpu = max(0, self.stats.waiting_for_gpu - 1)
                self._model_waiting[model] = max(0, self._model_waiting[model] - 1)
                self.stats.gpu_running = sum(self._busy.values())
                self._dispatch_locked()
            raise RequestCancelled(reason)

        queue_wait = time.perf_counter() - start_wait
        with self._lock:
            self.stats.waiting_for_gpu = max(0, self.stats.waiting_for_gpu - 1)
            self._model_waiting[model] = max(0, self._model_waiting[model] - 1)
            self.stats.gpu_running = sum(self._busy.values())
            if self.stats.occupied_started is None:
                self.stats.occupied_started = time.perf_counter()
            self._metrics[model].queue_wait_sum += queue_wait
            self._metrics[model].queue_wait_count += 1
        if ctx and ctx.server_request_id:
            registry.mark(ctx.server_request_id, RequestState.RUNNING)
        return exec_class

    def _release_gpu(self, exec_class: str, model: str, inference_seconds: float, batch_size: int) -> None:
        with self._cv:
            self._busy[exec_class] = max(0, self._busy[exec_class] - 1)
            self._maybe_mark_overlap_locked()
            if self.stats.occupied_started is not None and sum(self._busy.values()) == 0:
                self.stats.occupied_seconds += time.perf_counter() - self.stats.occupied_started
                self.stats.occupied_started = None
            self.stats.gpu_running = sum(self._busy.values())
            m = self._metrics[model]
            m.inference_sum += max(0.0, inference_seconds)
            m.inference_count += 1
            m.items_sum += max(batch_size, 1)
            m.effective_batch_sum += max(batch_size, 1)
            self._dispatch_locked()

    @contextmanager
    def run(
        self,
        *,
        model: str,
        batch_size: int = 1,
        ctx: Optional[RequestContext] = None,
        use_cuda_events: bool = True,
    ):
        import torch

        ctx = ctx or get_request_context()
        timer = ctx.timer if ctx else StageTimer()
        internal = self._normalize_model(model)
        exec_class = exec_class_for_model(internal)
        queue_class = queue_class_for_model(internal)
        if timer.gpu_queue_entered <= 0:
            timer.gpu_queue_entered = StageTimer.now()

        emit(
            "gpu_queue_entered",
            model=internal,
            queue_depth=self.snapshot()["waiting_for_gpu"] + 1,
            queue_class=queue_class,
            schedule_lane=exec_class,
            batch_size=batch_size,
            **self.snapshot(),
        )

        gpu_held = False
        _inference_for_release = [0.0]
        try:
            # Cancel-before-GPU: deadline / registry only (no concurrent is_disconnected).
            skip, reason = get_request_registry().should_skip_gpu(
                ctx.server_request_id if ctx else None
            )
            if skip:
                raise RequestCancelled(reason)

            self._acquire_gpu(exec_class, internal, ctx, timer)
            gpu_held = True

            timer.gpu_started = StageTimer.now()
            reset_peak_memory_stats()
            mem_before = gpu_memory_snapshot()
            emit(
                "gpu_inference_started",
                model=internal,
                batch_size=batch_size,
                queue_wait_seconds=round(timer.gpu_queue_wait_seconds or 0.0, 6),
                queue_class=queue_class,
                schedule_lane=exec_class,
                **mem_before,
                **self.snapshot(),
            )

            start_event = end_event = None
            if use_cuda_events and torch.cuda.is_available() and internal != "insightface":
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
                    model=internal,
                    batch_size=batch_size,
                    inference_seconds=round(inference_seconds, 6),
                    seconds_per_item=round(inference_seconds / max(batch_size, 1), 6),
                    items_per_second=round(max(batch_size, 1) / max(inference_seconds, 1e-9), 6),
                    effective_batch_size=batch_size,
                    queue_class=queue_class,
                    schedule_lane=exec_class,
                    **mem_after,
                    **self.snapshot(),
                )
                _inference_for_release[0] = inference_seconds
        finally:
            if gpu_held:
                self._release_gpu(exec_class, internal, _inference_for_release[0], batch_size)


_SCHEDULER: Optional[GpuScheduler] = None
_SCHEDULER_LOCK = threading.Lock()


def get_gpu_scheduler() -> GpuScheduler:
    global _SCHEDULER
    with _SCHEDULER_LOCK:
        if _SCHEDULER is None:
            _SCHEDULER = GpuScheduler()
        return _SCHEDULER


def reset_gpu_scheduler_for_tests() -> None:
    global _SCHEDULER
    with _SCHEDULER_LOCK:
        _SCHEDULER = None
