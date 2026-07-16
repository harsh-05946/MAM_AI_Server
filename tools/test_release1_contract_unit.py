#!/usr/bin/env python3
"""Release 1 unit tests: readiness, 429 defaults, cancel-before-GPU."""
from __future__ import annotations

import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _clear_sched_env():
    for key in (
        "AI_INTERNAL_MAX_WAITING_REQUESTS",
        "AI_MAX_ACCEPTED_INFERENCE_REQUESTS",
        "AI_ACCEPTED_QUEUE_LIMIT",
        "MAX_WAITING_GPU_REQUESTS",
        "AI_INTERNAL_REJECT_WHEN_FULL",
        "GPU_QUEUE_REJECT_WHEN_FULL",
        "GPU_EXECUTION_SLOTS",
        "AI_MAX_ACTIVE_GENERATIVE_REQUESTS",
        "AI_MAX_ACTIVE_QWEN_REQUESTS",
        "AI_MAX_ACTIVE_SARVAM_REQUESTS",
        "AI_COMBINED_GENERATIVE_LIMIT_ENABLED",
        "AI_CAPACITY_PROFILE",
        "MAX_PARALLEL_MEDIA",
        "PER_MEDIA_AI_INFLIGHT",
        "PER_MEDIA_MODEL_INFLIGHT",
        "GPU_FAIRNESS",
    ):
        os.environ.pop(key, None)


def test_scheduler_reject_defaults_true():
    _clear_sched_env()
    import runtime.gpu_scheduler as gs

    gs._SCHEDULER = None
    sched = gs.GpuScheduler()
    assert sched.max_accepted == 6, sched.max_accepted
    assert sched.reject_when_full is True
    assert sched.max_active_qwen == 1
    assert sched.max_active_sarvam == 1
    snap = sched.snapshot()
    assert snap["reject_when_full"] is True
    assert snap["ai_max_accepted_inference_requests"] == 6
    print("scheduler reject/generative defaults ok")


def test_readiness_alive_vs_ready_warmup():
    from runtime.readiness_manager import reset_readiness_manager_for_tests, get_readiness_manager

    reset_readiness_manager_for_tests()
    rm = get_readiness_manager()
    snap = rm.snapshot()
    assert snap.accepting_requests is False
    assert snap.as_dict()["status"] == "not_ready"
    assert snap.service_state == "STARTING"

    # Simulate loaded required models without opening traffic.
    loaded = {"face", "emotion", "scene", "ram_plus", "embed"}
    rm.mark_startup_loaded()
    rm.refresh(loaded)
    # Not warmed / not opened yet
    body = rm.snapshot().as_dict()
    assert body["status"] == "not_ready"
    assert body["reason"] in {"warming", "not_opened", "starting"}

    rm.mark_warmed()
    rm.refresh(loaded)
    rm.open_for_traffic()
    body = rm.snapshot().as_dict()
    # gpu_ready depends on torch.cuda — still may be not_ready without GPU in CI sandbox
    assert body["warmed"] is True
    assert body["startup_complete"] is True
    assert body["models_ready"] is True
    print("readiness warmup gates ok")


def test_readiness_drain_503_shape():
    from runtime.readiness_manager import reset_readiness_manager_for_tests, get_readiness_manager

    reset_readiness_manager_for_tests()
    rm = get_readiness_manager()
    loaded = {"face", "emotion", "scene", "ram_plus", "embed"}
    rm.mark_startup_loaded()
    rm.mark_warmed()
    rm.refresh(loaded)
    rm.open_for_traffic()
    rm.begin_drain()
    body = rm.snapshot().as_dict()
    assert body["status"] == "not_ready"
    assert body["draining"] is True
    assert body["accepting_requests"] is False
    print("drain readiness ok")


def test_admission_full_rejects():
    _clear_sched_env()
    os.environ["AI_ACCEPTED_QUEUE_LIMIT"] = "1"
    os.environ["AI_MAX_ACCEPTED_INFERENCE_REQUESTS"] = "1"
    os.environ["AI_INTERNAL_REJECT_WHEN_FULL"] = "true"
    import runtime.gpu_scheduler as gs

    gs._SCHEDULER = None
    sched = gs.GpuScheduler()
    ok1, _, _ = sched.try_admit_nowait("emotion")
    assert ok1 is True
    ok2, model, code = sched.try_admit_nowait("scene")
    assert ok2 is False
    assert code == "SERVER_CAPACITY_FULL"
    sched.release_admission("emotion")
    print("429 admission full ok")


def test_cancel_before_gpu():
    from runtime.request_registry import (
        RequestCancelled,
        RequestState,
        get_request_registry,
        reset_request_registry_for_tests,
    )
    from runtime.request_context import RequestContext
    from runtime.stage_timer import StageTimer
    import runtime.gpu_scheduler as gs

    _clear_sched_env()
    os.environ["GPU_FAIRNESS"] = "false"
    reset_request_registry_for_tests()
    gs._SCHEDULER = None
    sched = gs.GpuScheduler()
    reg = get_request_registry()

    # Hold the GPU with one thread so the second waits in queue.
    hold = threading.Event()
    released = threading.Event()

    def holder():
        ctx = RequestContext(
            server_run_id="s",
            run_id="r",
            job_id="",
            media_id="",
            request_id="holder",
            endpoint="/process/emotion",
            http_method="POST",
            model="emotion",
            timer=StageTimer(),
            server_request_id="holder-sid",
        )
        reg.register(
            logical_request_id="holder",
            endpoint=ctx.endpoint,
            model="emotion",
            server_request_id="holder-sid",
        )
        with sched.run(model="emotion", batch_size=1, ctx=ctx, use_cuda_events=False):
            hold.set()
            released.wait(timeout=5)

    t = threading.Thread(target=holder, daemon=True)
    t.start()
    assert hold.wait(timeout=5)

    waiter_ctx = RequestContext(
        server_run_id="s",
        run_id="r",
        job_id="",
        media_id="",
        request_id="waiter",
        endpoint="/process/emotion",
        http_method="POST",
        model="emotion",
        timer=StageTimer(),
        server_request_id="waiter-sid",
    )
    reg.register(
        logical_request_id="waiter",
        endpoint=waiter_ctx.endpoint,
        model="emotion",
        server_request_id="waiter-sid",
    )
    # Mark disconnect while waiting for GPU
    def cancel_soon():
        time.sleep(0.2)
        reg.mark_disconnected("waiter-sid")

    threading.Thread(target=cancel_soon, daemon=True).start()
    raised = None
    try:
        with sched.run(model="emotion", batch_size=1, ctx=waiter_ctx, use_cuda_events=False):
            pass
    except RequestCancelled as exc:
        raised = exc
    finally:
        released.set()
        t.join(timeout=5)

    assert raised is not None, "expected RequestCancelled"
    assert raised.reason == "client_disconnected"
    item = reg.get("waiter-sid")
    assert item is None or item.state == RequestState.CANCELLED
    print("cancel-before-gpu ok")


def test_deadline_cancel():
    from runtime.request_registry import get_request_registry, reset_request_registry_for_tests

    reset_request_registry_for_tests()
    reg = get_request_registry()
    mono = time.perf_counter() - 1.0  # already expired
    item = reg.register(
        logical_request_id="x",
        server_request_id="ddl",
        deadline_monotonic=mono,
    )
    skip, reason = reg.should_skip_gpu("ddl")
    assert skip is True
    assert reason == "deadline_exceeded"
    assert item.state.value == "CANCELLED"
    print("deadline cancel ok")


def main():
    test_scheduler_reject_defaults_true()
    test_readiness_alive_vs_ready_warmup()
    test_readiness_drain_503_shape()
    test_admission_full_rejects()
    test_deadline_cancel()
    test_cancel_before_gpu()
    print("ALL Release-1 unit tests passed")


if __name__ == "__main__":
    main()
