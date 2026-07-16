#!/usr/bin/env python3
"""Unit tests for l40s-six-lane-v1 capacity + model-aware scheduling."""
from __future__ import annotations

import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _clear_env():
    for key in list(os.environ):
        if key.startswith(("AI_", "GPU_", "MAX_", "PER_MEDIA")):
            if key in (
                "AI_CAPACITY_PROFILE",
                "AI_MAX_ACCEPTED_INFERENCE_REQUESTS",
                "AI_ACCEPTED_QUEUE_LIMIT",
                "AI_INTERNAL_MAX_WAITING_REQUESTS",
                "AI_INTERNAL_REJECT_WHEN_FULL",
                "AI_COMBINED_GENERATIVE_LIMIT_ENABLED",
                "AI_ENABLE_QWEN_SARVAM_OVERLAP",
                "AI_ENABLE_VISUAL_QWEN_OVERLAP",
                "AI_ENABLE_VISUAL_SARVAM_OVERLAP",
                "AI_VISUAL_EXECUTION_SLOTS",
                "AI_QWEN_EXECUTION_SLOTS",
                "AI_SARVAM_EXECUTION_SLOTS",
                "AI_GPU_SLOTS_VISUAL",
                "AI_GPU_SLOTS_QWEN",
                "AI_GPU_SLOTS_SARVAM",
                "MAX_PARALLEL_MEDIA",
                "PER_MEDIA_AI_INFLIGHT",
                "PER_MEDIA_MODEL_INFLIGHT",
                "FACE_BATCH_MAX",
                "EMOTION_BATCH_MAX",
                "SCENE_BATCH_MAX",
                "RAM_BATCH_MAX",
                "QWEN_BATCH_MAX",
                "SARVAM_BATCH_MAX",
                "EMBED_BATCH_MAX",
            ):
                os.environ.pop(key, None)


def test_capacity_contract_shape():
    _clear_env()
    os.environ["AI_CAPACITY_PROFILE"] = "l40s-six-lane-v1"
    os.environ["AI_ACCEPTED_QUEUE_LIMIT"] = "6"
    os.environ["FACE_BATCH_MAX"] = "16"
    os.environ["EMOTION_BATCH_MAX"] = "64"
    from runtime.capacity_profile import build_capacity_response, capacity_limits

    lim = capacity_limits()
    assert lim["profile"] == "l40s-six-lane-v1"
    assert lim["recommended_client_inflight"] == 6
    body = build_capacity_response(ready=True)
    assert body["profile"] == "l40s-six-lane-v1"
    assert body["models"]["face"]["max_concurrent"] == 1
    assert body["models"]["face"]["batch"] == 16
    print("capacity contract ok")


def test_per_model_admission_rejects_second_face():
    _clear_env()
    os.environ["AI_CAPACITY_PROFILE"] = "l40s-six-lane-v1"
    os.environ["AI_ACCEPTED_QUEUE_LIMIT"] = "6"
    os.environ["MAX_PARALLEL_MEDIA"] = "1"
    os.environ["PER_MEDIA_MODEL_INFLIGHT"] = "1"
    import runtime.gpu_scheduler as gs

    gs.reset_gpu_scheduler_for_tests()
    sched = gs.GpuScheduler()
    ok1, _, code1 = sched.try_admit_nowait("insightface")
    assert ok1 and code1 == ""
    ok2, model, code2 = sched.try_admit_nowait("insightface")
    assert not ok2
    assert model == "face"
    assert code2 == "MODEL_CAPACITY_FULL"
    ok3, _, code3 = sched.try_admit_nowait("emotion")
    assert ok3 and code3 == ""
    sched.release_admission("insightface")
    sched.release_admission("emotion")
    print("per-model admission ok")


def test_qwen_sarvam_can_admit_together():
    _clear_env()
    os.environ["AI_CAPACITY_PROFILE"] = "l40s-scalable-lanes-v1"
    os.environ["AI_COMBINED_GENERATIVE_LIMIT_ENABLED"] = "false"
    os.environ["AI_ACCEPTED_QUEUE_LIMIT"] = "6"
    os.environ["MAX_PARALLEL_MEDIA"] = "2"
    import runtime.gpu_scheduler as gs

    gs.reset_gpu_scheduler_for_tests()
    sched = gs.GpuScheduler()
    ok_q, _, _ = sched.try_admit_nowait("qwen_vl")
    ok_s, _, _ = sched.try_admit_nowait("sarvam")
    assert ok_q and ok_s
    sched.release_admission("qwen_vl")
    sched.release_admission("sarvam")
    print("qwen+sarvam admit ok")


def test_qwen_sarvam_overlap_dispatch():
    _clear_env()
    os.environ["AI_ENABLE_QWEN_SARVAM_OVERLAP"] = "true"
    os.environ["AI_ENABLE_VISUAL_QWEN_OVERLAP"] = "false"
    os.environ["AI_ENABLE_VISUAL_SARVAM_OVERLAP"] = "false"
    import runtime.gpu_scheduler as gs
    from runtime.request_context import RequestContext
    from runtime.stage_timer import StageTimer

    gs.reset_gpu_scheduler_for_tests()
    sched = gs.GpuScheduler()

    started = []
    done = threading.Event()

    def run_model(name, hold_event):
        ctx = RequestContext(
            server_run_id="s",
            run_id="r",
            job_id="",
            media_id="",
            request_id=name,
            endpoint="/x",
            http_method="POST",
            model=name,
            timer=StageTimer(),
            server_request_id=name + "-sid",
        )
        with sched.run(model=name, batch_size=1, ctx=ctx, use_cuda_events=False):
            started.append(name)
            hold_event.wait(timeout=3)

    hold_q = threading.Event()
    hold_s = threading.Event()
    t1 = threading.Thread(target=run_model, args=("qwen_vl", hold_q), daemon=True)
    t2 = threading.Thread(target=run_model, args=("sarvam", hold_s), daemon=True)
    t1.start()
    t2.start()
    deadline = time.time() + 2
    while time.time() < deadline and len(started) < 2:
        time.sleep(0.05)
    assert set(started) == {"qwen_vl", "sarvam"}, started
    hold_q.set()
    hold_s.set()
    t1.join(timeout=3)
    t2.join(timeout=3)
    print("qwen/sarvam overlap dispatch ok")


def test_visual_blocks_qwen_without_overlap():
    _clear_env()
    os.environ["AI_ENABLE_QWEN_SARVAM_OVERLAP"] = "true"
    os.environ["AI_ENABLE_VISUAL_QWEN_OVERLAP"] = "false"
    import runtime.gpu_scheduler as gs
    from runtime.request_context import RequestContext
    from runtime.stage_timer import StageTimer

    gs.reset_gpu_scheduler_for_tests()
    sched = gs.GpuScheduler()

    visual_holding = threading.Event()
    qwen_started = threading.Event()

    def hold_visual():
        ctx = RequestContext(
            server_run_id="s",
            run_id="r",
            job_id="",
            media_id="",
            request_id="vis",
            endpoint="/x",
            http_method="POST",
            model="emotion",
            timer=StageTimer(),
            server_request_id="vis-sid",
        )
        with sched.run(model="emotion", batch_size=1, ctx=ctx, use_cuda_events=False):
            visual_holding.set()
            time.sleep(0.8)

    def try_qwen():
        visual_holding.wait(timeout=2)
        ctx = RequestContext(
            server_run_id="s",
            run_id="r",
            job_id="",
            media_id="",
            request_id="qw",
            endpoint="/x",
            http_method="POST",
            model="qwen_vl",
            timer=StageTimer(),
            server_request_id="qw-sid",
        )
        with sched.run(model="qwen_vl", batch_size=1, ctx=ctx, use_cuda_events=False):
            qwen_started.set()

    t1 = threading.Thread(target=hold_visual, daemon=True)
    t2 = threading.Thread(target=try_qwen, daemon=True)
    t1.start()
    t2.start()
    time.sleep(0.3)
    assert visual_holding.is_set()
    assert not qwen_started.is_set(), "qwen must not overlap visual"
    t1.join(timeout=3)
    t2.join(timeout=3)
    assert qwen_started.is_set()
    print("visual blocks qwen ok")


def main():
    test_capacity_contract_shape()
    test_per_model_admission_rejects_second_face()
    test_qwen_sarvam_can_admit_together()
    test_qwen_sarvam_overlap_dispatch()
    test_visual_blocks_qwen_without_overlap()
    print("ALL six-lane unit tests passed")


if __name__ == "__main__":
    main()
