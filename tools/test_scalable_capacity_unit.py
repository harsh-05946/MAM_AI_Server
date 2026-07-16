#!/usr/bin/env python3
"""Unit tests for l40s-scalable-lanes-v1 capacity derivation + /internal/capacity schema."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _clear():
    for key in (
        "AI_CAPACITY_PROFILE",
        "MAX_PARALLEL_MEDIA",
        "PER_MEDIA_AI_INFLIGHT",
        "PER_MEDIA_MODEL_INFLIGHT",
        "AI_ACCEPTED_QUEUE_LIMIT",
        "AI_MAX_ACCEPTED_INFERENCE_REQUESTS",
        "AI_INTERNAL_MAX_WAITING_REQUESTS",
        "AI_GPU_SLOTS_VISUAL",
        "AI_GPU_SLOTS_QWEN",
        "AI_GPU_SLOTS_SARVAM",
        "FACE_BATCH_MAX",
        "EMOTION_BATCH_MAX",
        "SCENE_BATCH_MAX",
        "RAM_BATCH_MAX",
        "QWEN_BATCH_MAX",
        "SARVAM_BATCH_MAX",
        "EMBED_BATCH_MAX",
    ):
        os.environ.pop(key, None)


def _set_media(n: int):
    os.environ["AI_CAPACITY_PROFILE"] = "l40s-scalable-lanes-v1"
    os.environ["MAX_PARALLEL_MEDIA"] = str(n)
    os.environ["PER_MEDIA_AI_INFLIGHT"] = "8"
    os.environ["PER_MEDIA_MODEL_INFLIGHT"] = "1"
    os.environ["AI_ACCEPTED_QUEUE_LIMIT"] = "6"


def test_media_scales_fleet():
    from runtime.capacity_profile import build_capacity_response, derive_admission

    for media, fleet_global, sarvam_fleet in ((1, 8, 1), (2, 16, 2), (3, 24, 3)):
        _set_media(media)
        adm = derive_admission()
        assert adm["fleet_global"] == fleet_global, (media, adm)
        assert adm["models"]["sarvam"]["fleet"] == sarvam_fleet
        assert adm["models"]["sarvam"]["per_media"] == 1
        assert adm["per_media"] == 8
        body = build_capacity_response()
        assert body["profile"] == "l40s-scalable-lanes-v1"
        assert body["profile_version"] == 1
        assert body["max_parallel_media"] == media
        assert body["admission"]["fleet_global"] == fleet_global
        assert body["admission"]["per_media"] == 8
        assert body["admission"]["models"]["sarvam"]["fleet"] == sarvam_fleet
        assert body["admission"]["models"]["sarvam"]["per_media"] == 1
        assert body["admission"]["models"]["face"]["batch"] == 16
        assert body["admission"]["models"]["emotion"]["batch"] == 64
        assert body["execution"]["accepted_queue_limit"] == 6
        assert body["execution"]["gpu_slots"] == {"visual": 1, "qwen": 1, "sarvam": 1}
        print(f"media={media} fleet ok")


def test_media_increase_does_not_raise_gpu_slots():
    _set_media(1)
    from runtime.capacity_profile import execution_bounds

    e1 = execution_bounds()
    _set_media(5)
    e5 = execution_bounds()
    assert e1["gpu_slots"] == e5["gpu_slots"]
    assert e1["accepted_queue_limit"] == e5["accepted_queue_limit"] == 6
    print("gpu slots independent of media ok")


def test_scheduler_uses_accepted_queue_not_fleet():
    _set_media(2)  # fleet_global=16
    import runtime.gpu_scheduler as gs

    gs.reset_gpu_scheduler_for_tests()
    sched = gs.GpuScheduler()
    assert sched.max_accepted == 6
    # fleet_model=2 → two face admits ok; third rejected at model lane
    ok1, _, _ = sched.try_admit_nowait("insightface")
    ok2, _, _ = sched.try_admit_nowait("insightface")
    ok3, model, code = sched.try_admit_nowait("insightface")
    assert ok1 and ok2
    assert not ok3 and model == "face" and code == "MODEL_CAPACITY_FULL"
    sched.release_admission("insightface")
    sched.release_admission("insightface")
    print("accepted queue + model fleet admit ok")


def test_profile_json_loads():
    _set_media(2)
    from runtime.capacity_profile import load_profile_json

    data = load_profile_json("l40s-scalable-lanes-v1")
    assert data["profile"] == "l40s-scalable-lanes-v1"
    assert data["scalable_fleet"] is True
    assert data["per_media_inflight"] == 8
    print("profile json ok")


def test_six_lane_rollback_still_builds():
    _clear()
    os.environ["AI_CAPACITY_PROFILE"] = "l40s-six-lane-v1"
    os.environ["AI_ACCEPTED_QUEUE_LIMIT"] = "6"
    from runtime.capacity_profile import build_capacity_response

    body = build_capacity_response(ready=True)
    assert body["profile"] == "l40s-six-lane-v1"
    assert body["recommended_client_inflight"] == 6
    print("six-lane rollback ok")


def main():
    _clear()
    test_media_scales_fleet()
    test_media_increase_does_not_raise_gpu_slots()
    test_scheduler_uses_accepted_queue_not_fleet()
    test_profile_json_loads()
    test_six_lane_rollback_still_builds()
    print("ALL scalable capacity tests passed")


if __name__ == "__main__":
    main()
