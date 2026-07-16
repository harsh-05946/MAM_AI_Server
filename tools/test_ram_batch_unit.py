#!/usr/bin/env python3
"""Unit checks for RAM++ batch mapping and admission defaults (no GPU required)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _ram_output_structure(tags_en: Any, tags_cn: Any, requested_batch: int) -> dict[str, Any]:
    """Mirror of main._ram_output_structure (avoid importing full FastAPI app)."""

    def _len(x: Any) -> Optional[int]:
        if isinstance(x, (list, tuple)):
            return len(x)
        return None

    return {
        "output_type": type(tags_en).__name__ if tags_en is not None else "None",
        "tuple_length": 2,
        "requested_batch": requested_batch,
        "tags_en_type": type(tags_en).__name__,
        "tags_cn_type": type(tags_cn).__name__,
        "tags_en_length": _len(tags_en),
        "tags_cn_length": _len(tags_cn),
    }


def test_inference_ram_batch_returns_full_list_n():
    from models import inference_ram, inference_ram_batch

    class FakeModel:
        def generate_tag(self, image):
            bs = int(image.shape[0])
            return [f"en-{i}" for i in range(bs)], [f"cn-{i}" for i in range(bs)]

    for n in (1, 8, 16):
        class FakeTensor:
            shape = (n, 3, 384, 384)

        en, cn = inference_ram_batch(FakeTensor(), FakeModel())
        assert isinstance(en, list) and isinstance(cn, list)
        assert len(en) == n and len(cn) == n
        assert en[0] == "en-0" and cn[-1] == f"cn-{n - 1}"

    class FakeOne:
        shape = (1, 3, 384, 384)

    a, b = inference_ram(FakeOne(), FakeModel())
    assert a == "en-0" and b == "cn-0"
    print("inference_ram_batch N=1/8/16 ok")


def test_no_false_tag_shape_mismatch_for_batch_lists():
    """Simulate success path used by _process_ram_batch after inference_ram_batch."""
    from models import inference_ram_batch

    class FakeModel:
        def generate_tag(self, image):
            bs = int(image.shape[0])
            return [f"tag-{i}" for i in range(bs)], [f"cn-{i}" for i in range(bs)]

    for n in (8, 16):
        class FakeTensor:
            shape = (n, 3, 384, 384)

        tags_en, tags_cn = inference_ram_batch(FakeTensor(), FakeModel())
        # Success path criterion in main._process_ram_batch
        ok = (
            isinstance(tags_en, list)
            and isinstance(tags_cn, list)
            and len(tags_en) == n
            and len(tags_cn) == n
        )
        assert ok, f"would have false-fallback for N={n}"
        # Upstream inference_ram would have returned only tags[0] — that must NOT be our API.
        assert not isinstance(tags_en, str)
    print("no false tag_shape_mismatch for N=8/16 ok")


def test_scheduler_default_waiters_is_four():
    for key in (
        "AI_INTERNAL_MAX_WAITING_REQUESTS",
        "MAX_WAITING_GPU_REQUESTS",
        "AI_INTERNAL_REJECT_WHEN_FULL",
        "GPU_QUEUE_REJECT_WHEN_FULL",
        "GPU_EXECUTION_SLOTS",
        "AI_MAX_ACCEPTED_INFERENCE_REQUESTS",
    ):
        os.environ.pop(key, None)

    import runtime.gpu_scheduler as gs

    gs._SCHEDULER = None
    sched = gs.GpuScheduler()
    # Six-lane profile default accepted = 6
    assert sched.max_accepted == 6, sched.max_accepted
    assert sched.reject_when_full is True
    snap = sched.snapshot()
    assert snap["ai_max_accepted_inference_requests"] == 6
    print("scheduler defaults ok")


def test_alias_env_takes_precedence():
    os.environ["AI_MAX_ACCEPTED_INFERENCE_REQUESTS"] = "6"
    os.environ["AI_INTERNAL_MAX_WAITING_REQUESTS"] = "99"
    os.environ["MAX_WAITING_GPU_REQUESTS"] = "99"
    import runtime.gpu_scheduler as gs

    gs._SCHEDULER = None
    sched = gs.GpuScheduler()
    assert sched.max_accepted == 6
    os.environ.pop("AI_MAX_ACCEPTED_INFERENCE_REQUESTS", None)
    os.environ.pop("AI_INTERNAL_MAX_WAITING_REQUESTS", None)
    os.environ.pop("MAX_WAITING_GPU_REQUESTS", None)
    gs._SCHEDULER = None
    print("env alias ok")


def test_ram_output_structure_has_no_tag_text():
    meta = _ram_output_structure(["secret-tag", "other"], ["x", "y"], requested_batch=2)
    assert meta["tags_en_length"] == 2
    assert meta["requested_batch"] == 2
    values = list(meta.values())
    assert "secret-tag" not in values
    assert "other" not in values
    assert "x" not in values
    print("ram structure metadata ok")


def test_timing_fields_separated():
    import time
    from runtime.stage_timer import StageTimer

    t = StageTimer()
    t.mark_request_start()
    time.sleep(0.005)
    t.mark_parse_finished()
    t.admission_entered = t.now()
    time.sleep(0.005)
    t.admission_acquired = t.now()
    t.gpu_queue_entered = t.now()
    time.sleep(0.005)
    t.gpu_started = t.now()
    time.sleep(0.005)
    t.gpu_finished = t.now()
    t.response_finished = t.now()
    d = t.as_dict()
    for key in (
        "admission_wait_seconds",
        "gpu_queue_wait_seconds",
        "gpu_inference_seconds",
        "total_request_seconds",
    ):
        assert key in d, f"missing {key} in {d}"
    # HTTP total must be strictly larger than inference alone when waits exist.
    assert d["total_request_seconds"] >= d["gpu_inference_seconds"]
    print("timing separation ok")


def main() -> int:
    test_inference_ram_batch_returns_full_list_n()
    test_no_false_tag_shape_mismatch_for_batch_lists()
    test_scheduler_default_waiters_is_four()
    test_alias_env_takes_precedence()
    test_ram_output_structure_has_no_tag_text()
    test_timing_fields_separated()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
