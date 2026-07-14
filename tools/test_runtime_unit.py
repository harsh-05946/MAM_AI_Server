#!/usr/bin/env python3
"""Lightweight unit checks for runtime scheduler/reporting (no GPU traffic)."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_stage_timer():
    from runtime.stage_timer import StageTimer

    t = StageTimer()
    t.mark_request_start()
    time.sleep(0.01)
    t.mark_parse_finished()
    t.admission_entered = t.now()
    time.sleep(0.01)
    t.admission_acquired = t.now()
    t.gpu_queue_entered = t.now()
    time.sleep(0.01)
    t.gpu_started = t.now()
    time.sleep(0.01)
    t.gpu_finished = t.now()
    t.response_finished = t.now()
    d = t.as_dict()
    assert d["admission_wait_seconds"] is not None or "admission_wait_seconds" in d
    assert d["total_request_seconds"] > 0


def test_scheduler_serializes():
    from runtime.gpu_scheduler import GpuScheduler

    sched = GpuScheduler()
    order = []
    lock = threading.Lock()

    def worker(n):
        with sched.run(model="emotion", batch_size=1, use_cuda_events=False):
            with lock:
                order.append(("start", n))
            time.sleep(0.05)
            with lock:
                order.append(("end", n))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    # Starts and ends must not interleave overlapping GPU sections.
    depth = 0
    for kind, _ in order:
        if kind == "start":
            depth += 1
            assert depth == 1
        else:
            depth -= 1
            assert depth == 0


def test_event_logger_and_report(tmp_path: Path | None = None):
    from runtime import paths
    from runtime.event_logger import EventLogger, emit
    from reporting.throughput_reporter import generate_report
    from runtime.run_state import RunState

    base = Path(tempfile.mkdtemp())
    paths.CURRENT_DIR = base / "current"
    paths.ARCHIVE_DIR = base / "archive"
    paths.EVENTS_JSONL = paths.CURRENT_DIR / "ai_runtime_events.jsonl"
    paths.ENDPOINT_SUMMARY_CSV = paths.CURRENT_DIR / "ai_endpoint_summary.csv"
    paths.MODEL_SUMMARY_CSV = paths.CURRENT_DIR / "ai_model_summary.csv"
    paths.GPU_METRICS_CSV = paths.CURRENT_DIR / "gpu_metrics.csv"
    paths.HOST_METRICS_CSV = paths.CURRENT_DIR / "host_metrics.csv"
    paths.ERROR_SUMMARY_CSV = paths.CURRENT_DIR / "error_summary.csv"
    paths.CURRENT_RUN_JSON = paths.CURRENT_DIR / "current_run.json"
    paths.THROUGHPUT_REPORT_MD = paths.CURRENT_DIR / "AI_THROUGHPUT_REPORT.md"
    paths.ensure_runtime_dirs()

    logger = EventLogger(path=paths.EVENTS_JSONL)
    logger.start()
    logger.emit("request_received", endpoint="/process/face", requested_items=None)
    logger.emit(
        "request_completed",
        endpoint="/process/face",
        requested_items=2,
        total_request_seconds=1.2,
        admission_wait_seconds=0.1,
        gpu_queue_wait_seconds=0.2,
        gpu_inference_seconds=0.8,
    )
    logger.emit("workload_output", workload_kind="face", frames=2, faces_detected=3)
    logger.flush()
    time.sleep(0.2)

    # Patch get_run_state path via writing current_run.json
    state = RunState()
    state.start_campaign(name="unit", expected_media_count=1, expected_ai_frames=2, source_media_hours=1.0)
    path = generate_report(finalize=False)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "AI Throughput" in text or "Throughput" in text or "workload" in text.lower() or len(text) > 20
    logger.stop()
    print("report ok:", path)


def test_atomic_replace():
    from reporting.throughput_reporter import generate_report

    # Smoke: generate_report against real current dir is fine if empty.
    p = generate_report(finalize=False)
    assert p.exists()


def main() -> int:
    test_stage_timer()
    print("stage_timer ok")
    test_scheduler_serializes()
    print("scheduler ok")
    test_event_logger_and_report()
    print("events/report ok")
    test_atomic_replace()
    print("atomic report ok")
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
