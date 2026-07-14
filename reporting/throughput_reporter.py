#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reporting.aggregators import aggregate, read_events, write_summary_csvs
from reporting.bottleneck_rules import detect_bottlenecks
from reporting.markdown_report import build_markdown
from runtime.gpu_scheduler import get_gpu_scheduler
from runtime.paths import CURRENT_DIR, THROUGHPUT_REPORT_MD, ensure_runtime_dirs
from runtime.run_state import get_run_state


def _campaign_wall_seconds(run_state: dict) -> float | None:
    started = run_state.get("started_at")
    finished = run_state.get("finished_at")
    if not started:
        return None
    try:
        start_dt = datetime.fromisoformat(started)
        end_dt = datetime.fromisoformat(finished) if finished else datetime.now().astimezone()
        return max((end_dt - start_dt).total_seconds(), 0.0)
    except Exception:
        return None


def generate_report(*, finalize: bool = False) -> Path:
    ensure_runtime_dirs()
    state = get_run_state()
    run_state = state.snapshot()
    events = read_events()
    sched = get_gpu_scheduler().snapshot()
    wall = _campaign_wall_seconds(run_state)
    occupied = float(sched.get("occupied_seconds") or 0)
    if wall and wall > 0:
        sched["occupancy_percent"] = round(100.0 * occupied / wall, 1)
        sched["idle_seconds"] = round(max(wall - occupied, 0.0), 1)
    else:
        sched["occupancy_percent"] = 0.0
        sched["idle_seconds"] = 0.0

    agg = aggregate(events, run_state, sched)
    agg["occupied_seconds"] = occupied

    source_hours = run_state.get("source_media_hours")
    target = float(run_state.get("target_media_hours_per_hour") or 3.0)
    ai_frames = int((agg.get("workload") or {}).get("face_frames_processed") or 0)
    capacity = {
        "source_media_hours": source_hours,
        "wall_seconds": round(wall, 1) if wall is not None else None,
        "ai_frames_per_sec": round(ai_frames / wall, 3) if wall and wall > 0 else None,
        "target": target,
    }
    if source_hours and wall and wall > 0:
        mph = float(source_hours) / (wall / 3600.0)
        capacity["media_hours_per_hour"] = round(mph, 3)
        capacity["result"] = "PASS" if mph >= target else "FAIL"
    else:
        capacity["media_hours_per_hour"] = None
        capacity["result"] = "PENDING"
    agg["capacity"] = capacity

    bottlenecks = detect_bottlenecks(agg)
    report_state = "COMPLETED" if finalize or run_state.get("state") == "COMPLETED" else (
        run_state.get("state") or "RUNNING"
    )
    md = build_markdown(agg, bottlenecks, report_state=report_state)
    write_summary_csvs(agg)

    tmp = THROUGHPUT_REPORT_MD.with_suffix(".md.tmp")
    tmp.write_text(md, encoding="utf-8")
    os.replace(tmp, THROUGHPUT_REPORT_MD)
    return THROUGHPUT_REPORT_MD


def watch(interval: float = 15.0) -> None:
    ensure_runtime_dirs()
    while True:
        try:
            generate_report(finalize=False)
        except Exception as exc:
            # Keep watcher alive.
            err = CURRENT_DIR / "reporter_error.txt"
            err.write_text(str(exc), encoding="utf-8")
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="AI throughput reporter")
    parser.add_argument("--watch", action="store_true", help="Update report every 15 seconds")
    parser.add_argument("--finalize", action="store_true", help="Write final report snapshot")
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    if args.watch:
        watch(args.interval)
        return 0
    path = generate_report(finalize=args.finalize)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
