#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python reporting/manage_run.py` from project root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reporting.throughput_reporter import generate_report
from runtime.event_logger import emit, get_event_logger
from runtime.gpu_scheduler import get_gpu_scheduler
from runtime.paths import ensure_runtime_dirs
from runtime.run_state import get_run_state


def cmd_start(args: argparse.Namespace) -> int:
    ensure_runtime_dirs()
    state = get_run_state()
    snap = state.start_campaign(
        name=args.name,
        expected_media_count=args.expected_media_count,
        expected_ai_frames=args.expected_ai_frames,
        source_media_hours=args.source_media_hours,
        target_media_hours_per_hour=args.target_media_hours_per_hour,
        force=args.force,
    )
    get_gpu_scheduler().mark_campaign_start()
    get_event_logger().start()
    emit(
        "campaign_started",
        server_run_id=snap["server_run_id"],
        run_id=snap["run_id"],
        campaign_name=snap["campaign_name"],
        expected_media_count=snap["expected_media_count"],
        expected_ai_frames=snap["expected_ai_frames"],
        source_media_hours=snap["source_media_hours"],
        target_media_hours_per_hour=snap["target_media_hours_per_hour"],
    )
    generate_report(finalize=False)
    print(json.dumps(snap, indent=2))
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    ensure_runtime_dirs()
    state = get_run_state()
    sched = get_gpu_scheduler().snapshot()
    active = int(sched.get("active_requests") or 0)
    waiting_adm = int(sched.get("waiting_for_admission") or 0)
    waiting_gpu = int(sched.get("waiting_for_gpu") or 0)
    running = int(sched.get("gpu_running") or 0)
    if not args.force and (active or waiting_adm or waiting_gpu or running):
        raise SystemExit(
            f"Refusing finalize while work is active "
            f"(active={active}, waiting_admission={waiting_adm}, waiting_gpu={waiting_gpu}, gpu_running={running}). "
            f"Use --force only if you intentionally want a partial archive."
        )
    generate_report(finalize=True)
    snap = state.finalize_campaign(force=args.force)
    emit(
        "campaign_finalized",
        server_run_id=snap["server_run_id"],
        run_id=snap["run_id"],
        archive_dir=snap.get("archive_dir"),
    )
    get_event_logger().flush()
    print(json.dumps(snap, indent=2))
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    ensure_runtime_dirs()
    state = get_run_state().snapshot()
    sched = get_gpu_scheduler().snapshot()
    print(json.dumps({"run": state, "scheduler": sched}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage AI throughput campaign runs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="Start a new campaign without reloading models")
    p_start.add_argument("--name", required=True)
    p_start.add_argument("--expected-media-count", type=int, default=None)
    p_start.add_argument("--expected-ai-frames", type=int, default=None)
    p_start.add_argument("--source-media-hours", type=float, default=None)
    p_start.add_argument("--target-media-hours-per-hour", type=float, default=3.0)
    p_start.add_argument("--force", action="store_true")
    p_start.set_defaults(func=cmd_start)

    p_fin = sub.add_parser("finalize", help="Finalize and archive current campaign")
    p_fin.add_argument("--force", action="store_true")
    p_fin.set_defaults(func=cmd_finalize)

    p_status = sub.add_parser("status", help="Show server/campaign status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
