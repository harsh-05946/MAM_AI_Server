from __future__ import annotations

from datetime import datetime
from typing import Any


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---" if i == 0 else "---:" for i in range(len(headers))]) + " |"
    body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([line, sep, *body]) if rows else line + "\n" + sep + "\n| (none) |" + " |" * (len(headers) - 1)


def build_markdown(agg: dict[str, Any], bottlenecks: list[str], *, report_state: str = "RUNNING") -> str:
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    rs = agg.get("run_state") or {}
    counts = agg.get("counts") or {}
    gpu = agg.get("gpu") or {}
    host = agg.get("host") or {}
    workload = agg.get("workload") or {}
    timeout = agg.get("timeout_risk") or {}
    sched = agg.get("scheduler") or {}
    capacity = agg.get("capacity") or {}

    provider_rows = []
    for p in agg.get("providers") or []:
        provider_rows.append(
            [
                p.get("model"),
                p.get("active_provider"),
                "PASS" if p.get("validation_passed") else "FAIL",
            ]
        )

    endpoint_rows = [
        [
            r["endpoint"],
            r["requests"],
            r["items"],
            r["avg_total"],
            r["avg_admission"],
            r["avg_queue"],
            r["avg_inference"],
            r["items_per_sec"],
            r["p95_total"],
        ]
        for r in agg.get("endpoint_rows") or []
    ]
    model_rows = [
        [r["model"], r["items"], r["total_gpu_time"], r["avg_batch"], r["avg_sec_per_item"], r["items_per_sec"]]
        for r in agg.get("model_rows") or []
    ]

    insight = next((r for r in (agg.get("model_rows") or []) if r.get("model") == "insightface"), None)
    face_provider = "UNKNOWN"
    for p in agg.get("providers") or []:
        if str(p.get("model") or "").startswith("insightface"):
            face_provider = p.get("active_provider") or face_provider
            if not p.get("validation_passed"):
                face_provider = f"FAIL ({face_provider})"
                break
            face_provider = f"PASS ({p.get('active_provider')})"

    sections = [
        "# AI Server Throughput Report",
        "",
        "## Report Status",
        "",
        f"Last updated: {now}",
        "",
        f"Report state: {report_state}",
        "",
        f"Server run ID: `{rs.get('server_run_id')}`",
        "",
        f"Campaign run ID: `{rs.get('run_id')}`",
        "",
        f"Campaign name: {rs.get('campaign_name') or '—'}",
        "",
        "## Run Information",
        "",
        _md_table(
            ["Field", "Value"],
            [
                ["Server run ID", rs.get("server_run_id")],
                ["Campaign run ID", rs.get("run_id")],
                ["State", rs.get("state")],
                ["Started at", rs.get("started_at")],
                ["Finished at", rs.get("finished_at") or "—"],
                ["Expected media", rs.get("expected_media_count")],
                ["Expected AI frames", rs.get("expected_ai_frames")],
                ["Source media hours", rs.get("source_media_hours")],
                ["Target media-hours/hour", rs.get("target_media_hours_per_hour")],
            ],
        ),
        "",
        "## Current Runtime Status",
        "",
        _md_table(
            ["Metric", "Value"],
            [
                ["Active requests", sched.get("active_requests", 0)],
                ["Waiting for admission", sched.get("waiting_for_admission", 0)],
                ["Waiting for GPU", sched.get("waiting_for_gpu", 0)],
                ["GPU running", sched.get("gpu_running", 0)],
            ],
        ),
        "",
        "## Provider Validation",
        "",
        _md_table(["Model", "Active provider", "Status"], provider_rows or [["—", "—", "PENDING"]]),
        "",
        "## Overall Request Summary",
        "",
        _md_table(
            ["Metric", "Value"],
            [
                ["Requests received", counts.get("received", 0)],
                ["Requests completed", counts.get("completed", 0)],
                ["Requests failed", counts.get("failed", 0)],
                ["Queue rejected", counts.get("queue_rejected", 0)],
                ["CUDA OOM events", counts.get("oom", 0)],
                ["RAM++ fallbacks", counts.get("ram_fallbacks", 0)],
            ],
        ),
        "",
        "## Workload Summary",
        "",
        _md_table(
            ["Workload", "Count"],
            [
                ["Media workflows observed", workload.get("media_workflows_observed", 0)],
                ["Expected AI frames", rs.get("expected_ai_frames") or "—"],
                ["Face frames processed", workload.get("face_frames_processed", 0)],
                ["Faces detected", workload.get("faces_detected", 0)],
                ["Emotion face crops", workload.get("emotion_face_crops", 0)],
                ["Scene frames", workload.get("scene_frames", 0)],
                ["RAM++ frames", workload.get("ram_plus_frames", 0)],
                ["Qwen images", workload.get("qwen_images", 0)],
                ["Translation inputs", workload.get("translation_inputs", 0)],
                ["Embedding inputs", workload.get("embedding_inputs", 0)],
            ],
        ),
        "",
        "## Endpoint Performance",
        "",
        _md_table(
            ["Endpoint", "Requests", "Items", "Avg total", "Avg admission", "Avg queue", "Avg inference", "Items/sec", "P95 total"],
            endpoint_rows,
        ),
        "",
        "## Model Performance",
        "",
        _md_table(
            ["Model", "Items", "Total GPU time", "Avg batch", "Avg sec/item", "Items/sec"],
            model_rows,
        ),
        "",
        "## GPU Queue Analysis",
        "",
        _md_table(
            ["Metric", "Value"],
            [
                ["Current admission waiters", sched.get("waiting_for_admission", 0)],
                ["Current GPU waiters", sched.get("waiting_for_gpu", 0)],
                ["GPU running", sched.get("gpu_running", 0)],
                ["Occupied seconds", round(float(agg.get("occupied_seconds") or 0), 1)],
                ["Scheduler occupancy", f"{sched.get('occupancy_percent', 0)}%"],
                ["Scheduler idle seconds", sched.get("idle_seconds", 0)],
            ],
        ),
        "",
        "## Timeout Risk",
        "",
        _md_table(
            ["Metric", "Value"],
            [
                ["Requests waiting >300 s", timeout.get("waiting_gt_300", 0)],
                ["Requests waiting >600 s", timeout.get("waiting_gt_600", 0)],
                ["Requests waiting >840 s", timeout.get("waiting_gt_840", 0)],
                ["Maximum request wait", timeout.get("max_wait_seconds", 0)],
                ["Proxy timeout", timeout.get("proxy_timeout_seconds", 900)],
            ],
        ),
        "",
        "## GPU Utilization",
        "",
        _md_table(
            ["Metric", "Value"],
            [
                ["Average GPU utilization", f"{gpu.get('avg_util', 0)}%"],
                ["P50 GPU utilization", f"{gpu.get('p50_util', 0)}%"],
                ["P95 GPU utilization", f"{gpu.get('p95_util', 0)}%"],
                ["Maximum GPU utilization", f"{gpu.get('max_util', 0)}%"],
                ["GPU idle percentage", f"{gpu.get('idle_pct', 0)}%"],
            ],
        ),
        "",
        "## GPU Memory",
        "",
        _md_table(
            ["Metric", "Value"],
            [
                ["Average VRAM", f"{round(float(gpu.get('avg_vram_mb') or 0)/1024, 2)} GB"],
                ["Peak VRAM", f"{round(float(gpu.get('peak_vram_mb') or 0)/1024, 2)} GB"],
                ["CUDA OOM events", counts.get("oom", 0)],
            ],
        ),
        "",
        "## GPU Power",
        "",
        _md_table(
            ["Metric", "Value"],
            [
                ["Average power", f"{gpu.get('avg_power', 0)} W"],
                ["P95 power", f"{gpu.get('p95_power', 0)} W"],
                ["Maximum power", f"{gpu.get('max_power', 0)} W"],
            ],
        ),
        "",
        "## Host CPU and RAM",
        "",
        _md_table(
            ["Metric", "Value"],
            [
                ["Average CPU", f"{host.get('avg_cpu', 0)}%"],
                ["P95 CPU", f"{host.get('p95_cpu', 0)}%"],
            ],
        ),
        "",
        "## InsightFace Performance",
        "",
        f"Active provider status: **{face_provider}**",
        "",
        _md_table(
            ["Metric", "Value"],
            [
                ["Frames processed", insight.get("items") if insight else 0],
                ["Average sec/item", insight.get("avg_sec_per_item") if insight else 0],
                ["Throughput items/sec", insight.get("items_per_sec") if insight else 0],
                ["Total ORT/GPU time", insight.get("total_gpu_time") if insight else 0],
            ],
        ),
        "",
        "## Capacity Indicators",
        "",
        _md_table(
            ["Metric", "Value"],
            [
                ["Media processed (observed IDs)", workload.get("media_workflows_observed", 0)],
                ["Source media hours", capacity.get("source_media_hours", rs.get("source_media_hours") or "—")],
                ["Wall-clock duration seconds", capacity.get("wall_seconds", "—")],
                ["AI frame throughput", capacity.get("ai_frames_per_sec", "—")],
                ["Media-hours/hour", capacity.get("media_hours_per_hour", "—")],
                ["Target media-hours/hour", capacity.get("target", rs.get("target_media_hours_per_hour") or 3.0)],
                ["Capacity result", capacity.get("result", "PENDING")],
            ],
        ),
        "",
        "## Current Bottlenecks",
        "",
    ]
    for i, note in enumerate(bottlenecks, 1):
        sections.append(f"{i}. {note}")
    sections.extend(["", "## Final Run Summary", "", f"Report state: {report_state}", ""])
    return "\n".join(sections)
