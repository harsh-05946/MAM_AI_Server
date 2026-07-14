from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import csv
import json
import math
import statistics

from runtime.paths import (
    ENDPOINT_SUMMARY_CSV,
    ERROR_SUMMARY_CSV,
    EVENTS_JSONL,
    GPU_METRICS_CSV,
    HOST_METRICS_CSV,
    MODEL_SUMMARY_CSV,
)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def read_events(path: Path = EVENTS_JSONL) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def aggregate(events: list[dict[str, Any]], run_state: dict[str, Any], scheduler_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    completed = [e for e in events if e.get("event") == "request_completed"]
    failed = [e for e in events if e.get("event") == "request_failed"]
    received = [e for e in events if e.get("event") == "request_received"]
    gpu_done = [e for e in events if e.get("event") == "gpu_inference_completed"]
    micro = [e for e in events if e.get("event") == "microbatch_formed"]
    providers = [e for e in events if e.get("event") == "provider_validation"]
    ram_fb = [e for e in events if e.get("event") == "ram_batch_fallback"]
    wait_thr = [e for e in events if e.get("event") == "request_wait_threshold_exceeded"]
    oom = [e for e in events if e.get("event") in {"cuda_oom", "cuda_oom_retry", "cuda_oom_retry_failed"}]
    rejects = [e for e in events if e.get("event") == "queue_rejected"]

    by_endpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in completed:
        by_endpoint[e.get("endpoint") or "unknown"].append(e)

    endpoint_rows = []
    for endpoint, rows in sorted(by_endpoint.items()):
        totals = [float(r.get("total_request_seconds") or 0) for r in rows]
        queues = [float(r.get("gpu_queue_wait_seconds") or 0) for r in rows]
        admissions = [float(r.get("admission_wait_seconds") or 0) for r in rows]
        inferences = [float(r.get("gpu_inference_seconds") or 0) for r in rows]
        items = sum(int(r.get("requested_items") or 0) for r in rows)
        avg_inf = statistics.mean(inferences) if inferences else 0.0
        endpoint_rows.append(
            {
                "endpoint": endpoint,
                "requests": len(rows),
                "items": items,
                "avg_total": round(statistics.mean(totals), 3) if totals else 0,
                "avg_admission": round(statistics.mean(admissions), 3) if admissions else 0,
                "avg_queue": round(statistics.mean(queues), 3) if queues else 0,
                "avg_inference": round(avg_inf, 3),
                "items_per_sec": round(items / max(sum(inferences), 1e-9), 3),
                "p95_total": round(_pct(totals, 0.95), 3) if totals else 0,
            }
        )

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in gpu_done:
        by_model[e.get("model") or "unknown"].append(e)
    model_rows = []
    for model, rows in sorted(by_model.items()):
        items = sum(int(r.get("batch_size") or 0) for r in rows)
        gpu_time = sum(float(r.get("inference_seconds") or 0) for r in rows)
        batches = [int(r.get("batch_size") or 0) for r in rows]
        model_rows.append(
            {
                "model": model,
                "items": items,
                "total_gpu_time": round(gpu_time, 3),
                "avg_batch": round(statistics.mean(batches), 2) if batches else 0,
                "avg_sec_per_item": round(gpu_time / max(items, 1), 4),
                "items_per_sec": round(items / max(gpu_time, 1e-9), 3),
            }
        )

    batch_rows = []
    for e in micro:
        limit = float(e.get("batch_limit") or 0) or 1
        eff = float(e.get("effective_batch_size") or 0)
        batch_rows.append(
            {
                "model": e.get("model"),
                "batch_limit": limit,
                "effective_batch_size": eff,
                "fill_percent": float(e.get("batch_fill_percent") or (100.0 * eff / limit)),
            }
        )

    # Prefer explicit workload_output events; fall back to request_completed item counts.
    workload_events = [e for e in events if e.get("event") == "workload_output"]
    if workload_events:
        face_frames = sum(int(e.get("frames") or 0) for e in workload_events if e.get("workload_kind") == "face")
        faces_detected = sum(int(e.get("faces_detected") or 0) for e in workload_events if e.get("workload_kind") == "face")
        emotion_crops = sum(int(e.get("emotion_crops") or 0) for e in workload_events if e.get("workload_kind") == "emotion")
        scene_frames = sum(int(e.get("images") or 0) for e in workload_events if e.get("workload_kind") == "scene")
        ram_frames = sum(int(e.get("images") or 0) for e in workload_events if e.get("workload_kind") == "ram_plus")
        qwen_images = sum(int(e.get("images") or 0) for e in workload_events if e.get("workload_kind") == "qwen_vl")
        translation_inputs = sum(int(e.get("texts") or 0) for e in workload_events if e.get("workload_kind") == "sarvam")
        embedding_inputs = sum(int(e.get("texts") or 0) for e in workload_events if e.get("workload_kind") == "embed")
    else:
        faces_detected = sum(int(e.get("faces_detected") or 0) for e in completed if (e.get("endpoint") or "").startswith("/process/face"))
        face_frames = sum(int(e.get("successful_items") or e.get("requested_items") or 0) for e in completed if (e.get("endpoint") or "").startswith("/process/face"))
        emotion_crops = sum(int(e.get("successful_items") or e.get("requested_items") or 0) for e in completed if (e.get("endpoint") or "").startswith("/process/emotion"))
        scene_frames = sum(int(e.get("successful_items") or e.get("requested_items") or 0) for e in completed if (e.get("endpoint") or "").startswith("/process/scene"))
        ram_frames = sum(int(e.get("successful_items") or e.get("requested_items") or 0) for e in completed if (e.get("endpoint") or "").startswith("/process/object-detection"))
        qwen_images = sum(int(e.get("successful_items") or e.get("requested_items") or 0) for e in completed if (e.get("endpoint") or "").startswith("/process/caption/qwen"))
        translation_inputs = sum(int(e.get("successful_items") or e.get("requested_items") or 0) for e in completed if (e.get("endpoint") or "").startswith("/process/translation"))
        embedding_inputs = sum(int(e.get("successful_items") or e.get("requested_items") or 0) for e in completed if e.get("endpoint") == "/process/embeddings")

    media_ids = {e.get("media_id") for e in completed if e.get("media_id") and e.get("media_id") != "unknown_media"}

    gpu_rows = read_csv_dicts(GPU_METRICS_CSV)
    host_rows = read_csv_dicts(HOST_METRICS_CSV)

    def _nums(rows, key):
        out = []
        for r in rows:
            try:
                if r.get(key) not in (None, ""):
                    out.append(float(r[key]))
            except Exception:
                pass
        return out

    gpu_util = _nums(gpu_rows, "gpu_utilization_percent")
    power = _nums(gpu_rows, "power_draw_watts")
    mem_used = _nums(gpu_rows, "memory_used_mb")
    cpu_total = _nums(host_rows, "cpu_total_percent")

    wait_warnings = sum(1 for e in wait_thr if e.get("severity") == "warning")
    wait_critical = sum(1 for e in wait_thr if e.get("severity") == "critical")
    wait_timeout = sum(1 for e in wait_thr if e.get("severity") == "timeout_risk")
    max_wait = max([float(e.get("total_wait_seconds") or 0) for e in wait_thr] + [0])

    occupied = float((scheduler_snapshot or {}).get("occupied_seconds") or 0)
    # Campaign duration from run_state timestamps if available is handled by caller.

    return {
        "counts": {
            "received": len(received),
            "completed": len(completed),
            "failed": len(failed),
            "oom": len(oom),
            "queue_rejected": len(rejects),
            "ram_fallbacks": len(ram_fb),
        },
        "endpoint_rows": endpoint_rows,
        "model_rows": model_rows,
        "batch_rows": batch_rows,
        "providers": providers,
        "workload": {
            "media_workflows_observed": len(media_ids),
            "face_frames_processed": face_frames,
            "faces_detected": faces_detected,
            "emotion_face_crops": emotion_crops,
            "scene_frames": scene_frames,
            "ram_plus_frames": ram_frames,
            "qwen_images": qwen_images,
            "translation_inputs": translation_inputs,
            "embedding_inputs": embedding_inputs,
        },
        "gpu": {
            "avg_util": round(statistics.mean(gpu_util), 1) if gpu_util else 0,
            "p50_util": round(_pct(gpu_util, 0.5), 1) if gpu_util else 0,
            "p95_util": round(_pct(gpu_util, 0.95), 1) if gpu_util else 0,
            "max_util": round(max(gpu_util), 1) if gpu_util else 0,
            "idle_pct": round(100.0 * sum(1 for u in gpu_util if u < 5) / max(len(gpu_util), 1), 1) if gpu_util else 0,
            "avg_power": round(statistics.mean(power), 1) if power else 0,
            "p95_power": round(_pct(power, 0.95), 1) if power else 0,
            "max_power": round(max(power), 1) if power else 0,
            "avg_vram_mb": round(statistics.mean(mem_used), 1) if mem_used else 0,
            "peak_vram_mb": round(max(mem_used), 1) if mem_used else 0,
        },
        "host": {
            "avg_cpu": round(statistics.mean(cpu_total), 1) if cpu_total else 0,
            "p95_cpu": round(_pct(cpu_total, 0.95), 1) if cpu_total else 0,
        },
        "timeout_risk": {
            "waiting_gt_300": wait_warnings,
            "waiting_gt_600": wait_critical,
            "waiting_gt_840": wait_timeout,
            "max_wait_seconds": round(max_wait, 1),
            "proxy_timeout_seconds": 900,
        },
        "scheduler": scheduler_snapshot or {},
        "occupied_seconds": occupied,
        "run_state": run_state,
    }


def write_summary_csvs(agg: dict[str, Any]) -> None:
    with open(ENDPOINT_SUMMARY_CSV, "w", newline="", encoding="utf-8") as fh:
        fields = [
            "endpoint",
            "requests",
            "items",
            "avg_total",
            "avg_admission",
            "avg_queue",
            "avg_inference",
            "items_per_sec",
            "p95_total",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in agg["endpoint_rows"]:
            w.writerow(row)

    with open(MODEL_SUMMARY_CSV, "w", newline="", encoding="utf-8") as fh:
        fields = ["model", "items", "total_gpu_time", "avg_batch", "avg_sec_per_item", "items_per_sec"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in agg["model_rows"]:
            w.writerow(row)

    with open(ERROR_SUMMARY_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["error", "count"])
        w.writerow(["request_failed", agg["counts"]["failed"]])
        w.writerow(["cuda_oom_events", agg["counts"]["oom"]])
        w.writerow(["queue_rejected", agg["counts"]["queue_rejected"]])
        w.writerow(["ram_batch_fallback", agg["counts"]["ram_fallbacks"]])
