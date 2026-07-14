from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_REPORTS = PROJECT_ROOT / "runtime_reports"
CURRENT_DIR = RUNTIME_REPORTS / "current"
ARCHIVE_DIR = RUNTIME_REPORTS / "archive"

EVENTS_JSONL = CURRENT_DIR / "ai_runtime_events.jsonl"
ENDPOINT_SUMMARY_CSV = CURRENT_DIR / "ai_endpoint_summary.csv"
MODEL_SUMMARY_CSV = CURRENT_DIR / "ai_model_summary.csv"
GPU_METRICS_CSV = CURRENT_DIR / "gpu_metrics.csv"
HOST_METRICS_CSV = CURRENT_DIR / "host_metrics.csv"
ERROR_SUMMARY_CSV = CURRENT_DIR / "error_summary.csv"
CURRENT_RUN_JSON = CURRENT_DIR / "current_run.json"
THROUGHPUT_REPORT_MD = CURRENT_DIR / "AI_THROUGHPUT_REPORT.md"


def ensure_runtime_dirs() -> None:
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
