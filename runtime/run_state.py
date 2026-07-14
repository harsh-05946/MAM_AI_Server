from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import json
import os
import shutil
import threading

from runtime.paths import (
    ARCHIVE_DIR,
    CURRENT_DIR,
    CURRENT_RUN_JSON,
    ensure_runtime_dirs,
)


def _stamp(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class RunState:
    """Separate long-lived server identity from reloadable campaign identity."""

    def __init__(self) -> None:
        ensure_runtime_dirs()
        self._lock = threading.RLock()
        self.server_run_id = os.getenv("SERVER_RUN_ID") or _stamp("server")
        self.campaign_run_id: Optional[str] = None
        self.campaign_name: Optional[str] = None
        self.campaign_state = "IDLE"  # IDLE | RUNNING | COMPLETED
        self.campaign_started_at: Optional[str] = None
        self.campaign_finished_at: Optional[str] = None
        self.expected_media_count: Optional[int] = None
        self.expected_ai_frames: Optional[int] = None
        self.source_media_hours: Optional[float] = None
        self.target_media_hours_per_hour: float = 3.0
        # Prefer immutable server id already written by the long-lived Uvicorn process.
        if CURRENT_RUN_JSON.exists():
            try:
                existing = json.loads(CURRENT_RUN_JSON.read_text(encoding="utf-8"))
                if existing.get("server_run_id") and not os.getenv("SERVER_RUN_ID"):
                    self.server_run_id = existing["server_run_id"]
                if existing.get("run_id"):
                    self.campaign_run_id = existing.get("run_id")
                self.campaign_name = existing.get("campaign_name")
                self.campaign_state = existing.get("state") or self.campaign_state
                self.campaign_started_at = existing.get("started_at")
                self.campaign_finished_at = existing.get("finished_at")
                self.expected_media_count = existing.get("expected_media_count")
                self.expected_ai_frames = existing.get("expected_ai_frames")
                self.source_media_hours = existing.get("source_media_hours")
                if existing.get("target_media_hours_per_hour") is not None:
                    self.target_media_hours_per_hour = float(existing["target_media_hours_per_hour"])
            except Exception:
                pass
        self._write_current()

    def active_run_id(self) -> str:
        with self._lock:
            return self.campaign_run_id or self.server_run_id

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "server_run_id": self.server_run_id,
                "run_id": self.campaign_run_id,
                "campaign_name": self.campaign_name,
                "state": self.campaign_state,
                "started_at": self.campaign_started_at,
                "finished_at": self.campaign_finished_at,
                "expected_media_count": self.expected_media_count,
                "expected_ai_frames": self.expected_ai_frames,
                "source_media_hours": self.source_media_hours,
                "target_media_hours_per_hour": self.target_media_hours_per_hour,
                "instance": os.getenv("INSTANCE_NAME", "main"),
                "pid": os.getpid(),
            }

    def _write_current(self) -> None:
        ensure_runtime_dirs()
        tmp = CURRENT_RUN_JSON.with_suffix(".json.tmp")
        data = self.snapshot()
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, CURRENT_RUN_JSON)

    def reload_from_disk(self) -> None:
        """Allow manage_run.py to update campaign state while Uvicorn runs."""
        if not CURRENT_RUN_JSON.exists():
            return
        try:
            data = json.loads(CURRENT_RUN_JSON.read_text(encoding="utf-8"))
        except Exception:
            return
        with self._lock:
            # Keep immutable server id from process start.
            if data.get("run_id"):
                self.campaign_run_id = data.get("run_id")
            self.campaign_name = data.get("campaign_name")
            self.campaign_state = data.get("state") or self.campaign_state
            self.campaign_started_at = data.get("started_at")
            self.campaign_finished_at = data.get("finished_at")
            self.expected_media_count = data.get("expected_media_count")
            self.expected_ai_frames = data.get("expected_ai_frames")
            self.source_media_hours = data.get("source_media_hours")
            if data.get("target_media_hours_per_hour") is not None:
                self.target_media_hours_per_hour = float(data["target_media_hours_per_hour"])

    def start_campaign(
        self,
        *,
        name: str,
        expected_media_count: Optional[int] = None,
        expected_ai_frames: Optional[int] = None,
        source_media_hours: Optional[float] = None,
        target_media_hours_per_hour: float = 3.0,
        force: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            if self.campaign_state == "RUNNING" and not force:
                raise RuntimeError(
                    f"Campaign {self.campaign_run_id} is still RUNNING. Finalize it first."
                )
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip()) or "campaign"
            self.campaign_run_id = f"run_{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.campaign_name = name
            self.campaign_state = "RUNNING"
            self.campaign_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
            self.campaign_finished_at = None
            self.expected_media_count = expected_media_count
            self.expected_ai_frames = expected_ai_frames
            self.source_media_hours = source_media_hours
            self.target_media_hours_per_hour = target_media_hours_per_hour
            self._reset_current_artifacts()
            self._write_current()
            return self.snapshot()

    def finalize_campaign(self, *, force: bool = False) -> dict[str, Any]:
        with self._lock:
            if self.campaign_state != "RUNNING" and not force:
                raise RuntimeError("No RUNNING campaign to finalize.")
            if not self.campaign_run_id:
                raise RuntimeError("No campaign_run_id set.")
            self.campaign_state = "COMPLETED"
            self.campaign_finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
            self._write_current()
            archive_dir = ARCHIVE_DIR / self.campaign_run_id
            if archive_dir.exists():
                shutil.rmtree(archive_dir)
            shutil.copytree(CURRENT_DIR, archive_dir)
            summary = self.snapshot()
            summary["archive_dir"] = str(archive_dir)
            (archive_dir / "run_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            return summary

    def _reset_current_artifacts(self) -> None:
        ensure_runtime_dirs()
        for name in (
            "ai_runtime_events.jsonl",
            "ai_endpoint_summary.csv",
            "ai_model_summary.csv",
            "gpu_metrics.csv",
            "host_metrics.csv",
            "error_summary.csv",
            "AI_THROUGHPUT_REPORT.md",
        ):
            path = CURRENT_DIR / name
            if path.exists():
                path.unlink()
        # Recreate empty events file.
        (CURRENT_DIR / "ai_runtime_events.jsonl").touch()


_STATE: Optional[RunState] = None
_STATE_LOCK = threading.Lock()


def get_run_state() -> RunState:
    global _STATE
    with _STATE_LOCK:
        if _STATE is None:
            _STATE = RunState()
        else:
            _STATE.reload_from_disk()
        return _STATE
