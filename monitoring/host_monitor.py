from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import os
import threading

from runtime.paths import HOST_METRICS_CSV, ensure_runtime_dirs
from runtime.run_state import get_run_state


class HostMonitor:
    def __init__(self, interval_sec: float = 1.0, path: Path = HOST_METRICS_CSV):
        self.interval_sec = interval_sec
        self.path = path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        ensure_runtime_dirs()
        try:
            import psutil

            self._proc = psutil.Process(os.getpid())
            # Prime cpu percent.
            self._proc.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except Exception:
            self._proc = None
        self._ensure_header()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="host-monitor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _ensure_header(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(
                [
                    "timestamp",
                    "run_id",
                    "server_run_id",
                    "cpu_total_percent",
                    "cpu_process_percent",
                    "system_ram_used_gb",
                    "system_ram_available_gb",
                    "process_ram_rss_gb",
                    "process_thread_count",
                    "open_file_count",
                ]
            )

    def _sample(self) -> list:
        state = get_run_state()
        ts = datetime.now().astimezone().isoformat(timespec="seconds")
        try:
            import psutil

            vm = psutil.virtual_memory()
            cpu_total = psutil.cpu_percent(interval=None)
            if self._proc is None:
                self._proc = psutil.Process(os.getpid())
            cpu_proc = self._proc.cpu_percent(interval=None)
            rss = self._proc.memory_info().rss / (1024**3)
            threads = self._proc.num_threads()
            try:
                files = self._proc.num_fds()
            except Exception:
                files = ""
            return [
                ts,
                state.active_run_id(),
                state.server_run_id,
                cpu_total,
                cpu_proc,
                round((vm.total - vm.available) / (1024**3), 3),
                round(vm.available / (1024**3), 3),
                round(rss, 3),
                threads,
                files,
            ]
        except Exception:
            return [ts, state.active_run_id(), state.server_run_id, "", "", "", "", "", "", ""]

    def _loop(self) -> None:
        while not self._stop.is_set():
            row = self._sample()
            try:
                with open(self.path, "a", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow(row)
            except Exception:
                pass
            self._stop.wait(self.interval_sec)


_MONITOR: HostMonitor | None = None


def start_host_monitor() -> HostMonitor:
    global _MONITOR
    if _MONITOR is None:
        _MONITOR = HostMonitor()
    _MONITOR.start()
    return _MONITOR


def stop_host_monitor() -> None:
    global _MONITOR
    if _MONITOR is not None:
        _MONITOR.stop()


if __name__ == "__main__":
    import time

    mon = start_host_monitor()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        mon.stop()
