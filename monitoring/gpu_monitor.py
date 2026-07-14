from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import threading

from runtime.paths import GPU_METRICS_CSV, ensure_runtime_dirs
from runtime.run_state import get_run_state


class GpuMonitor:
    def __init__(self, interval_sec: float = 1.0, path: Path = GPU_METRICS_CSV):
        self.interval_sec = interval_sec
        self.path = path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        ensure_runtime_dirs()
        self._init_nvml()
        self._ensure_header()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._shutdown_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None
            self._handle = None

    def _shutdown_nvml(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        self._nvml = None
        self._handle = None

    def _ensure_header(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "timestamp",
                    "run_id",
                    "server_run_id",
                    "gpu_index",
                    "gpu_utilization_percent",
                    "memory_utilization_percent",
                    "memory_used_mb",
                    "memory_free_mb",
                    "power_draw_watts",
                    "power_limit_watts",
                    "temperature_celsius",
                    "sm_clock_mhz",
                    "memory_clock_mhz",
                ]
            )

    def _sample(self) -> list:
        state = get_run_state()
        ts = datetime.now().astimezone().isoformat(timespec="seconds")
        if self._nvml is None or self._handle is None:
            return [ts, state.active_run_id(), state.server_run_id, 0, "", "", "", "", "", "", "", "", ""]
        pynvml = self._nvml
        handle = self._handle
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                power = ""
            try:
                power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
            except Exception:
                power_limit = ""
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = ""
            try:
                sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                sm = ""
                mem_clock = ""
            return [
                ts,
                state.active_run_id(),
                state.server_run_id,
                0,
                util.gpu,
                util.memory,
                round(mem.used / (1024**2), 2),
                round(mem.free / (1024**2), 2),
                power,
                power_limit,
                temp,
                sm,
                mem_clock,
            ]
        except Exception:
            return [ts, state.active_run_id(), state.server_run_id, 0, "", "", "", "", "", "", "", "", ""]

    def _loop(self) -> None:
        while not self._stop.is_set():
            row = self._sample()
            try:
                with open(self.path, "a", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow(row)
            except Exception:
                pass
            self._stop.wait(self.interval_sec)


_MONITOR: GpuMonitor | None = None


def start_gpu_monitor() -> GpuMonitor:
    global _MONITOR
    if _MONITOR is None:
        _MONITOR = GpuMonitor()
    _MONITOR.start()
    return _MONITOR


def stop_gpu_monitor() -> None:
    global _MONITOR
    if _MONITOR is not None:
        _MONITOR.stop()


if __name__ == "__main__":
    import time

    mon = start_gpu_monitor()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        mon.stop()
