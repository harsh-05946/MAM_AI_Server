from __future__ import annotations

from typing import Any
import torch


def gpu_memory_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 4),
        "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 4),
        "gpu_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
        "gpu_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 4),
    }


def reset_peak_memory_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
