from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import time


@dataclass
class StageTimer:
    request_started: float = 0.0
    parse_started: float = 0.0
    parse_finished: float = 0.0
    preprocess_started: float = 0.0
    preprocess_finished: float = 0.0
    microbatch_started: float = 0.0
    microbatch_finished: float = 0.0
    admission_entered: float = 0.0
    admission_acquired: float = 0.0
    gpu_queue_entered: float = 0.0
    gpu_started: float = 0.0
    gpu_finished: float = 0.0
    postprocess_finished: float = 0.0
    response_finished: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def now() -> float:
        return time.perf_counter()

    def mark_request_start(self) -> None:
        self.request_started = self.now()
        self.parse_started = self.request_started

    def mark_parse_finished(self) -> None:
        self.parse_finished = self.now()

    def mark_preprocess_start(self) -> None:
        self.preprocess_started = self.now()

    def mark_preprocess_finished(self) -> None:
        self.preprocess_finished = self.now()

    def _delta(self, start: float, end: float) -> Optional[float]:
        if start <= 0 or end <= 0 or end < start:
            return None
        return end - start

    @property
    def request_parse_seconds(self) -> Optional[float]:
        return self._delta(self.parse_started, self.parse_finished)

    @property
    def preprocess_seconds(self) -> Optional[float]:
        return self._delta(self.preprocess_started, self.preprocess_finished)

    @property
    def microbatch_wait_seconds(self) -> Optional[float]:
        return self._delta(self.microbatch_started, self.microbatch_finished)

    @property
    def admission_wait_seconds(self) -> Optional[float]:
        return self._delta(self.admission_entered, self.admission_acquired)

    @property
    def gpu_queue_wait_seconds(self) -> Optional[float]:
        return self._delta(self.gpu_queue_entered, self.gpu_started)

    @property
    def gpu_inference_seconds(self) -> Optional[float]:
        return self._delta(self.gpu_started, self.gpu_finished)

    @property
    def postprocess_seconds(self) -> Optional[float]:
        return self._delta(self.gpu_finished, self.postprocess_finished)

    @property
    def total_request_seconds(self) -> Optional[float]:
        end = self.response_finished or self.now()
        return self._delta(self.request_started, end)

    def as_dict(self) -> dict[str, Any]:
        out = {
            "request_parse_seconds": self.request_parse_seconds,
            "preprocess_seconds": self.preprocess_seconds,
            "microbatch_wait_seconds": self.microbatch_wait_seconds,
            "admission_wait_seconds": self.admission_wait_seconds,
            "gpu_queue_wait_seconds": self.gpu_queue_wait_seconds,
            "gpu_inference_seconds": self.gpu_inference_seconds,
            "postprocess_seconds": self.postprocess_seconds,
            "total_request_seconds": self.total_request_seconds,
        }
        out.update(self.extras)
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items() if v is not None}
