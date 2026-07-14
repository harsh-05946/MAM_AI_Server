#!/usr/bin/env python3
"""Validate InsightFace ORT providers without generating inference traffic."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("REQUIRE_FACE_CUDA", "true")


def main() -> int:
    from models import (
        FaceAnalysis,
        REQUIRE_FACE_CUDA,
        _preload_ort_cuda_libs,
        validate_face_providers,
    )

    _preload_ort_cuda_libs()
    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider",
    ]
    try:
        face_app = FaceAnalysis(name="buffalo_l", providers=providers)
    except TypeError:
        face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    status = validate_face_providers(face_app)
    print(json.dumps(status, indent=2))
    if REQUIRE_FACE_CUDA and not status.get("pass"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
