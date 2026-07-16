#!/usr/bin/env python3
"""GPU smoke: RAM++ stacked batch N=8/16 returns N tags once (no false fallback)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main() -> int:
    import torch
    from PIL import Image
    from models import inference_ram_batch, _load_ram_plus_model
    from ram import get_transform

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0

    device = "cuda"
    print("Loading RAM++…", flush=True)
    t0 = time.perf_counter()
    model = _load_ram_plus_model(device)
    transform = get_transform(image_size=384)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    for n in (8, 16):
        images = [Image.new("RGB", (384, 384), color=(40 + i * 10, 80, 120)) for i in range(n)]
        stacked = torch.stack([transform(img) for img in images]).to(device)
        t1 = time.perf_counter()
        with torch.inference_mode():
            tags_en, tags_cn = inference_ram_batch(stacked, model)
        elapsed = time.perf_counter() - t1
        assert isinstance(tags_en, list) and isinstance(tags_cn, list), (
            type(tags_en),
            type(tags_cn),
        )
        assert len(tags_en) == n and len(tags_cn) == n, (
            f"shape mismatch N={n} en={len(tags_en) if isinstance(tags_en, list) else tags_en} "
            f"cn={len(tags_cn) if isinstance(tags_cn, list) else tags_cn}"
        )
        # Sanity: each entry is a string (tag list joined or similar).
        assert all(isinstance(x, str) for x in tags_en)
        print(f"N={n} ok in {elapsed:.3f}s; sample_en={tags_en[0][:80]!r}")

    print("RAM GPU smoke PASS (zero tag_shape_mismatch)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
