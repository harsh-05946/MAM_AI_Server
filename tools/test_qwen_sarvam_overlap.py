#!/usr/bin/env python3
"""Compare sequential vs concurrent Qwen + Sarvam wall time (GPU required).

Keep concurrent execution when:
  concurrent < 0.9 * sequential
and no OOM / output regression / severe p95 hang.

Usage:
  AI base must be ready. From this host:
    python tools/test_qwen_sarvam_overlap.py --base-url http://127.0.0.1:9001
"""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import sys
import time

import httpx
from PIL import Image


def _png_bytes() -> bytes:
    img = Image.new("RGB", (128, 128), color=(90, 40, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _post_qwen(client: httpx.Client, base: str, data: bytes) -> float:
    t0 = time.perf_counter()
    r = client.post(
        f"{base}/process/caption/qwen/batch",
        files=[("files", ("a.png", data, "image/png"))],
        data={"prompt": "one word"},
    )
    r.raise_for_status()
    return time.perf_counter() - t0


def _post_sarvam(client: httpx.Client, base: str) -> float:
    t0 = time.perf_counter()
    r = client.post(
        f"{base}/process/translation/sarvam/batch",
        json={"texts": ["hello world"], "target_lang": "Hindi"},
    )
    r.raise_for_status()
    return time.perf_counter() - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:9001")
    args = p.parse_args()
    base = args.base_url.rstrip("/")
    data = _png_bytes()

    with httpx.Client(timeout=600.0) as client:
        cap = client.get(f"{base}/internal/capacity")
        cap.raise_for_status()
        print("capacity:", json.dumps(cap.json().get("execution"), indent=2))

        print("sequential...")
        t_q = _post_qwen(client, base, data)
        t_s = _post_sarvam(client, base)
        sequential = t_q + t_s
        print(f"  qwen={t_q:.3f}s sarvam={t_s:.3f}s sum={sequential:.3f}s")

        print("concurrent...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            # Separate clients so connections don't serialize on one HTTP/1 session.
            def q():
                with httpx.Client(timeout=600.0) as c:
                    return _post_qwen(c, base, data)

            def s():
                with httpx.Client(timeout=600.0) as c:
                    return _post_sarvam(c, base)

            t0 = time.perf_counter()
            futs = [pool.submit(q), pool.submit(s)]
            times = [f.result() for f in futs]
            concurrent_wall = time.perf_counter() - t0
        print(f"  each={times} wall={concurrent_wall:.3f}s")

        threshold = 0.9 * sequential
        keep = concurrent_wall < threshold
        print(
            json.dumps(
                {
                    "sequential_seconds": round(sequential, 3),
                    "concurrent_wall_seconds": round(concurrent_wall, 3),
                    "threshold_0_9_sequential": round(threshold, 3),
                    "keep_concurrent_overlap": keep,
                    "recommendation": (
                        "AI_ENABLE_QWEN_SARVAM_OVERLAP=true"
                        if keep
                        else "AI_ENABLE_QWEN_SARVAM_OVERLAP=false (serialize GPU)"
                    ),
                },
                indent=2,
            )
        )
        return 0 if keep else 2


if __name__ == "__main__":
    sys.exit(main())
