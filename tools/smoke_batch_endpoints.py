#!/usr/bin/env python3
"""
Smoke checks for embedding + emotion batch APIs.

Usage:
  python tools/smoke_batch_endpoints.py --base-url http://127.0.0.1:8001

Requires the main inference server running with models loaded (emotion/embed).
"""
from __future__ import annotations

import argparse
import io
import json
import sys

import httpx
from PIL import Image


def _tiny_png(name: str) -> tuple[str, bytes, str]:
    img = Image.new("RGB", (64, 64), color=(120, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return (name, buf.getvalue(), "image/png")


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke test emotion batch + embeddings batch/microbatch")
    p.add_argument("--base-url", default="http://127.0.0.1:8001", help="Main API base URL (no trailing slash)")
    args = p.parse_args()
    base = args.base_url.rstrip("/")

    with httpx.Client(timeout=120.0) as client:
        r = client.get(f"{base}/health")
        r.raise_for_status()
        health = r.json()
        loaded = set(health.get("loaded_models") or [])
        print("health:", {k: health.get(k) for k in ("status", "loaded_models") if k in health})

        # Embeddings: multi-text one request
        r = client.post(
            f"{base}/process/embeddings",
            json={"texts": ["alpha", "beta", "gamma"]},
        )
        r.raise_for_status()
        data = r.json()
        assert data.get("count") == 3, data
        assert len(data.get("embeddings") or []) == 3, data
        dim = len(data["embeddings"][0])
        print(f"embeddings batch ok: count=3 dim={dim}")

        # Embeddings: single (may use micro-batcher when concurrent; one call still valid)
        r = client.post(f"{base}/process/embeddings", json={"texts": ["singleton"]})
        r.raise_for_status()
        data = r.json()
        assert data.get("count") == 1 and len(data["embeddings"][0]) == dim, data
        print("embeddings single ok")

        if "emotion" not in loaded:
            print("skip emotion batch: emotion model not in loaded_models")
            return 0

        png1 = _tiny_png("a.png")
        png2 = _tiny_png("b.png")
        files = [
            ("files", png1),
            ("files", png2),
        ]
        r = client.post(f"{base}/process/emotion/batch", files=files)
        if r.status_code >= 400:
            print("emotion batch failed:", r.status_code, r.text[:500])
            return 1
        arr = r.json()
        if not isinstance(arr, list) or len(arr) != 2:
            print("unexpected emotion batch response:", arr)
            return 1
        for i, row in enumerate(arr):
            if "emotion" not in row or "confidence" not in row:
                print("bad row", i, row)
                return 1
        print("emotion batch ok:", json.dumps(arr, indent=2)[:400])

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except httpx.HTTPError as e:
        print("HTTP error:", e, file=sys.stderr)
        sys.exit(1)
