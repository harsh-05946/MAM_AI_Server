#!/usr/bin/env python3
"""
Smoke checks for embedding + image batch APIs.

Usage:
  python tools/smoke_batch_endpoints.py --base-url http://127.0.0.1:8001

Requires the main inference server running with models loaded.
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


def _files_n(prefix: str, n: int) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [("files", _tiny_png(f"{prefix}_{i}.png")) for i in range(n)]


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke test batch endpoints")
    p.add_argument("--base-url", default="http://127.0.0.1:8001", help="Main API base URL (no trailing slash)")
    args = p.parse_args()
    base = args.base_url.rstrip("/")

    with httpx.Client(timeout=120.0) as client:
        r = client.get(f"{base}/health")
        r.raise_for_status()
        health = r.json()
        loaded = set(health.get("loaded_models") or [])
        print("health:", {k: health.get(k) for k in ("status", "loaded_models") if k in health})

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

        r = client.post(f"{base}/process/embeddings", json={"texts": ["singleton"]})
        r.raise_for_status()
        data = r.json()
        assert data.get("count") == 1 and len(data["embeddings"][0]) == dim, data
        print("embeddings single ok")

        two_files = _files_n("img", 2)

        if "emotion" in loaded:
            r = client.post(f"{base}/process/emotion/batch", files=two_files)
            if r.status_code >= 400:
                print("emotion batch failed:", r.status_code, r.text[:500])
                return 1
            arr = r.json()
            if not isinstance(arr, list) or len(arr) != 2:
                print("unexpected emotion batch response:", arr)
                return 1
            for i, row in enumerate(arr):
                if "emotion" not in row or "confidence" not in row:
                    print("bad emotion row", i, row)
                    return 1
            print("emotion batch ok:", json.dumps(arr, indent=2)[:400])
        else:
            print("skip emotion batch: emotion not in loaded_models")

        if "scene" in loaded:
            r = client.post(f"{base}/process/scene/batch", files=two_files)
            if r.status_code >= 400:
                print("scene batch failed:", r.status_code, r.text[:500])
                return 1
            arr = r.json()
            if not isinstance(arr, list) or len(arr) != 2:
                print("unexpected scene batch response:", arr)
                return 1
            for i, row in enumerate(arr):
                if "scene" not in row:
                    print("bad scene row", i, row)
                    return 1
            print("scene batch ok:", json.dumps(arr, indent=2)[:400])
        else:
            print("skip scene batch: scene not in loaded_models")

        if "ram_plus" in loaded:
            r = client.post(f"{base}/process/object-detection/batch", files=two_files)
            if r.status_code >= 400:
                print("ram batch failed:", r.status_code, r.text[:500])
                return 1
            arr = r.json()
            if not isinstance(arr, list) or len(arr) != 2:
                print("unexpected ram batch response:", arr)
                return 1
            for i, row in enumerate(arr):
                if "tags_en" not in row or "tags_cn" not in row:
                    print("bad ram row", i, row)
                    return 1
            print("ram batch ok:", json.dumps(arr, indent=2)[:400])
        else:
            print("skip ram batch: ram_plus not in loaded_models")

        if "face" in loaded:
            r = client.post(f"{base}/process/face/batch", files=two_files)
            if r.status_code >= 400:
                print("face batch failed:", r.status_code, r.text[:500])
                return 1
            arr = r.json()
            if not isinstance(arr, list) or len(arr) != 2:
                print("unexpected face batch response:", arr)
                return 1
            for i, row in enumerate(arr):
                if "faces" not in row or not isinstance(row["faces"], list):
                    print("bad face row", i, row)
                    return 1
            print("face batch ok:", json.dumps(arr, indent=2)[:400])
        else:
            print("skip face batch: face not in loaded_models")

        if "sarvam" in loaded:
            r = client.post(
                f"{base}/process/translation/sarvam/batch",
                json={"texts": ["Hello", "Good morning"], "target_lang": "hindi"},
            )
            if r.status_code >= 400:
                print("sarvam batch failed:", r.status_code, r.text[:500])
                return 1
            data = r.json()
            if data.get("count") != 2 or len(data.get("translated_texts") or []) != 2:
                print("unexpected sarvam batch response:", data)
                return 1
            print("sarvam batch ok:", json.dumps(data, ensure_ascii=False)[:400])
        else:
            print("skip sarvam batch: sarvam not in loaded_models")

        if "qwen_vl" in loaded:
            r = client.post(
                f"{base}/process/caption/qwen/batch",
                files=_files_n("qwen", 10),
                data={"prompt": "Describe this image in one word."},
            )
            if r.status_code >= 400:
                print("qwen batch failed:", r.status_code, r.text[:500])
                return 1
            arr = r.json()
            if not isinstance(arr, list) or len(arr) != 10:
                print("unexpected qwen batch response:", arr)
                return 1
            for i, row in enumerate(arr):
                if "caption" not in row:
                    print("bad qwen row", i, row)
                    return 1
            print("qwen batch ok:", json.dumps(arr, indent=2)[:400])
        else:
            print("skip qwen batch: qwen_vl not in loaded_models")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except httpx.HTTPError as e:
        print("HTTP error:", e, file=sys.stderr)
        sys.exit(1)
