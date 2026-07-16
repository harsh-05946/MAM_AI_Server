#!/usr/bin/env python3
"""Parity: native RAM++ tags vs Triton logits+postprocess."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--ort-local", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    import torch
    from models import _load_ram_plus_model, inference_ram_batch
    from ram import get_transform
    from runtime.ram_triton import logits_to_ram_tags, preprocess_ram_images
    from runtime.triton_client import TritonClient
    from tools.export_ram_onnx import RamLogitsWrapper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Prefer CPU load if GPU is full; parity does not need GPU speed.
    load_device = "cpu"
    print(f"Loading RAM++ on {load_device}…", flush=True)
    model = _load_ram_plus_model(load_device)
    transform = get_transform(image_size=384)
    images = [
        Image.new("RGB", (384, 384), color=(30 + i * 15, 60 + i * 10, 90))
        for i in range(args.batch)
    ]
    pixel = preprocess_ram_images(transform, images)
    tensor = torch.from_numpy(pixel)

    with torch.inference_mode():
        native_en, native_cn = inference_ram_batch(tensor.to(load_device), model)
        wrap = RamLogitsWrapper(model)
        native_logits = wrap(tensor.to(load_device)).detach().cpu().numpy()

    compared = False

    if args.ort_local:
        import onnxruntime as ort

        onnx_path = ROOT / "triton_models" / "ram_plus" / "1" / "model.onnx"
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_logits = sess.run(None, {"image": pixel})[0]
        max_abs = float(np.max(np.abs(native_logits - ort_logits)))
        print(f"ORT local max|Δlogits|={max_abs:.6g}")
        if max_abs > args.atol:
            print("FAIL ORT logits")
            return 1
        ort_en, ort_cn = logits_to_ram_tags(ort_logits)
        if ort_en != list(native_en):
            print("FAIL ORT tags", ort_en[:1], native_en[:1])
            return 1
        print("ORT tag parity OK")
        compared = True

    client = TritonClient(infer_timeout_sec=180.0)
    if client.is_live() and client.model_ready("ram_plus"):
        tri_logits = client.infer_fp32("ram_plus", "image", "logits", pixel)
        max_abs = float(np.max(np.abs(native_logits - tri_logits)))
        print(f"Triton max|Δlogits|={max_abs:.6g}")
        tri_en, tri_cn = logits_to_ram_tags(tri_logits)
        if list(native_en) != tri_en:
            print("FAIL tag mismatch")
            print("native", native_en[0][:120])
            print("triton", tri_en[0][:120])
            return 1
        print(f"Triton parity OK batch={args.batch} sample_en={native_en[0][:80]!r}")
        compared = True
    else:
        print("Triton ram_plus not ready — skip")

    if not compared:
        print("FAIL: nothing compared")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
