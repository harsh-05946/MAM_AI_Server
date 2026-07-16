#!/usr/bin/env python3
"""Parity: native PyTorch emotion vs Triton (or local ORT) on the same pixel_values.

Usage:
  # Requires Triton with emotion loaded (USE_TRITON not required — calls Triton directly)
  python tools/parity_emotion_triton.py

  # Also compare ONNX ORT CPU without Triton:
  python tools/parity_emotion_triton.py --ort-local
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--ort-local", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    from models import _resolve_model_source, _hf_kwargs
    from runtime.emotion_triton import logits_to_emotion_results, preprocess_emotion_pixel_values
    from runtime.triton_client import TritonClient

    source, _ = _resolve_model_source("emotion", service="main")
    print(f"Native model: {source}")
    processor = AutoImageProcessor.from_pretrained(source, **_hf_kwargs())
    model = AutoModelForImageClassification.from_pretrained(source, **_hf_kwargs())
    model.eval()

    images = [
        Image.new("RGB", (224, 224), color=(40 + i * 20, 80, 120 + i * 5))
        for i in range(args.batch)
    ]
    pixel_values = preprocess_emotion_pixel_values(processor, images)
    with torch.inference_mode():
        native_logits = model(pixel_values=torch.from_numpy(pixel_values)).logits.cpu().numpy()

    native_res = logits_to_emotion_results(
        native_logits, id2label={int(k): v for k, v in model.config.id2label.items()}
    )

    compared = False

    if args.ort_local:
        import onnxruntime as ort

        onnx_path = ROOT / "triton_models" / "emotion" / "1" / "model.onnx"
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_logits = sess.run(None, {"pixel_values": pixel_values})[0]
        max_abs = float(np.max(np.abs(native_logits - ort_logits)))
        print(f"ORT local max|Δlogits|={max_abs:.6g}")
        if not np.allclose(native_logits, ort_logits, atol=args.atol, rtol=args.rtol):
            print("FAIL: ORT vs native logits mismatch")
            return 1
        compared = True

    client = TritonClient()
    if client.is_live() and client.model_ready("emotion"):
        tri_logits = client.infer_fp32(
            "emotion",
            input_name="pixel_values",
            output_name="logits",
            array=pixel_values,
        )
        max_abs = float(np.max(np.abs(native_logits - tri_logits)))
        print(f"Triton max|Δlogits|={max_abs:.6g}")
        if not np.allclose(native_logits, tri_logits, atol=args.atol, rtol=args.rtol):
            print("FAIL: Triton vs native logits mismatch")
            print("native sample", native_res[0])
            print("triton sample", logits_to_emotion_results(tri_logits)[0])
            return 1
        # Label agreement
        tri_res = logits_to_emotion_results(
            tri_logits, id2label={int(k): v for k, v in model.config.id2label.items()}
        )
        for i, (a, b) in enumerate(zip(native_res, tri_res)):
            if a["emotion"] != b["emotion"]:
                print(f"FAIL: label mismatch at {i}: {a} vs {b}")
                return 1
        print(f"Triton parity OK batch={args.batch} sample={native_res[0]}")
        compared = True
    else:
        print("Triton not ready — skip Triton parity (start scripts/start_triton.sh)")

    if not compared:
        print("FAIL: nothing compared (pass --ort-local and/or start Triton)")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
