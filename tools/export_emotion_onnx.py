#!/usr/bin/env python3
"""Export trpakov/vit-face-expression to ONNX for Triton (Phase 3 — Emotion).

Writes:
  triton_models/emotion/1/model.onnx
  triton_models/emotion/config.pbtxt
  triton_models/emotion/labels.json

Usage:
  python tools/export_emotion_onnx.py
  python tools/export_emotion_onnx.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "triton_models" / "emotion"
VERSION_DIR = OUT_DIR / "1"


def _config_pbtxt(max_batch: int = 64, num_labels: int = 7) -> str:
    # With max_batch_size > 0 Triton prepends the batch dim; do not include it here.
    return f"""name: "emotion"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch}
default_model_filename: "model.onnx"
input [
  {{
    name: "pixel_values"
    data_type: TYPE_FP32
    dims: [ 3, -1, -1 ]
  }}
]
output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ {num_labels} ]
  }}
]
instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
dynamic_batching {{
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 2000
}}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Export emotion ViT to ONNX for Triton")
    parser.add_argument("--device", default="cpu", help="Export device (cpu recommended)")
    parser.add_argument("--source", default=None, help="HF id or local dir (default MODEL_IDS emotion)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max-batch", type=int, default=64)
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    from models import _resolve_model_source, _hf_kwargs

    source = args.source
    if not source:
        source, _ = _resolve_model_source("emotion", service="main")
    print(f"Loading emotion model from: {source}")

    processor = AutoImageProcessor.from_pretrained(source, **_hf_kwargs())
    model = AutoModelForImageClassification.from_pretrained(source, **_hf_kwargs())
    model.eval()
    model.to(args.device)

    # Dummy batch=1 image; processor sets size (typically 224).
    from PIL import Image

    img = Image.new("RGB", (224, 224), color=(120, 80, 40))
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(args.device)

    class EmotionWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m(pixel_values=pixel_values).logits

    wrapped = EmotionWrapper(model)
    VERSION_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = VERSION_DIR / "model.onnx"

    print(f"Exporting ONNX -> {onnx_path}")
    torch.onnx.export(
        wrapped,
        (pixel_values,),
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    labels = {
        "id2label": {str(k): v for k, v in model.config.id2label.items()},
        "label2id": {k: int(v) for k, v in model.config.label2id.items()},
        "source": source,
        "num_labels": int(model.config.num_labels),
    }
    (OUT_DIR / "labels.json").write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "config.pbtxt").write_text(
        _config_pbtxt(args.max_batch, num_labels=int(model.config.num_labels)),
        encoding="utf-8",
    )

    # Quick ORT sanity check (CPU).
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"pixel_values": pixel_values.detach().cpu().numpy()})[0]
        assert out.shape[-1] == model.config.num_labels, out.shape
        print(f"ORT smoke OK logits shape={out.shape}")
    except Exception as exc:
        print(f"WARN: ORT smoke skipped/failed: {exc}")

    print("Done.")
    print(f"  {onnx_path}")
    print(f"  {OUT_DIR / 'config.pbtxt'}")
    print(f"  {OUT_DIR / 'labels.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
