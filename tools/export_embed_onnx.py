#!/usr/bin/env python3
"""Export SentenceTransformer all-MiniLM-L6-v2 to ONNX for Triton (Phase 3)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "triton_models" / "embed"
VERSION_DIR = OUT_DIR / "1"


class MeanPoolEmbed(nn.Module):
    def __init__(self, auto_model: nn.Module):
        super().__init__()
        self.auto_model = auto_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.auto_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        token_embeddings = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return F.normalize(summed / counts, p=2, dim=1)


def _config_pbtxt(dim: int = 384, max_batch: int = 32, max_seq: int = 128) -> str:
    return f"""name: "embed"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch}
default_model_filename: "model.onnx"
input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ {max_seq} ]
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ {max_seq} ]
  }}
]
output [
  {{
    name: "sentence_embedding"
    data_type: TYPE_FP32
    dims: [ {dim} ]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-seq", type=int, default=128)
    parser.add_argument("--max-batch", type=int, default=32)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    from models import _resolve_model_source

    source, _ = _resolve_model_source("embed", service="main")
    print(f"Loading SentenceTransformer from {source}", flush=True)
    st = SentenceTransformer(source, device=args.device)
    auto_model = st[0].auto_model
    try:
        auto_model.set_attn_implementation("eager")
    except Exception:
        auto_model.config._attn_implementation = "eager"

    wrapped = MeanPoolEmbed(auto_model).to(args.device).eval()
    tokenizer = st.tokenizer
    texts = ["hello world", "parity check", "batch three"]
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_seq,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded["attention_mask"].to(args.device)

    with torch.inference_mode():
        ref = wrapped(input_ids, attention_mask).detach().cpu().numpy()
        native = st.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    cos = float(np.mean(np.sum(native * ref, axis=1)))
    print(f"ST.encode vs wrap mean cosine={cos:.6f}")

    VERSION_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = VERSION_DIR / "model.onnx"
    batch_dim = torch.export.Dim("batch", min=1, max=args.max_batch)
    print(f"Exporting ONNX (dynamo, dynamic batch) -> {onnx_path}", flush=True)
    torch.onnx.export(
        wrapped,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["sentence_embedding"],
        dynamo=True,
        dynamic_shapes={
            "input_ids": {0: batch_dim},
            "attention_mask": {0: batch_dim},
        },
    )

    dim = int(ref.shape[-1])
    (OUT_DIR / "meta.json").write_text(
        json.dumps(
            {
                "model_id": source,
                "embedding_dim": dim,
                "max_seq_length": args.max_seq,
                "normalize": True,
                "pooling": "mean",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "config.pbtxt").write_text(
        _config_pbtxt(dim=dim, max_batch=args.max_batch, max_seq=args.max_seq),
        encoding="utf-8",
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(
        None,
        {
            "input_ids": input_ids.detach().cpu().numpy(),
            "attention_mask": attention_mask.detach().cpu().numpy(),
        },
    )[0]
    max_abs = float(np.max(np.abs(out - ref)))
    print(f"ORT vs wrap max|Δ|={max_abs:.6g}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
