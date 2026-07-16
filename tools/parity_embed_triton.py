#!/usr/bin/env python3
"""Parity: SentenceTransformer.encode vs Triton embed ONNX."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    from models import _resolve_model_source
    from runtime.embed_triton import infer_embed_triton, tokenize_for_embed
    from runtime.triton_client import TritonClient

    source, _ = _resolve_model_source("embed", service="main")
    st = SentenceTransformer(source, device="cpu")
    texts = [f"parity sentence number {i} with some words" for i in range(args.batch)]
    native = st.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    client = TritonClient()
    if not (client.is_live() and client.model_ready("embed")):
        print("FAIL: Triton embed not ready")
        return 1

    ids, mask = tokenize_for_embed(st.tokenizer, texts)
    tri = np.asarray(infer_embed_triton(ids, mask, client=client), dtype=np.float32)
    cos = np.sum(native * tri, axis=1)
    print(f"mean cosine={float(cos.mean()):.6f} min={float(cos.min()):.6f}")
    if float(cos.min()) < 0.999:
        print("FAIL cosine too low")
        print("native0", native[0][:8])
        print("triton0", tri[0][:8])
        return 1
    print(f"PASS batch={args.batch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
