"""Embedding Triton helpers: tokenize in FastAPI, embeddings from Triton."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Optional

import numpy as np

from runtime.paths import PROJECT_ROOT
from runtime.triton_client import TritonClient, get_triton_client
from runtime.triton_router import TRITON_MODEL_NAMES

_META_PATH = PROJECT_ROOT / "triton_models" / "embed" / "meta.json"


@lru_cache(maxsize=1)
def embed_max_seq_length() -> int:
    if _META_PATH.is_file():
        meta = json.loads(_META_PATH.read_text(encoding="utf-8"))
        return int(meta.get("max_seq_length", 128))
    return 128


def tokenize_for_embed(tokenizer: Any, texts: list[str], *, max_length: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    max_len = max_length or embed_max_seq_length()
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np",
    )
    return (
        np.ascontiguousarray(encoded["input_ids"].astype(np.int64)),
        np.ascontiguousarray(encoded["attention_mask"].astype(np.int64)),
    )


def infer_embed_triton(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    *,
    client: Optional[TritonClient] = None,
) -> list[list[float]]:
    c = client or get_triton_client()
    emb = c.infer_tensors(
        TRITON_MODEL_NAMES["embed"],
        [
            ("input_ids", input_ids, "INT64"),
            ("attention_mask", attention_mask, "INT64"),
        ],
        "sentence_embedding",
        output_dtype=np.float32,
    )
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    return [emb[i].tolist() for i in range(emb.shape[0])]
