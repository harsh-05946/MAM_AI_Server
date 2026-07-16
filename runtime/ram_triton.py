"""RAM++ Triton helpers: logits infer + tag string postprocess."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from runtime.paths import PROJECT_ROOT
from runtime.triton_client import TritonClient, get_triton_client
from runtime.triton_router import TRITON_MODEL_NAMES

_META_DIR = PROJECT_ROOT / "triton_models" / "ram_plus"


@lru_cache(maxsize=1)
def _load_ram_meta() -> dict[str, Any]:
    meta_path = _META_DIR / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    tag_en = (_META_DIR / "tag_list.txt").read_text(encoding="utf-8").splitlines()
    tag_cn = (_META_DIR / "tag_list_chinese.txt").read_text(encoding="utf-8").splitlines()
    thr_path = _META_DIR / "class_threshold.npy"
    if thr_path.is_file():
        thr = np.load(thr_path).astype(np.float32)
    else:
        thr = np.full((int(meta.get("num_class", len(tag_en))),), 0.68, dtype=np.float32)
    return {
        "meta": meta,
        "tag_list": np.asarray(tag_en, dtype=object),
        "tag_list_chinese": np.asarray(tag_cn, dtype=object),
        "class_threshold": thr,
        "delete_tag_index": [int(x) for x in meta.get("delete_tag_index", [])],
    }


def logits_to_ram_tags(logits: np.ndarray) -> tuple[list[str], list[str]]:
    """Match RAM_plus.generate_tag postprocess from raw logits."""
    data = _load_ram_meta()
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    logits = logits.astype(np.float32, copy=False)
    thr = data["class_threshold"]
    if thr.shape[0] != logits.shape[1]:
        raise RuntimeError(f"threshold dim {thr.shape[0]} != logits {logits.shape[1]}")
    # sigmoid > threshold
    probs = 1.0 / (1.0 + np.exp(-logits))
    targets = (probs > thr.reshape(1, -1)).astype(np.float32)
    for idx in data["delete_tag_index"]:
        if 0 <= idx < targets.shape[1]:
            targets[:, idx] = 0.0
    tags_en: list[str] = []
    tags_cn: list[str] = []
    tag_list = data["tag_list"]
    tag_cn = data["tag_list_chinese"]
    for b in range(targets.shape[0]):
        index = np.argwhere(targets[b] == 1.0).reshape(-1)
        token = tag_list[index]
        token_cn = tag_cn[index] if len(tag_cn) == len(tag_list) else tag_list[index]
        tags_en.append(" | ".join(map(str, token.tolist())))
        tags_cn.append(" | ".join(map(str, token_cn.tolist())))
    return tags_en, tags_cn


def preprocess_ram_images(transform: Any, images: list[Image.Image]) -> np.ndarray:
    stacked = [transform(img) for img in images]
    # transform returns torch tensors
    import torch

    arr = torch.stack(stacked).detach().cpu().numpy().astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)


def infer_ram_triton(
    image_nchw: np.ndarray,
    *,
    client: Optional[TritonClient] = None,
) -> tuple[list[str], list[str]]:
    c = client or get_triton_client()
    logits = c.infer_fp32(
        TRITON_MODEL_NAMES["ram_plus"],
        input_name="image",
        output_name="logits",
        array=image_nchw,
    )
    return logits_to_ram_tags(logits)
