"""Emotion inference helpers shared by native and Triton paths."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from runtime.paths import PROJECT_ROOT
from runtime.triton_client import TritonClient, get_triton_client
from runtime.triton_router import TRITON_MODEL_NAMES

_LABELS_PATH = PROJECT_ROOT / "triton_models" / "emotion" / "labels.json"
_ID2LABEL: Optional[dict[int, str]] = None


def load_emotion_id2label(path: Path = _LABELS_PATH) -> dict[int, str]:
    global _ID2LABEL
    if _ID2LABEL is not None:
        return _ID2LABEL
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        _ID2LABEL = {int(k): v for k, v in raw.get("id2label", {}).items()}
        return _ID2LABEL
    # Fallback labels used by trpakov/vit-face-expression
    _ID2LABEL = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprise",
    }
    return _ID2LABEL


def logits_to_emotion_results(logits: np.ndarray, id2label: Optional[dict[int, str]] = None) -> list[dict[str, Any]]:
    """Convert [N, C] logits to [{emotion, confidence}, ...]."""
    labels = id2label or load_emotion_id2label()
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    # softmax
    x = logits.astype(np.float64, copy=False)
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    out: list[dict[str, Any]] = []
    for row in probs:
        idx = int(np.argmax(row))
        out.append({"emotion": labels.get(idx, str(idx)), "confidence": float(row[idx])})
    return out


def preprocess_emotion_pixel_values(processor: Any, images: list[Image.Image]) -> np.ndarray:
    """Run HF AutoImageProcessor → float32 NCHW numpy."""
    inputs = processor(images=images, return_tensors="pt")
    arr = inputs["pixel_values"].detach().cpu().numpy().astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)


def infer_emotion_triton(
    pixel_values: np.ndarray,
    *,
    client: Optional[TritonClient] = None,
    id2label: Optional[dict[int, str]] = None,
) -> list[dict[str, Any]]:
    c = client or get_triton_client()
    model_name = TRITON_MODEL_NAMES["emotion"]
    logits = c.infer_fp32(
        model_name,
        input_name="pixel_values",
        output_name="logits",
        array=pixel_values,
    )
    return logits_to_emotion_results(logits, id2label=id2label)
