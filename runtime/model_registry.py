from __future__ import annotations

from typing import Optional

# Canonical model keys used in events and reports.
ENDPOINT_MODEL = {
    "/process/face": "insightface",
    "/process/face/batch": "insightface",
    "/process/emotion": "emotion",
    "/process/emotion/batch": "emotion",
    "/process/scene": "scene",
    "/process/scene/batch": "scene",
    "/process/object-detection": "ram_plus",
    "/process/object-detection/batch": "ram_plus",
    "/process/caption/qwen": "qwen_vl",
    "/process/caption/qwen/batch": "qwen_vl",
    "/process/translation/sarvam": "sarvam",
    "/process/translation/sarvam/batch": "sarvam",
    "/process/embeddings": "embed",
}

QUEUE_CLASS = {
    "insightface": "visual",
    "emotion": "visual",
    "scene": "visual",
    "ram_plus": "visual",
    "qwen_vl": "generative",
    "sarvam": "generative",
    "embed": "embedding",
}

HTTP_BATCH_LIMIT_ENV = {
    "insightface": ("FACE_BATCH_MAX", 16),
    "emotion": ("EMOTION_BATCH_MAX", 64),
    "scene": ("SCENE_BATCH_MAX", 16),
    "ram_plus": ("RAM_BATCH_MAX", 16),
    "qwen_vl": ("QWEN_BATCH_MAX", 20),
    "sarvam": ("SARVAM_BATCH_MAX", 20),
    "embed": ("EMBED_BATCH_MAX", 32),
}

MICRO_BATCH_LIMIT_ENV = {
    "insightface": ("BATCH_MAX_FACE", 8),
    "emotion": ("BATCH_MAX_EMOTION", 16),
    "scene": ("BATCH_MAX_SCENE", 8),
    "ram_plus": ("BATCH_MAX_RAM_PLUS", 8),
    "qwen_vl": ("BATCH_MAX_QWEN", 10),
    "sarvam": ("BATCH_MAX_SARVAM", 10),
    "embed": ("BATCH_MAX_EMBED", 32),
}


def model_for_endpoint(path: str) -> Optional[str]:
    if path in ENDPOINT_MODEL:
        return ENDPOINT_MODEL[path]
    for prefix, model in ENDPOINT_MODEL.items():
        if path.startswith(prefix):
            return model
    return None


def queue_class_for_model(model: str) -> str:
    return QUEUE_CLASS.get(model, "other")
