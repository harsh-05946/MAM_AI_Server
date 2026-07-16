#!/usr/bin/env python3
"""Unit tests for offline-mode flag wiring and asset preflight."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_offline_flag_derives_hf_env() -> None:
    from runtime.offline_mode import apply_offline_env_defaults, is_offline_mode

    for key in ("AI_OFFLINE_MODE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "ALLOW_HF_FALLBACK"):
        os.environ.pop(key, None)
    os.environ["AI_OFFLINE_MODE"] = "true"
    apply_offline_env_defaults()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["ALLOW_HF_FALLBACK"] == "0"
    assert is_offline_mode() is True


def test_preflight_detects_missing_assets() -> None:
    from models import preflight_local_assets

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        os.environ["AI_OFFLINE_MODE"] = "true"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["ALLOW_HF_FALLBACK"] = "0"
        os.environ["LOCAL_INSIGHTFACE_ROOT"] = str(root / "insightface")
        os.environ["LOCAL_RAM_PLUS_WEIGHTS"] = str(root / "missing_ram.pth")
        os.environ["LOCAL_EMOTION_DIR"] = str(root / "missing_emotion")
        os.environ["LOCAL_BLIP_DIR"] = str(root / "missing_scene")
        os.environ["LOCAL_EMBED_DIR"] = str(root / "missing_embed")
        os.environ["LOCAL_SARVAM_DIR"] = str(root / "missing_sarvam")
        os.environ["LOCAL_QWEN_VL_DIR"] = str(root / "missing_qwen")
        raised = False
        try:
            preflight_local_assets(strict=True)
        except RuntimeError:
            raised = True
        assert raised is True


if __name__ == "__main__":
    test_offline_flag_derives_hf_env()
    test_preflight_detects_missing_assets()
    print("offline mode unit tests passed")
