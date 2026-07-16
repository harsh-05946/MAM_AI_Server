#!/usr/bin/env python3
"""Unit checks for Triton flags / router (no Triton server required)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _clear_flags() -> None:
    for key in (
        "USE_TRITON_EMOTION",
        "USE_TRITON_RAM",
        "USE_TRITON_SCENE",
        "USE_TRITON_EMBED",
        "USE_TRITON_FACE",
    ):
        os.environ.pop(key, None)


def test_defaults_are_native():
    _clear_flags()
    from runtime.triton_flags import any_triton_flag_enabled, triton_flag_for_model
    from runtime.triton_router import resolve_backend, require_native_or_raise, triton_runtime_status

    assert any_triton_flag_enabled() is False
    for model in ("emotion", "ram_plus", "scene", "embed", "insightface"):
        assert triton_flag_for_model(model) is False
        assert resolve_backend(model) == "native"
        require_native_or_raise(model)

    status = triton_runtime_status()
    assert status["phase"] == 3
    assert status["any_flag_enabled"] is False
    assert all(v == "native" for v in status["backends"].values())
    print("defaults native ok")


def test_flag_on_without_migration_is_pending():
    _clear_flags()
    os.environ["USE_TRITON_RAM"] = "true"
    from runtime.triton_router import resolve_backend, require_native_or_raise

    assert resolve_backend("ram_plus") == "triton_pending"
    assert resolve_backend("emotion") == "native"
    try:
        require_native_or_raise("ram_plus")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "not ready" in str(exc).lower() or "USE_TRITON" in str(exc) or "migrated" in str(exc).lower()
    finally:
        os.environ.pop("USE_TRITON_RAM", None)
    print("pending path ok")


def test_emotion_is_migrated_but_defaults_native():
    _clear_flags()
    from runtime.triton_router import is_migrated, resolve_backend

    assert is_migrated("emotion") is True
    assert resolve_backend("emotion") == "native"
    print("emotion migrated flag-off ok")



def test_qwen_sarvam_have_no_triton_flag():
    from runtime.triton_flags import TRITON_FLAG_ENV, triton_flag_for_model

    assert "qwen_vl" not in TRITON_FLAG_ENV
    assert "sarvam" not in TRITON_FLAG_ENV
    assert triton_flag_for_model("qwen_vl") is False
    assert triton_flag_for_model("sarvam") is False
    print("generative excluded ok")


def main() -> int:
    test_defaults_are_native()
    test_flag_on_without_migration_is_pending()
    test_emotion_is_migrated_but_defaults_native()
    test_qwen_sarvam_have_no_triton_flag()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
