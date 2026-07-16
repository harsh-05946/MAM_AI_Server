#!/usr/bin/env python3
"""Quick local proof: preflight + model load under offline env."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("AI_OFFLINE_MODE", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ALLOW_HF_FALLBACK", "0")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import MODELS, get_runtime_model_status, load_main_models, preflight_local_assets, unload_models


def main() -> int:
    preflight = preflight_local_assets(strict=True)
    load_main_models()
    payload = {
        "proof": "offline_model_load_ok",
        "loaded_models": sorted(MODELS.keys()),
        "preflight": preflight,
        "runtime_status": get_runtime_model_status(),
    }
    print(json.dumps(payload, indent=2))
    unload_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
