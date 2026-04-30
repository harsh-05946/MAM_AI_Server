#!/usr/bin/env python3
"""Warm up main-app models (non-ASR)."""

import json
import logging

from models import MODELS, get_runtime_model_status, load_main_models, unload_models


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Starting main model warmup")
    load_main_models()
    payload = {
        "service": "main",
        "loaded_models": list(MODELS.keys()),
        "runtime_status": get_runtime_model_status(),
    }
    print(json.dumps(payload, indent=2))
    unload_models()
    logging.info("Main model warmup complete")


if __name__ == "__main__":
    main()
