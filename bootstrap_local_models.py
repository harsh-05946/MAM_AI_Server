#!/usr/bin/env python3
"""Download model assets into reusable local directories."""

import argparse
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

RAM_PLUS_URL = "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth"

MAIN_MODELS = {
    "emotion": "trpakov/vit-face-expression",
    "scene": "Salesforce/blip-image-captioning-large",
    "embed": "sentence-transformers/all-MiniLM-L6-v2",
    "sarvam": "sarvamai/sarvam-translate",
    "qwen_vl": "Qwen/Qwen2.5-VL-3B-Instruct",
}


def _download_hf_model(model_id: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} -> {local_dir}")
    snapshot_download(repo_id=model_id, local_dir=str(local_dir), local_dir_use_symlinks=False)


def _download_ram_weights(root_dir: Path) -> None:
    pretrained_dir = root_dir / "pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    weight_path = pretrained_dir / "ram_plus_swin_large_14m.pth"
    if weight_path.exists():
        print(f"RAM++ weights already present: {weight_path}")
        return
    print(f"Downloading RAM++ weights -> {weight_path}")
    urllib.request.urlretrieve(RAM_PLUS_URL, str(weight_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap local model directories")
    parser.add_argument(
        "--service",
        choices=["main", "all"],
        default="all",
        help="Which service model set to download",
    )
    parser.add_argument(
        "--root-dir",
        default=str(Path(__file__).parent / "models-local"),
        help="Root directory where model folders are saved",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser()
    root_dir.mkdir(parents=True, exist_ok=True)

    if args.service in {"main", "all"}:
        for model_key, model_id in MAIN_MODELS.items():
            _download_hf_model(model_id, root_dir / "main" / model_key)
        _download_ram_weights(Path(__file__).parent)

    print("Bootstrap complete.")


if __name__ == "__main__":
    main()
