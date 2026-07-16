#!/usr/bin/env python3
"""Provision and verify local model assets for online/offline operation."""

import argparse
import os
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


def _insightface_root() -> Path:
    raw = os.getenv("LOCAL_INSIGHTFACE_ROOT")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".insightface"


def _triton_image_cached(image: str) -> bool:
    try:
        import subprocess

        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _env_true(name: str, default: str = "false") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _triton_required() -> bool:
    names = (
        "USE_TRITON_EMOTION",
        "USE_TRITON_RAM",
        "USE_TRITON_SCENE",
        "USE_TRITON_EMBED",
        "USE_TRITON_FACE",
    )
    return any(_env_true(name) for name in names)


def _triton_model_repo() -> Path:
    raw = os.getenv("TRITON_MODEL_REPO")
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).parent / "triton_models"


def _verify_triton_onnx_assets() -> list[str]:
    """Return human-readable missing Triton local artifacts for enabled flags."""
    repo = _triton_model_repo()
    missing: list[str] = []
    checks = (
        ("USE_TRITON_EMOTION", "emotion", "python tools/export_emotion_onnx.py"),
        ("USE_TRITON_EMBED", "embed", "python tools/export_embed_onnx.py"),
        ("USE_TRITON_RAM", "ram_plus", "python tools/export_ram_onnx.py"),
        ("USE_TRITON_SCENE", "scene", "export scene ONNX then place under triton_models/scene/1/"),
        ("USE_TRITON_FACE", "insightface", "export InsightFace ONNX then place under triton_models/insightface/1/"),
    )
    for flag, model_name, hint in checks:
        if not _env_true(flag):
            continue
        config = repo / model_name / "config.pbtxt"
        onnx = repo / model_name / "1" / "model.onnx"
        if not config.is_file() or not onnx.is_file():
            missing.append(f"{model_name} (need {config.name} + 1/model.onnx; {hint})")
    return missing


def _download_hf_model(model_id: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    if any(local_dir.iterdir()):
        print(f"HF model already present: {local_dir}")
        return
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


def _verify_assets(root_dir: Path, triton_image: str) -> int:
    status = 0
    print("Verifying local asset inventory:")
    for model_key in MAIN_MODELS:
        model_dir = root_dir / "main" / model_key
        present = model_dir.exists() and any(model_dir.iterdir())
        print(f" - {model_key:8s}: {'OK' if present else 'MISSING'}  path={model_dir}")
        if not present:
            status = 1

    ram_weights = Path(__file__).parent / "pretrained" / "ram_plus_swin_large_14m.pth"
    print(f" - ram_plus_weights: {'OK' if ram_weights.exists() else 'MISSING'}  path={ram_weights}")
    if not ram_weights.exists():
        status = 1

    insightface_dir = _insightface_root() / "models" / "buffalo_l"
    required_face = ("det_10g.onnx", "2d106det.onnx", "genderage.onnx", "w600k_r50.onnx")
    missing_face = [name for name in required_face if not (insightface_dir / name).exists()]
    print(
        f" - insightface_buffalo_l: {'OK' if not missing_face else 'MISSING'}  path={insightface_dir}"
    )
    if missing_face:
        print(f"   missing files: {', '.join(missing_face)}")
        status = 1

    if _triton_required():
        triton_cached = _triton_image_cached(triton_image)
        print(f" - triton_image_cached: {'OK' if triton_cached else 'MISSING'}  image={triton_image}")
        if not triton_cached:
            status = 1
            print(f"   Pre-pull while online: docker pull {triton_image}")
        missing_onnx = _verify_triton_onnx_assets()
        if missing_onnx:
            status = 1
            print(" - triton_onnx_assets: MISSING")
            for item in missing_onnx:
                print(f"   - {item}")
        else:
            print(" - triton_onnx_assets: OK")
    else:
        print(" - triton_image_cached: SKIPPED  (all USE_TRITON_*=false)")
        print(" - triton_onnx_assets: SKIPPED  (all USE_TRITON_*=false)")
    if status != 0:
        print(
            "\nOffline assets incomplete. While online, download/export first:\n"
            "  uv run python bootstrap_local_models.py\n"
            "  docker pull <TRITON_IMAGE>   # if any USE_TRITON_*=true\n"
            "  uv run python tools/export_emotion_onnx.py   # as needed\n"
            "  uv run python tools/export_embed_onnx.py\n"
            "  uv run python tools/export_ram_onnx.py\n"
            "Then re-run: uv run python bootstrap_local_models.py --verify-only"
        )
    return status


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
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download; only verify required local artifacts",
    )
    parser.add_argument(
        "--triton-image",
        default=os.getenv("TRITON_IMAGE", "nvcr.io/nvidia/tritonserver:24.08-py3"),
        help="Triton image tag used for offline cache verification",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser()
    root_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        raise SystemExit(_verify_assets(root_dir, args.triton_image))

    if args.service in {"main", "all"}:
        for model_key, model_id in MAIN_MODELS.items():
            _download_hf_model(model_id, root_dir / "main" / model_key)
        _download_ram_weights(Path(__file__).parent)

    verify_rc = _verify_assets(root_dir, args.triton_image)
    mode = "offline-ready" if verify_rc == 0 else "partially provisioned"
    print(f"Bootstrap complete ({mode}).")


if __name__ == "__main__":
    main()
