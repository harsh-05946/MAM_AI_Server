# Local/Offline Model Setup

This app now supports local-first model loading with optional offline mode.

## Model split

- Main service (`main.py`): `face`, `emotion`, `scene`, `ram_plus`, `embed`, `sarvam`, `qwen_vl`
- VibeVoice/ASR service has been removed.

## Bootstrap local model folders

Run once on a machine with internet:

```bash
python3 bootstrap_local_models.py --service all
```

This creates:

- `models-local/main/emotion`
- `models-local/main/scene`
- `models-local/main/embed`
- `models-local/main/sarvam`
- `models-local/main/qwen_vl`
- `pretrained/ram_plus_swin_large_14m.pth`

## Environment variables (optional overrides)

- `LOCAL_EMOTION_DIR`
- `LOCAL_BLIP_DIR`
- `LOCAL_EMBED_DIR`
- `LOCAL_SARVAM_DIR`
- `LOCAL_QWEN_VL_DIR`
- `LOCAL_RAM_PLUS_WEIGHTS`

If these are unset, default local paths are used under `models-local/...`.

## Offline mode

Set both for strict offline behavior:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

In offline mode:

- Transformers loaders use `local_files_only=True`
- Hub fallback is disabled
- Missing local files will raise startup errors

## Warmup script

- Main app warmup:
  ```bash
  python3 warmup_main_models.py
  ```
The script loads models, prints status JSON, and unloads models.
