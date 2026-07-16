# Triton (Phase 3)

| Service | Port |
| --- | ---: |
| FastAPI AI | 9001 |
| Triton HTTP | **8001** |
| Triton gRPC | **8002** |

## Live (flags on by default in start script)

| Model | Flag | Status |
| --- | --- | --- |
| emotion | `USE_TRITON_EMOTION=true` | ONNX on Triton — live |
| embed | `USE_TRITON_EMBED=true` | ONNX on Triton — live |
| ram_plus | `USE_TRITON_RAM=false` | ONNX tag parity failed; config disabled |
| scene | `USE_TRITON_SCENE=false` | not migrated |
| insightface | `USE_TRITON_FACE=false` | Phase 4 |

```bash
bash scripts/start_triton.sh
bash scripts/start_single_ai_optimized.sh
curl -sS http://127.0.0.1:9001/ready | python3 -m json.tool   # readiness (also /internal/runtime for detail)
```

Parity:

```bash
uv run python tools/parity_emotion_triton.py --batch 8
uv run python tools/parity_embed_triton.py --batch 8
```

Export:

```bash
uv run python tools/export_emotion_onnx.py
uv run python tools/export_embed_onnx.py
```

## RAM++ deferred

`tools/export_ram_onnx.py` produces logits ONNX, but tag strings diverge from native. Config: `triton_models/ram_plus/config.pbtxt.disabled_until_parity`. Next fix: Triton **Python backend** with PyTorch `generate_tag`.

## Next

1. Scene (BLIP) — expect similar export hard case; may need Python backend
2. RAM Python backend
3. Face true-batch (Phase 4)
