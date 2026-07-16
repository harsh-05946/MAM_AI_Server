# Triton model repository (Phase 2 skeleton)

Empty / placeholder model tree for a local Triton Inference Server.

| Host port | Triton container port | Protocol |
| --- | --- | --- |
| **8001** | 8000 | HTTP |
| **8002** | 8001 | gRPC |

Metrics (optional): map host `8003` → container `8002` in the start script.

## Layout

```
triton_models/
  emotion/          # Phase 3
  ram_plus/         # Phase 3
  scene/            # Phase 3
  embed/            # Phase 3
  insightface/      # Phase 4 (will split into detector/recognizer/…)
```

Each directory currently holds only a `config.pbtxt.example`. Copy to `config.pbtxt`
and add `1/model.onnx` (or TensorRT plan) when that model is migrated. Do **not**
enable `USE_TRITON_*` until a real artifact exists and parity tests pass.

**Do not commit weights.** `triton_models/**/1/` (ONNX / `.onnx.data` / plans) is
gitignored. Export locally with `tools/export_*_onnx.py` on each machine.

## Start / stop

```bash
bash scripts/start_triton.sh    # requires Docker
bash scripts/status_triton.sh
bash scripts/stop_triton.sh
```

AI FastAPI (`:9001`) does **not** require Triton when all `USE_TRITON_*=false` (default).
