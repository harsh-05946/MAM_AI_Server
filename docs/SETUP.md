# Setup guide

Single-GPU MAM AI Server: one `main:app` Uvicorn behind NGINX `:9000`.

## Requirements

- NVIDIA GPU + driver (CUDA 12-compatible; PyTorch wheels are **cu126**)
- Python **3.12**
- [uv](https://docs.astral.sh/uv/)
- NGINX for the public `:9000` frontend

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv git nginx build-essential
```

## Install with uv

```bash
cd MAM_AI_Server
uv venv .venv --python 3.12
uv sync
```

All dependencies come from `pyproject.toml` / `uv.lock`. There is no separate `requirements.txt`.

## Models

```bash
uv run python bootstrap_local_models.py
uv run python warmup_main_models.py
REQUIRE_FACE_CUDA=true uv run python tools/validate_providers.py
```

InsightFace must report CUDA providers first. Failures here usually mean missing CUDA 12 libs in the venv (`nvidia-*` packages from Torch cu126 wheels + `onnxruntime-gpu`).

## NGINX

Copy `deploy/nginx/ai-server-test.conf` into sites-available/enabled, then:

```bash
sudo nginx -t && sudo systemctl reload nginx
```

Public: **9000** → upstream `127.0.0.1:9001`. Buffering/timeouts unchanged (1 GiB body, 900 s read/send).

## Start / stop

Scripts resolve the project directory from their own location (not a hardcoded home path).

```bash
bash scripts/start_single_ai_optimized.sh
bash scripts/status_single_ai_optimized.sh
curl -sS http://127.0.0.1:9000/health | python3 -m json.tool
bash scripts/stop_single_ai_optimized.sh
```

Environment highlights set by the start script:

- `REQUIRE_FACE_CUDA=true`
- 2× HTTP batch caps (`FACE_BATCH_MAX`, etc.)
- `GPU_QUEUE_REJECT_WHEN_FULL=false` (wait, do not 429)

## Campaign reporting

```bash
uv run python reporting/manage_run.py start --name run_name \
  --expected-media-count 2 --expected-ai-frames 1800 \
  --source-media-hours 2 --target-media-hours-per-hour 3
uv run python reporting/manage_run.py finalize
```

Artifacts land in `runtime_reports/` (gitignored).

## Processing Server guidance

This AI host is sized for about **2 medias at a time**.

| Setting | Recommendation |
| --- | --- |
| AI URL | `http://<AI_SERVER_IP>:9000` |
| `AI_SERVER_MAX_INFLIGHT_BATCHES` | **2** (raise to 4 only with evidence) |
| Client timeout | Prefer **900s** to match NGINX |

Uncapped batch floods cause queue waits past client timeouts and stuck vision slots on Processing — not an “AI is down” condition.

## Smoke checks

```bash
uv run python tools/smoke_batch_endpoints.py   # if models are warm
uv run python tools/test_runtime_unit.py       # no GPU traffic
```

## API contracts

See [API_REQUESTS.md](API_REQUESTS.md).
