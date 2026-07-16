# MAM AI Server

Single-GPU FastAPI inference API for the MAM Processing Server. One Uvicorn process (`main:app`) behind NGINX on port **9000**.

Validated operating point: **up to 2 medias in parallel** with Processing-side `AI_SERVER_MAX_INFLIGHT_BATCHES=2`. Uncapped concurrency on one GPU causes timeouts.

## Stack

| Piece | Detail |
| --- | --- |
| App | `main.py` / `models.py` |
| Package manager | **uv** (`pyproject.toml` + `uv.lock`) |
| Torch | 2.7.1 + CUDA **12.6** wheels |
| Public URL | `http://<AI_SERVER_IP>:9000` |
| Backend | `127.0.0.1:9001` (single worker) |
| InsightFace | CUDA EP required (`REQUIRE_FACE_CUDA=true`) |
| AI admission | Profile `l40s-scalable-lanes-v1`: fleet from `MAX_PARALLEL_MEDIA`; AI queue=`AI_ACCEPTED_QUEUE_LIMIT` (`GET /internal/capacity`) |
| Probes | `GET /health` liveness; `GET /ready` cached readiness (200/503) |

Do **not** run multiple Uvicorn copies of this API on one GPU.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if needed
cd MAM_AI_Server
uv venv .venv --python 3.12
uv sync
```

Bootstrap / warm models (once):

```bash
uv run python bootstrap_local_models.py
uv run python warmup_main_models.py
uv run python tools/validate_providers.py
```

Offline-first proof path:

```bash
uv run python bootstrap_local_models.py          # download while online
uv run python bootstrap_local_models.py --verify-only
uv run python tools/prove_offline_ready.py
# then set AI_OFFLINE_MODE=true in .env and start normally
# start script blocks with download/export instructions if assets are missing
```

Triton ONNX weights under `triton_models/**/1/` are gitignored — export them locally with `tools/export_*_onnx.py`; do not commit or push them.

Install NGINX site from `deploy/nginx/ai-server-test.conf` (listen **9000** → upstream **9001**).

## Run

```bash
bash scripts/start_single_ai_optimized.sh
bash scripts/status_single_ai_optimized.sh
bash scripts/drain_ai_server.sh          # preferred graceful stop
# or: bash scripts/stop_single_ai_optimized.sh --force
```

Measured campaign (optional):

```bash
uv run python reporting/manage_run.py start \
  --name dual_media \
  --expected-media-count 2 \
  --expected-ai-frames 1800 \
  --source-media-hours 2 \
  --target-media-hours-per-hour 3

# …send traffic from Processing Server…

uv run python reporting/manage_run.py finalize
```

Live report: `runtime_reports/current/AI_THROUGHPUT_REPORT.md` (gitignored). Report admission wait separately from GPU inference time.

## Processing Server

- Base URL: `http://<AI_SERVER_IP>:9000`
- Set **`AI_SERVER_MAX_INFLIGHT_BATCHES=2`** (primary overload control)
- Keep client/proxy timeouts aligned with NGINX (**900s**)
- AI keeps a small safety queue (4 waiters); do not expect it to absorb uncapped Celery floods

## RAM++ batching

RAM++ uses true stacked-tensor batching via `generate_tag` (full batch outputs). Fallback to per-image should be rare (&lt;1%). Metrics: `ram_batch_success` / `ram_batch_fallback`.

## Next (Triton Phase 3 + generative)

**Live on Triton:** Emotion + Embeddings. **Native FastAPI:** Qwen/Sarvam with GPU fairness (`GPU_FAIRNESS=true`, visual share 0.6) and key-aware micro-batching. **RAM++/Scene/Face** still native — come back later. See [docs/TRITON.md](docs/TRITON.md) and [docs/SETUP.md](docs/SETUP.md).

## Layout

```
main.py models.py
runtime/ (incl. triton_flags|client|router)
triton_models/          # Phase 2 empty repo
monitoring/ reporting/
scripts/start|stop|status_single_ai_optimized.sh
scripts/start|stop|status_triton.sh
deploy/nginx/ai-server-test.conf
tools/
docs/SETUP.md docs/API_REQUESTS.md docs/TRITON.md
```

## Docs

- [docs/SETUP.md](docs/SETUP.md) — install and ops
- [docs/API_REQUESTS.md](docs/API_REQUESTS.md) — endpoint contracts
- [docs/TRITON.md](docs/TRITON.md) — Triton Phase 2 scaffold
