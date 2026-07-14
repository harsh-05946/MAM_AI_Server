# MAM AI Server

Single-GPU FastAPI inference API for the MAM Processing Server. One Uvicorn process (`main:app`) behind NGINX on port **9000**.

Validated operating point: **up to 2 medias in parallel** with Processing-side `AI_SERVER_MAX_INFLIGHT_BATCHES=2` (or 4). Uncapped concurrency on one GPU causes timeouts.

## Stack

| Piece | Detail |
| --- | --- |
| App | `main.py` / `models.py` |
| Package manager | **uv** (`pyproject.toml` + `uv.lock`) |
| Torch | 2.7.1 + CUDA **12.6** wheels |
| Public URL | `http://<AI_SERVER_IP>:9000` |
| Backend | `127.0.0.1:9001` |
| InsightFace | CUDA EP required (`REQUIRE_FACE_CUDA=true`) |

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

Install NGINX site from `deploy/nginx/ai-server-test.conf` (listen **9000** → upstream **9001**).

## Run

```bash
bash scripts/start_single_ai_optimized.sh
bash scripts/status_single_ai_optimized.sh
bash scripts/stop_single_ai_optimized.sh
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

Live report: `runtime_reports/current/AI_THROUGHPUT_REPORT.md` (gitignored).

## Processing Server

- Base URL: `http://<AI_SERVER_IP>:9000`
- Set **`AI_SERVER_MAX_INFLIGHT_BATCHES=2`** (try `4` only if AI stays healthy)
- Keep client/proxy timeouts aligned with NGINX (**900s**)

## Layout

```
main.py models.py
runtime/ monitoring/ reporting/
scripts/start|stop|status_single_ai_optimized.sh
deploy/nginx/ai-server-test.conf
tools/validate_providers.py tools/smoke_batch_endpoints.py
docs/SETUP.md docs/API_REQUESTS.md
```

## Docs

- [docs/SETUP.md](docs/SETUP.md) — install and ops
- [docs/API_REQUESTS.md](docs/API_REQUESTS.md) — endpoint contracts
