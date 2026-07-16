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

Offline preparation/proof (while online first, then local-only):

```bash
# Provision + inventory report (downloads only missing assets).
uv run python bootstrap_local_models.py
# Validate inventory without downloading.
uv run python bootstrap_local_models.py --verify-only
# Proof load with strict local-only flags.
uv run python tools/prove_offline_ready.py
```

InsightFace must report CUDA providers first. Failures here usually mean missing CUDA 12 libs in the venv (`nvidia-*` packages from Torch cu126 wheels + `onnxruntime-gpu`).

## NGINX

Copy `deploy/nginx/ai-server-test.conf` into sites-available/enabled, then:

```bash
sudo nginx -t && sudo systemctl reload nginx
```

Public: **9000** → upstream `127.0.0.1:9001`. Buffering/timeouts unchanged (1 GiB body, 900 s read/send).

## Start / stop

Scripts resolve the project directory from their own location (not a hardcoded home path). Use **one** Uvicorn worker only.

```bash
bash scripts/start_single_ai_optimized.sh
bash scripts/status_single_ai_optimized.sh
curl -sS http://127.0.0.1:9000/health | python3 -m json.tool   # liveness: {"status":"alive"}
curl -sS -w '\n%{http_code}\n' http://127.0.0.1:9000/ready     # readiness: 200/503
bash scripts/drain_ai_server.sh                                 # graceful drain then stop
# or: bash scripts/stop_single_ai_optimized.sh --force
```

Environment highlights set by the start script:

## Capacity profile (`l40s-scalable-lanes-v1`)

Processing validates `GET /internal/capacity`. Defaults:

| Env | Default | Role |
| --- | --- | --- |
| `REQUIRE_FACE_CUDA` | `true` | Fail if InsightFace not on CUDA EP |
| `AI_OFFLINE_MODE` | `false` | Single offline switch (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `ALLOW_HF_FALLBACK=0`) |
| `AI_CAPACITY_PROFILE` | `l40s-scalable-lanes-v1` | Shared contract name |
| `MAX_PARALLEL_MEDIA` | `2` | Scales fleet admission |
| `PER_MEDIA_AI_INFLIGHT` | `8` | per_media; fleet_global = 8 × media |
| `PER_MEDIA_MODEL_INFLIGHT` | `1` | per_media model; fleet_model = 1 × media |
| `AI_ACCEPTED_QUEUE_LIMIT` | `6` | AI accepted queue (not fleet_global) |
| `AI_GPU_SLOTS_VISUAL/QWEN/SARVAM` | `1` | GPU execution slots (do not auto-scale) |
| `AI_INTERNAL_REJECT_WHEN_FULL` | `true` | **429** when accepted queue / model fleet full |
| `AI_OVERLOAD_RETRY_AFTER_SECONDS` | `10` | `Retry-After` for 429/503 |
| `AI_ENABLE_QWEN_SARVAM_OVERLAP` | `true` | Concurrent Qwen+Sarvam GPU when slots allow |
| `AI_ENABLE_VISUAL_*_OVERLAP` | `false` | Visual vs generative physical overlap |
| `AI_REQUIRED_MODELS` | `face,emotion,scene,ram_plus,embeddings` | Required for `/ready` |
| `AI_OPTIONAL_MODELS` | `qwen,sarvam_translation` | Missing → DEGRADED when accepting |
| `LOCAL_INSIGHTFACE_ROOT` | `~/.insightface` | Pre-cached InsightFace root for offline runs |
| HTTP batch caps | face/scene/object **16**, emotion **64**, qwen/sarvam **20**, embed **32** | see capacity `admission.models.*.batch` |

With media=2: Processing should use `fleet_global=16`, each model `fleet=2`, `per_media=8`. AI still accepts at most **6** queued inferences and **1** GPU slot per class.

Rollback: `AI_CAPACITY_PROFILE=l40s-six-lane-v1` (profile JSON in `configs/capacity/`).

Offline run mode:
- Pre-stage assets with `bootstrap_local_models.py` and ensure `bootstrap_local_models.py --verify-only` passes.
- If any `USE_TRITON_*=true`, also `docker pull` the Triton image and export ONNX via `tools/export_*_onnx.py` (weights under `triton_models/**/1/` are local-only / gitignored).
- Set only `AI_OFFLINE_MODE=true` in `.env`.
- Start normally (`bash scripts/start_single_ai_optimized.sh`). Offline start runs `--verify-only` first and exits with download/export instructions if anything is missing.
- If startup fails, error messages include missing local paths (HF model dirs, RAM++ weights, InsightFace cache, Triton image / ONNX).

Processing: validate capacity at startup; set Redis/`AI_SERVER_*` inflight from `admission.*`; honor **429** + `Retry-After`.

## Campaign reporting

```bash
uv run python reporting/manage_run.py start --name run_name \
  --expected-media-count 2 --expected-ai-frames 1800 \
  --source-media-hours 2 --target-media-hours-per-hour 3
uv run python reporting/manage_run.py finalize
```

Artifacts land in `runtime_reports/` (gitignored). Timings separate admission wait, GPU queue wait, and inference.

## Processing Server guidance

Match `GET /internal/capacity` (`l40s-scalable-lanes-v1`):

| Setting | Recommendation (media=2) |
| --- | --- |
| AI URL | `http://<AI_SERVER_IP>:9000` |
| global / fleet inflight | **`admission.fleet_global`** (=16) |
| per_media_inflight | **`admission.per_media`** (=8) |
| per_media_model | **`admission.models.*.per_media`** (=1) |
| Client timeout | Prefer **900s** |
| On 429 | Honor `Retry-After` (10) + `code` |

Prefer HTTP `/batch`; avoid singles fan-out under capacity pressure.

## RAM++

Object-detection batch uses stacked tensors and full `generate_tag` outputs (one tag string per image). False `tag_shape_mismatch` fallbacks from upstream `inference_ram`’s `tags[0]` strip are fixed. Expect `ram_batch_fallback` **&lt;1%**.

## Generative (Qwen / Sarvam)

Native FastAPI only (no Triton). Single GPU slot with fairness so long Qwen/Sarvam jobs do not starve visuals:

| Env | Default | Meaning |
| --- | --- | --- |
| `GPU_FAIRNESS` | `true` | Visual vs generative deficit scheduling |
| `GPU_VISUAL_SHARE` | `0.6` | Target GPU-time share for visual (+embed) |
| `AI_MAX_ACTIVE_GENERATIVE_REQUESTS` | `1` | At most one Qwen/Sarvam job holding generative slot |
| `BATCH_WAIT_MS_QWEN` | `40` | Micro-batch window for same-prompt coalescing |
| `BATCH_WAIT_MS_SARVAM` | `40` | Micro-batch window for same-`target_lang` |

Single-item `/process/caption/qwen` and `/process/translation/sarvam` use key-aware micro-batching (prefer matching prompt / lang within the wait window). `/batch` endpoints stay direct multi-item calls (unchanged schemas).

```bash
uv run python tools/test_generative_fairness_unit.py
```

**Phase 3 (Emotion):** ONNX in `triton_models/emotion/`, Triton on `:8001/:8002`, parity via `tools/parity_emotion_triton.py`. Default `USE_TRITON_EMOTION=false`. Next: RAM++ → Scene → Embed. Face is Phase 4. See [TRITON.md](TRITON.md).

## Campaign reports

`runtime_reports/current/` aggregates JSONL across server restarts. Before a measured dual-media run:

```bash
uv run python reporting/manage_run.py finalize   # if a campaign is still RUNNING
# optionally archive/clear current/ so the report is not polluted by older servers
uv run python reporting/manage_run.py start --name dual_media \
  --expected-media-count 2 --expected-ai-frames 1800 \
  --source-media-hours 2 --target-media-hours-per-hour 3
```

Filter events by `server_run_id` when comparing RAM fallback rates (Phase 1 fix vs older `tag_shape_mismatch` storms).

## Smoke checks

```bash
uv run python tools/smoke_batch_endpoints.py   # if models are warm
uv run python tools/test_runtime_unit.py       # no GPU traffic
uv run python tools/test_ram_batch_unit.py     # RAM++ batch mapping / scheduler defaults
uv run python tools/test_release1_contract_unit.py  # readiness / 429 / cancel
uv run python tools/test_triton_router_unit.py # Phase 2 flags default off
uv run python tools/smoke_ram_batch_gpu.py     # RAM++ stacked N=8/16 on GPU
bash scripts/status_triton.sh                  # Triton optional; AI does not require it
```

## API contracts

See [API_REQUESTS.md](API_REQUESTS.md).
