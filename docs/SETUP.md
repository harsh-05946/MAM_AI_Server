# Setup guide

This document walks a new developer through installing dependencies, preparing models, and starting the MAM AI Server.

## What you are running

| Component | File | Default port | Role |
|-----------|------|--------------|------|
| Main inference API | `main.py` | 8000 (single) or 8001/8002 (A/B) | Loads ML models and serves `/process/*` endpoints |
| Router API | `router_api.py` | 9000 | Queues requests and forwards to backend A or B |
| NGINX (optional) | `nginx_main_api.conf` | 8000 | Reverse proxy in front of the router |

Most API paths are documented in [API_REQUESTS.md](API_REQUESTS.md).

---

## Requirements

### Hardware

- **NVIDIA GPU with CUDA** — required for face recognition, emotion, scene, RAM++, Sarvam, and Qwen models.
- Enough GPU memory for all main models loaded at once (plan for several GB; exact usage depends on models and batch size).

### Software

| Requirement | Version / notes |
|-------------|-----------------|
| OS | Linux (Ubuntu 22.04+ recommended) |
| Python | **3.12** (`requires-python` in `pyproject.toml`) |
| CUDA driver | Compatible with **PyTorch CUDA 11.8** wheels (`torch==2.7.1+cu118`) |
| Git | To clone the repository |
| NGINX | Only if you use the NGINX + router stack (`nginx` package) |

Install system packages on Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip git nginx build-essential
```

For GPU support, install NVIDIA drivers and a CUDA toolkit/driver stack that works with CUDA 11.8 PyTorch wheels. Do **not** commit `.deb` installer files to this repo; install CUDA/keyring packages on the host via `apt` or your image build.

---

## 1. Clone the repository

```bash
git clone <your-repo-url> MAM_AI_Server
cd MAM_AI_Server
```

**Note:** The provided `start_*.sh` scripts use the path `/home/ubuntu/MAM_AI_Server`. If you clone elsewhere, either:

- run `uvicorn` manually (see below), or
- edit the `cd` and `source` paths in the shell scripts.

---

## 2. Create a virtual environment

Using **uv** (recommended — matches `pyproject.toml` and `uv.lock`):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
uv venv .venv --python 3.12
source .venv/bin/activate
uv sync
```

Using **pip**:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

PyTorch is pinned to CUDA 11.8 builds. If `pip install` fails to resolve `torch==2.7.1+cu118`, install PyTorch from the official index first:

```bash
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Verify GPU visibility:

```bash
python3 -c "import torch; print('cuda:', torch.cuda.is_available())"
```

---

## 3. Download model weights (first run)

On first startup the app can pull models from Hugging Face. To prefetch everything into local folders (recommended for repeatable deploys):

```bash
source .venv/bin/activate
python3 bootstrap_local_models.py --service all
```

This creates:

- `models-local/main/{emotion,scene,embed,sarvam,qwen_vl}/`
- `pretrained/ram_plus_swin_large_14m.pth`

For offline/air-gapped use, see [LOCAL_MODELS.md](../LOCAL_MODELS.md) and set:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Optional warmup (loads models once, prints status, unloads):

```bash
python3 warmup_main_models.py
```

---

## 4. Start the application

Activate the venv before any start command:

```bash
source .venv/bin/activate
```

### Option A — Single instance (simplest)

One process, direct access on port **8000**:

```bash
bash ./start_main_api.sh
```

Or manually:

```bash
export CUDA_VISIBLE_DEVICES=0
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

First startup loads all models and can take several minutes.

### Option B — Router + two backends + NGINX (production-like)

Use this when you want queued routing across two identical inference processes.

Terminal 1 — backend A (port 8001):

```bash
bash ./start_main_api_a.sh
```

Terminal 2 — backend B (port 8002):

```bash
bash ./start_main_api_b.sh
```

Terminal 3 — router (port 9000):

```bash
bash ./start_router_api.sh
```

Terminal 4 — NGINX (public port 8000 → router 9000):

```bash
bash ./start_nginx_main_api.sh
```

Clients call **`http://<host>:8000`** (NGINX). Router details: [router.md](router.md).

### Option C — 2× batch limits (throughput testing)

Same URLs as Option A, but doubles per-request batch caps (`FACE_BATCH_MAX`, `EMOTION_BATCH_MAX`, etc.):

```bash
bash ./start_main_api_2x_batch.sh
```

See [API_REQUESTS.md](API_REQUESTS.md) for default and doubled limits.

---

## 5. Verify the server

Health check:

```bash
curl -sS http://127.0.0.1:8000/health | python3 -m json.tool
```

For the A/B stack without NGINX, check a backend directly:

```bash
curl -sS http://127.0.0.1:8001/health | python3 -m json.tool
curl -sS http://127.0.0.1:9000/health | python3 -m json.tool   # router
```

Smoke test batch endpoints (requires running main API):

```bash
python3 tools/smoke_batch_endpoints.py --base-url http://127.0.0.1:8001
```

---

## 6. Stop services

```bash
bash ./stop_main_api.sh          # single main API
bash ./stop_main_api_a.sh        # backend A
bash ./stop_main_api_b.sh        # backend B
bash ./stop_router_api.sh        # router
bash ./stop_nginx_main_api.sh    # NGINX
```

---

## Common environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | `0` in start scripts | Which GPU to use |
| `PORT` | `8000` / `8001` / `8002` / `9000` | Listen port (per script) |
| `INSTANCE_NAME` | `main` / `main-a` / `main-b` | Shown in `/health` |
| `HF_HUB_OFFLINE` | `0` | Hugging Face offline mode |
| `TRANSFORMERS_OFFLINE` | `0` | Transformers offline mode |
| `BATCHING_ENABLED` | `true` | Server-side micro-batching for concurrent single-item requests |
| `FACE_BATCH_MAX`, `EMOTION_BATCH_MAX`, … | see [API_REQUESTS.md](API_REQUESTS.md) | Max items per HTTP batch request |

---

## Troubleshooting

| Symptom | Things to check |
|---------|-----------------|
| `cuda: False` | NVIDIA driver installed; correct `CUDA_VISIBLE_DEVICES`; PyTorch CUDA wheel installed |
| Face endpoint returns 503 | InsightFace requires CUDA; see `/health` for loaded models |
| Slow or OOM on large batches | Lower batch env vars or use default (non-2×) start script |
| Models fail to download | Network/HF token; run `bootstrap_local_models.py` with internet |
| Port already in use | `ss -tlnp \| grep 8000` and stop the existing process |
| NGINX fails to start | `nginx` installed; port 8000 free; logs in `nginx-run/error.log` |

Logs (when using start scripts with logging configured):

- `main_api.log`, `main_api_a.log`, `main_api_b.log`
- `nginx-run/access.log`, `nginx-run/error.log`

---

## Related docs

- [API_REQUESTS.md](API_REQUESTS.md) — request/response shapes and batch limits
- [router.md](router.md) — router queue, round-robin, and path pinning
- [LOCAL_MODELS.md](../LOCAL_MODELS.md) — local model paths and offline mode
