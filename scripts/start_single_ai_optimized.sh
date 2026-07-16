#!/bin/bash
# Start one long-lived main:app behind NGINX :9000.
# Prefers .venv with CUDA 12 cublas; falls back to .venv-cu12 if needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_UV="$PROJECT_DIR/.venv"
VENV_CU12="$PROJECT_DIR/.venv-cu12"
APP_MODULE="main:app"
LOG_DIR="$PROJECT_DIR/logs"
RUN_DIR="$PROJECT_DIR/run"
REPORT_DIR="$PROJECT_DIR/runtime_reports/current"
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-900}"
HEALTH_POLL_SEC="${HEALTH_POLL_SEC:-2}"
NAME="ai-optimized"
PORT=9001

cd "$PROJECT_DIR" || exit 1

if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.env"
  set +a
fi

is_true() {
  local v="${1:-}"
  [[ "${v,,}" == "true" || "$v" == "1" || "${v,,}" == "yes" || "${v,,}" == "on" ]]
}

_has_cublas12() {
  local root="$1"
  [[ -f "$root/lib/python3.12/site-packages/nvidia/cublas/lib/libcublasLt.so.12" ]] \
    || [[ -n "$(find "$root/lib" -name 'libcublasLt.so.12' 2>/dev/null | head -n 1)" ]]
}

if [[ -x "$VENV_UV/bin/uvicorn" ]] && _has_cublas12 "$VENV_UV"; then
  VENV_PATH="$VENV_UV"
  echo "Using uv venv (CUDA 12): $VENV_PATH"
elif [[ -x "$VENV_CU12/bin/uvicorn" ]] && _has_cublas12 "$VENV_CU12"; then
  VENV_PATH="$VENV_CU12"
  echo "Using .venv-cu12 (CUDA 12): $VENV_PATH"
elif [[ -x "$VENV_UV/bin/uvicorn" ]]; then
  VENV_PATH="$VENV_UV"
  echo "WARNING: $VENV_PATH lacks libcublasLt.so.12 — InsightFace CUDA may fail. Fix with: uv sync" >&2
else
  echo "ERROR: no venv with uvicorn. Run: uv venv .venv --python 3.12 && uv sync" >&2
  exit 1
fi

UVICORN_BIN="$VENV_PATH/bin/uvicorn"
PYTHON_BIN="$VENV_PATH/bin/python"

mkdir -p "$LOG_DIR" "$RUN_DIR" "$REPORT_DIR"

export AI_OFFLINE_MODE="${AI_OFFLINE_MODE:-false}"
if is_true "$AI_OFFLINE_MODE"; then
  # Single-flag offline mode: force local-only model resolution.
  export HF_HUB_OFFLINE="1"
  export TRANSFORMERS_OFFLINE="1"
  export ALLOW_HF_FALLBACK="0"
else
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
  export ALLOW_HF_FALLBACK="${ALLOW_HF_FALLBACK:-1}"
fi

# Fail fast in offline mode so operators download/export assets before start.
if is_true "$AI_OFFLINE_MODE"; then
  echo "AI_OFFLINE_MODE=true — verifying local models / Triton assets before start..."
  if ! "$PYTHON_BIN" "$PROJECT_DIR/bootstrap_local_models.py" --verify-only; then
    echo "ERROR: Offline start blocked — required local assets are missing." >&2
    echo "  1) Go online and run: uv run python bootstrap_local_models.py" >&2
    echo "  2) If USE_TRITON_*=true: docker pull \${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:24.08-py3}" >&2
    echo "  3) Export Triton ONNX as needed: tools/export_emotion_onnx.py / export_embed_onnx.py / export_ram_onnx.py" >&2
    echo "  4) Confirm: uv run python bootstrap_local_models.py --verify-only" >&2
    echo "  5) Then start again with AI_OFFLINE_MODE=true" >&2
    exit 1
  fi
  echo "Offline asset verification passed."
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export REQUIRE_FACE_CUDA="${REQUIRE_FACE_CUDA:-true}"
export GPU_EXECUTION_SLOTS="${GPU_EXECUTION_SLOTS:-1}"
export AI_CAPACITY_PROFILE="${AI_CAPACITY_PROFILE:-l40s-scalable-lanes-v1}"
export MAX_PARALLEL_MEDIA="${MAX_PARALLEL_MEDIA:-2}"
export PER_MEDIA_AI_INFLIGHT="${PER_MEDIA_AI_INFLIGHT:-8}"
export PER_MEDIA_MODEL_INFLIGHT="${PER_MEDIA_MODEL_INFLIGHT:-1}"
# AI accepted queue / GPU slots are independent of MAX_PARALLEL_MEDIA.
export AI_ACCEPTED_QUEUE_LIMIT="${AI_ACCEPTED_QUEUE_LIMIT:-6}"
export AI_MAX_ACCEPTED_INFERENCE_REQUESTS="${AI_MAX_ACCEPTED_INFERENCE_REQUESTS:-$AI_ACCEPTED_QUEUE_LIMIT}"
export AI_INTERNAL_MAX_WAITING_REQUESTS="${AI_INTERNAL_MAX_WAITING_REQUESTS:-$AI_ACCEPTED_QUEUE_LIMIT}"
export MAX_WAITING_GPU_REQUESTS="${MAX_WAITING_GPU_REQUESTS:-$AI_INTERNAL_MAX_WAITING_REQUESTS}"
export AI_INTERNAL_REJECT_WHEN_FULL="${AI_INTERNAL_REJECT_WHEN_FULL:-true}"
export GPU_QUEUE_REJECT_WHEN_FULL="${GPU_QUEUE_REJECT_WHEN_FULL:-$AI_INTERNAL_REJECT_WHEN_FULL}"
export AI_OVERLOAD_RETRY_AFTER_SECONDS="${AI_OVERLOAD_RETRY_AFTER_SECONDS:-10}"
export AI_MODEL_OVERLOAD_RETRY_AFTER_SECONDS="${AI_MODEL_OVERLOAD_RETRY_AFTER_SECONDS:-10}"
export AI_MAX_ACTIVE_GENERATIVE_REQUESTS="${AI_MAX_ACTIVE_GENERATIVE_REQUESTS:-1}"
export AI_MAX_ACTIVE_QWEN_REQUESTS="${AI_MAX_ACTIVE_QWEN_REQUESTS:-1}"
export AI_MAX_ACTIVE_SARVAM_REQUESTS="${AI_MAX_ACTIVE_SARVAM_REQUESTS:-1}"
export AI_COMBINED_GENERATIVE_LIMIT_ENABLED="${AI_COMBINED_GENERATIVE_LIMIT_ENABLED:-false}"
export AI_GPU_SLOTS_VISUAL="${AI_GPU_SLOTS_VISUAL:-1}"
export AI_GPU_SLOTS_QWEN="${AI_GPU_SLOTS_QWEN:-1}"
export AI_GPU_SLOTS_SARVAM="${AI_GPU_SLOTS_SARVAM:-1}"
export AI_VISUAL_EXECUTION_SLOTS="${AI_VISUAL_EXECUTION_SLOTS:-$AI_GPU_SLOTS_VISUAL}"
export AI_QWEN_EXECUTION_SLOTS="${AI_QWEN_EXECUTION_SLOTS:-$AI_GPU_SLOTS_QWEN}"
export AI_SARVAM_EXECUTION_SLOTS="${AI_SARVAM_EXECUTION_SLOTS:-$AI_GPU_SLOTS_SARVAM}"
export AI_ENABLE_VISUAL_QWEN_OVERLAP="${AI_ENABLE_VISUAL_QWEN_OVERLAP:-false}"
export AI_ENABLE_VISUAL_SARVAM_OVERLAP="${AI_ENABLE_VISUAL_SARVAM_OVERLAP:-false}"
export AI_ENABLE_QWEN_SARVAM_OVERLAP="${AI_ENABLE_QWEN_SARVAM_OVERLAP:-true}"
export AI_GRACEFUL_DRAIN_SECONDS="${AI_GRACEFUL_DRAIN_SECONDS:-30}"
export AI_FORCE_SHUTDOWN_SECONDS="${AI_FORCE_SHUTDOWN_SECONDS:-90}"
export AI_REQUIRED_MODELS="${AI_REQUIRED_MODELS:-face,emotion,scene,ram_plus,embeddings}"
export AI_OPTIONAL_MODELS="${AI_OPTIONAL_MODELS:-qwen,sarvam_translation}"
export AI_MODEL_CANARY_INTERVAL_SECONDS="${AI_MODEL_CANARY_INTERVAL_SECONDS:-300}"
export AI_STARTUP_WARMUP="${AI_STARTUP_WARMUP:-true}"
export ENABLE_CUDNN_BENCHMARK="${ENABLE_CUDNN_BENCHMARK:-false}"
export INSTANCE_NAME="${INSTANCE_NAME:-ai-01}"

# Phase 2 Triton flags — all off. Public API stays native FastAPI.
export TRITON_HTTP_URL="${TRITON_HTTP_URL:-http://127.0.0.1:8001}"
export TRITON_GRPC_URL="${TRITON_GRPC_URL:-127.0.0.1:8002}"
export USE_TRITON_EMOTION="${USE_TRITON_EMOTION:-true}"
export USE_TRITON_RAM="${USE_TRITON_RAM:-false}"
export USE_TRITON_SCENE="${USE_TRITON_SCENE:-false}"
export USE_TRITON_EMBED="${USE_TRITON_EMBED:-true}"
export USE_TRITON_FACE="${USE_TRITON_FACE:-false}"
export GPU_FAIRNESS="${GPU_FAIRNESS:-true}"
export GPU_VISUAL_SHARE="${GPU_VISUAL_SHARE:-0.6}"
# Coalesce same-prompt / same-lang singles into denser GPU batches.
export BATCH_WAIT_MS_QWEN="${BATCH_WAIT_MS_QWEN:-40}"
export BATCH_WAIT_MS_SARVAM="${BATCH_WAIT_MS_SARVAM:-40}"

# When any USE_TRITON_*=true, Triton must be live or /ready stays 503.
_need_triton=0
for _f in USE_TRITON_EMOTION USE_TRITON_RAM USE_TRITON_SCENE USE_TRITON_EMBED USE_TRITON_FACE; do
  _v="${!_f:-false}"
  if [[ "${_v,,}" == "true" || "${_v}" == "1" || "${_v,,}" == "yes" || "${_v,,}" == "on" ]]; then
    _need_triton=1
    break
  fi
done
if [[ "$_need_triton" -eq 1 ]]; then
  echo "Triton required by USE_TRITON_* flags — ensuring scripts/start_triton.sh ..."
  bash "$SCRIPT_DIR/start_triton.sh" || {
    echo "ERROR: Triton failed to start but USE_TRITON_*=true. /ready will stay 503." >&2
    echo "  Fix: bash scripts/start_triton.sh   OR set USE_TRITON_EMOTION=false USE_TRITON_EMBED=false" >&2
    exit 1
  }
fi

# Match 2x HTTP batch caps used by Processing Server.
export FACE_BATCH_MAX=$(( ${FACE_BATCH_MAX:-8} * 2 ))
export EMOTION_BATCH_MAX=$(( ${EMOTION_BATCH_MAX:-32} * 2 ))
export SCENE_BATCH_MAX=$(( ${SCENE_BATCH_MAX:-8} * 2 ))
export RAM_BATCH_MAX=$(( ${RAM_BATCH_MAX:-8} * 2 ))
export QWEN_BATCH_MAX=$(( ${QWEN_BATCH_MAX:-10} * 2 ))
export SARVAM_BATCH_MAX=$(( ${SARVAM_BATCH_MAX:-10} * 2 ))
export EMBED_BATCH_MAX="${EMBED_BATCH_MAX:-32}"

if [[ -d "$VENV_PATH/lib" ]]; then
  EXTRA_LIB="$(find "$VENV_PATH/lib" -type d \( -path '*/nvidia/*/lib' -o -path '*/onnxruntime/capi' \) 2>/dev/null | paste -sd: - || true)"
  if [[ -n "${EXTRA_LIB:-}" ]]; then
    export LD_LIBRARY_PATH="${EXTRA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi
echo "LD_LIBRARY_PATH nvidia/ort libs: ${EXTRA_LIB:-none}"

SERVER_RUN_ID="server_$(date +%Y%m%d_%H%M%S)"
export SERVER_RUN_ID
echo "$SERVER_RUN_ID" >"$RUN_DIR/server_run_id"

PID_FILE="$RUN_DIR/${NAME}.pid"
LOG_FILE="$LOG_DIR/${NAME}.log"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ERROR: $NAME already running (PID $old_pid). Stop it first." >&2
    exit 1
  fi
  rm -f "$PID_FILE"
fi

if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "ERROR: Port $PORT is already in use." >&2
  ss -ltnp 2>/dev/null | grep ":${PORT}" || true
  exit 1
fi

echo "Starting $NAME on 127.0.0.1:$PORT (SERVER_RUN_ID=$SERVER_RUN_ID)..."
CUDA_VISIBLE_DEVICES=0 \
nohup "$UVICORN_BIN" "$APP_MODULE" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --workers 1 \
  --timeout-keep-alive 120 \
  >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
APP_PID="$(cat "$PID_FILE")"
echo "Started app PID $APP_PID (log: $LOG_FILE)"

wait_for_ready() {
  local url="http://127.0.0.1:${PORT}/ready"
  local live="http://127.0.0.1:${PORT}/health"
  local elapsed=0
  local code
  echo "Waiting for readiness at $url (timeout ${HEALTH_TIMEOUT_SEC}s; /health is liveness-only)..."
  while (( elapsed < HEALTH_TIMEOUT_SEC )); do
    code="$(curl -s -o /tmp/ai_ready_wait.json -w '%{http_code}' --connect-timeout 2 --max-time 5 "$url" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      echo "Ready (HTTP 200)."
      return 0
    fi
    live_code="$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 1 --max-time 2 "$live" 2>/dev/null || true)"
    reason="$(python3 -c 'import json; print(json.load(open("/tmp/ai_ready_wait.json")).get("reason") or json.load(open("/tmp/ai_ready_wait.json")).get("service_state",""))' 2>/dev/null || echo "?")"
    if [[ "$live_code" == "200" ]]; then
      echo "  alive but not ready yet (ready_http=${code} reason=${reason}) t=${elapsed}s"
    fi
    if ! kill -0 "$APP_PID" 2>/dev/null; then
      echo "ERROR: app exited before ready. Tail of log:" >&2
      tail -n 80 "$LOG_FILE" >&2 || true
      return 1
    fi
    sleep "$HEALTH_POLL_SEC"
    elapsed=$((elapsed + HEALTH_POLL_SEC))
  done
  echo "ERROR: readiness timeout (last reason=${reason:-unknown})" >&2
  python3 -m json.tool </tmp/ai_ready_wait.json 2>/dev/null | head -n 40 >&2 || true
  tail -n 80 "$LOG_FILE" >&2 || true
  return 1
}

wait_for_ready

start_helper() {
  local helper_name="$1"
  local cmd="$2"
  local helper_log="$LOG_DIR/${helper_name}.log"
  local helper_pid="$RUN_DIR/${helper_name}.pid"
  if [[ -f "$helper_pid" ]]; then
    local hp
    hp="$(cat "$helper_pid" 2>/dev/null || true)"
    if [[ -n "${hp:-}" ]] && kill -0 "$hp" 2>/dev/null; then
      echo "$helper_name already running (PID $hp)"
      return 0
    fi
  fi
  nohup bash -lc "$cmd" >"$helper_log" 2>&1 &
  echo $! >"$helper_pid"
  echo "Started $helper_name PID $(cat "$helper_pid") (log: $helper_log)"
}

start_helper "gpu-monitor" "cd '$PROJECT_DIR' && PYTHONPATH='$PROJECT_DIR' '$PYTHON_BIN' -m monitoring.gpu_monitor"
start_helper "host-monitor" "cd '$PROJECT_DIR' && PYTHONPATH='$PROJECT_DIR' '$PYTHON_BIN' -m monitoring.host_monitor"
start_helper "throughput-reporter" "cd '$PROJECT_DIR' && PYTHONPATH='$PROJECT_DIR' '$PYTHON_BIN' reporting/throughput_reporter.py --watch --interval 15"

echo
echo "AI instance ready on 127.0.0.1:$PORT (NGINX public :9000)"
echo "Report: $REPORT_DIR/AI_THROUGHPUT_REPORT.md"
echo "Validated load: ~2 medias; set Processing AI_SERVER_MAX_INFLIGHT_BATCHES=2"
echo "Campaign example:"
echo "  $PYTHON_BIN reporting/manage_run.py start --name dual_media --expected-media-count 2 --expected-ai-frames 1800 --source-media-hours 2"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi
