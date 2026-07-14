#!/bin/bash
# Start one long-lived main:app behind NGINX :9000.
# Uses uv-managed .venv by default; falls back to .venv-cu12 if present.

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

if [[ -x "$VENV_UV/bin/uvicorn" ]]; then
  VENV_PATH="$VENV_UV"
  echo "Using uv venv: $VENV_PATH"
elif [[ -x "$VENV_CU12/bin/uvicorn" ]]; then
  VENV_PATH="$VENV_CU12"
  echo "Using legacy .venv-cu12: $VENV_PATH"
else
  echo "ERROR: no venv with uvicorn. Run: uv venv .venv --python 3.12 && uv sync" >&2
  exit 1
fi

UVICORN_BIN="$VENV_PATH/bin/uvicorn"
PYTHON_BIN="$VENV_PATH/bin/python"

mkdir -p "$LOG_DIR" "$RUN_DIR" "$REPORT_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export ALLOW_HF_FALLBACK="${ALLOW_HF_FALLBACK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export REQUIRE_FACE_CUDA="${REQUIRE_FACE_CUDA:-true}"
export GPU_QUEUE_REJECT_WHEN_FULL="${GPU_QUEUE_REJECT_WHEN_FULL:-false}"
export MAX_WAITING_GPU_REQUESTS="${MAX_WAITING_GPU_REQUESTS:-16}"
export ENABLE_CUDNN_BENCHMARK="${ENABLE_CUDNN_BENCHMARK:-false}"
export INSTANCE_NAME="${INSTANCE_NAME:-ai-01}"

# Match 2x HTTP batch caps used by Processing Server.
export FACE_BATCH_MAX=$(( ${FACE_BATCH_MAX:-8} * 2 ))
export EMOTION_BATCH_MAX=$(( ${EMOTION_BATCH_MAX:-32} * 2 ))
export SCENE_BATCH_MAX=$(( ${SCENE_BATCH_MAX:-8} * 2 ))
export RAM_BATCH_MAX=$(( ${RAM_BATCH_MAX:-8} * 2 ))
export QWEN_BATCH_MAX=$(( ${QWEN_BATCH_MAX:-10} * 2 ))
export SARVAM_BATCH_MAX=$(( ${SARVAM_BATCH_MAX:-10} * 2 ))

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

wait_for_health() {
  local url="http://127.0.0.1:${PORT}/health"
  local elapsed=0
  local code
  echo "Waiting for health at $url (timeout ${HEALTH_TIMEOUT_SEC}s)..."
  while (( elapsed < HEALTH_TIMEOUT_SEC )); do
    code="$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 --max-time 5 "$url" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      echo "Healthy (HTTP 200)."
      return 0
    fi
    if ! kill -0 "$APP_PID" 2>/dev/null; then
      echo "ERROR: app exited before healthy. Tail of log:" >&2
      tail -n 80 "$LOG_FILE" >&2 || true
      return 1
    fi
    sleep "$HEALTH_POLL_SEC"
    elapsed=$((elapsed + HEALTH_POLL_SEC))
  done
  echo "ERROR: health timeout" >&2
  tail -n 80 "$LOG_FILE" >&2 || true
  return 1
}

wait_for_health

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
