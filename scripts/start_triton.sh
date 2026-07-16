#!/bin/bash
# Start local Triton Inference Server (Phase 3: Emotion ONNX when present).
# Host: HTTP :8001, gRPC :8002, metrics :8003.
# Safe alongside FastAPI :9001 when USE_TRITON_*=false.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_REPO="${TRITON_MODEL_REPO:-$PROJECT_DIR/triton_models}"
CONTAINER_NAME="${TRITON_CONTAINER_NAME:-mam-triton}"
IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:24.08-py3}"
HTTP_PORT="${TRITON_HTTP_PORT:-8001}"
GRPC_PORT="${TRITON_GRPC_PORT:-8002}"
METRICS_PORT="${TRITON_METRICS_PORT:-8003}"

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

DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  if sudo docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
  else
    echo "ERROR: Docker daemon not reachable." >&2
    exit 1
  fi
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: Docker is not installed." >&2
  exit 1
fi

if "${DOCKER[@]}" ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  if "${DOCKER[@]}" ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Triton already running: $CONTAINER_NAME"
    "${DOCKER[@]}" ps --filter "name=^${CONTAINER_NAME}$"
    exit 0
  fi
  echo "Removing stopped container $CONTAINER_NAME..."
  "${DOCKER[@]}" rm "$CONTAINER_NAME" >/dev/null
fi

mkdir -p "$MODEL_REPO"

LOAD_ARGS=()
if [[ -f "$MODEL_REPO/emotion/config.pbtxt" && -f "$MODEL_REPO/emotion/1/model.onnx" ]]; then
  LOAD_ARGS+=(--load-model=emotion)
  echo "Will load model: emotion"
else
  echo "NOTE: emotion ONNX missing. Run: python tools/export_emotion_onnx.py"
fi
if [[ -f "$MODEL_REPO/ram_plus/config.pbtxt" && -f "$MODEL_REPO/ram_plus/1/model.onnx" ]]; then
  LOAD_ARGS+=(--load-model=ram_plus)
  echo "Will load model: ram_plus"
elif [[ -f "$MODEL_REPO/ram_plus/config.pbtxt.disabled_until_parity" ]]; then
  echo "NOTE: ram_plus disabled until tag parity (see docs/TRITON.md)"
else
  echo "NOTE: ram_plus ONNX missing. Run: python tools/export_ram_onnx.py"
fi
if [[ -f "$MODEL_REPO/embed/config.pbtxt" && -f "$MODEL_REPO/embed/1/model.onnx" ]]; then
  LOAD_ARGS+=(--load-model=embed)
  echo "Will load model: embed"
else
  echo "NOTE: embed ONNX missing. Run: python tools/export_embed_onnx.py"
fi

echo "Starting Triton ($IMAGE) model-repo=$MODEL_REPO"
echo "  HTTP  127.0.0.1:${HTTP_PORT}"
echo "  gRPC  127.0.0.1:${GRPC_PORT}"
echo "  metrics 127.0.0.1:${METRICS_PORT}"

if is_true "${AI_OFFLINE_MODE:-false}"; then
  if ! "${DOCKER[@]}" image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: AI_OFFLINE_MODE=true and Triton image is not cached locally: $IMAGE" >&2
    echo "  Pre-pull while online: ${DOCKER[*]} pull $IMAGE" >&2
    exit 1
  fi
fi

"${DOCKER[@]}" run -d --name "$CONTAINER_NAME" --gpus all --shm-size=1g \
  -p "${HTTP_PORT}:8000" \
  -p "${GRPC_PORT}:8001" \
  -p "${METRICS_PORT}:8002" \
  -v "${MODEL_REPO}:/models:ro" \
  "$IMAGE" \
  tritonserver \
    --model-repository=/models \
    --model-control-mode=explicit \
    --strict-model-config=false \
    --exit-on-error=false \
    --allow-http=true \
    --allow-grpc=true \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002 \
    "${LOAD_ARGS[@]}"

echo "Waiting for Triton live..."
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:${HTTP_PORT}/v2/health/live" >/dev/null 2>&1; then
    echo "Triton live on http://127.0.0.1:${HTTP_PORT}"
    if [[ ${#LOAD_ARGS[@]} -gt 0 ]]; then
      for model in emotion ram_plus embed; do
        # only wait for models we asked to load
        if printf '%s\n' "${LOAD_ARGS[@]}" | grep -q "load-model=${model}"; then
          ok=0
          for j in $(seq 1 90); do
            if curl -sf "http://127.0.0.1:${HTTP_PORT}/v2/models/${model}/ready" >/dev/null 2>&1; then
              echo "${model} model ready."
              ok=1
              break
            fi
            sleep 2
          done
          if [[ "$ok" -ne 1 ]]; then
            echo "WARNING: Triton live but ${model} not ready. Check: ${DOCKER[*]} logs $CONTAINER_NAME" >&2
            exit 1
          fi
        fi
      done
      exit 0
    fi
    exit 0
  fi
  sleep 2
done

echo "ERROR: Triton did not become live. Logs:" >&2
"${DOCKER[@]}" logs "$CONTAINER_NAME" 2>&1 | tail -n 100 >&2
exit 1
