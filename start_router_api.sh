#!/bin/bash
# app/start_router_api.sh

set -euo pipefail

cd /home/ubuntu/app || exit 1
source /home/ubuntu/app/venv/bin/activate

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-9000}

export BACKEND_A=${BACKEND_A:-http://127.0.0.1:8001}
export BACKEND_B=${BACKEND_B:-http://127.0.0.1:8002}
export ROUTER_QUEUE_MAX=${ROUTER_QUEUE_MAX:-200}
export ROUTER_QUEUE_WAIT_TIMEOUT_SEC=${ROUTER_QUEUE_WAIT_TIMEOUT_SEC:-300}
export ROUTER_UPSTREAM_TIMEOUT_SEC=${ROUTER_UPSTREAM_TIMEOUT_SEC:-300}
export BACKEND_A_CAPACITY=${BACKEND_A_CAPACITY:-2}
export BACKEND_B_CAPACITY=${BACKEND_B_CAPACITY:-2}

# Optional path pinning (see docs/router.md). Example: send all emotion + embeddings to backend A only.
# export ROUTER_PIN_BACKEND=A
# export ROUTER_PIN_PATH_PREFIXES=/process/emotion,/process/embeddings

if pgrep -f "uvicorn router_api:app.*--port ${PORT}" > /dev/null; then
  echo "Router API already running on port $PORT. Skipping start."
  exit 0
fi

echo "Starting Router API on port $PORT..."
exec uvicorn router_api:app --host "$HOST" --port "$PORT" --workers 1 --log-level info

