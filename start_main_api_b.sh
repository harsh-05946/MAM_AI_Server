#!/bin/bash
# app/start_main_api_b.sh

set -euo pipefail

cd /home/ubuntu/app || exit 1
source /home/ubuntu/app/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export INSTANCE_NAME=${INSTANCE_NAME:-main-b}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8002}

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export ALLOW_HF_FALLBACK=${ALLOW_HF_FALLBACK:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOG_FILE=/home/ubuntu/app/main_api_b.log
PID_FILE=/home/ubuntu/app/main_api_b.pid

if pgrep -f "uvicorn main:app.*--port ${PORT}" > /dev/null; then
  echo "Main API B already running on port $PORT. Skipping start."
  exit 0
fi

echo "Starting Main Inference API B on port $PORT..."
exec uvicorn main:app --host "$HOST" --port "$PORT" --workers 1 --log-level info

