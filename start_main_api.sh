#!/bin/bash
# app/start_main_api.sh

set -euo pipefail

cd /home/ubuntu/MAM_AI_Server || exit 1
source /home/ubuntu/MAM_AI_Server/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
LOG_FILE=/home/ubuntu/MAM_AI_Server/main_api.log
PID_FILE=/home/ubuntu/MAM_AI_Server/main_api.pid

# Online-by-default for simple HF cache flow (opt-in offline).
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export ALLOW_HF_FALLBACK=${ALLOW_HF_FALLBACK:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Do not force local model paths here. Let loaders resolve defaults (HF online/cache).

if pgrep -f "uvicorn main:app" > /dev/null; then
  echo "Main API already running. Skipping start."
  exit 0
fi

echo "🚀 Starting Main Inference API on port $PORT..."
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE ALLOW_HF_FALLBACK=$ALLOW_HF_FALLBACK"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
exec uvicorn main:app --host "$HOST" --port "$PORT" --workers 1

sleep 2
if pgrep -f "uvicorn main:app" > /dev/null; then
  echo "✅ Main API started. Logs at $LOG_FILE"
  echo "   PID file: $PID_FILE"
else
  echo "❌ Failed to start Main API. Check $LOG_FILE"
fi
