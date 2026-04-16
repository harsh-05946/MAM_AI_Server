#!/bin/bash
# app/start_asr_api.sh

set -euo pipefail

cd /home/ubuntu/app || exit 1
source /home/ubuntu/app/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8001}
LOG_FILE=/home/ubuntu/app/asr_api.log
PID_FILE=/home/ubuntu/app/asr_api.pid

# Offline-first defaults
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export ALLOW_HF_FALLBACK=${ALLOW_HF_FALLBACK:-0}

# Local model directory for ASR service
export LOCAL_VIBEVOICE_DIR=${LOCAL_VIBEVOICE_DIR:-/home/ubuntu/app/models-local/asr/vibevoice}

if [ ! -d "$LOCAL_VIBEVOICE_DIR" ]; then
  echo "❌ Missing local VibeVoice model directory: $LOCAL_VIBEVOICE_DIR"
  echo "Run: python3 /home/ubuntu/app/bootstrap_local_models.py --service asr"
  exit 1
fi

if pgrep -f "uvicorn asr_api:app" > /dev/null; then
  echo "ASR API already running. Skipping start."
  exit 0
fi

echo "🚀 Starting ASR API (VibeVoice) on port $PORT..."
nohup uvicorn asr_api:app --host "$HOST" --port "$PORT" --workers 1 >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

sleep 2
if pgrep -f "uvicorn asr_api:app" > /dev/null; then
  echo "✅ ASR API started. Logs at $LOG_FILE"
  echo "   PID file: $PID_FILE"
  exit 0
else
  echo "❌ Failed to start ASR API. Check $LOG_FILE"
  exit 1
fi
