#!/bin/bash

cd /home/ubuntu/app || exit 1

# Activate virtualenv
source /home/ubuntu/app/venv/bin/activate

# Ensure GPU visible
export CUDA_VISIBLE_DEVICES=0

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-1}

LOG_FILE=/home/ubuntu/app/app.log

# Check if already running
if pgrep -f "uvicorn main:app" > /dev/null; then
  echo "App already running. Skipping start."
  exit 0
fi

echo "Starting FastAPI app..."

exec uvicorn main:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS" \
  >> "$LOG_FILE" 2>&1
