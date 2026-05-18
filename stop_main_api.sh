#!/bin/bash
# app/stop_main_api.sh
echo "🛑 Stopping Main Inference API..."
PID_FILE=/home/ubuntu/MAM_AI_Server/main_api.pid

if [ -f "$PID_FILE" ]; then
  kill "$(cat "$PID_FILE")" 2>/dev/null || true
  rm -f "$PID_FILE"
fi
pkill -f "uvicorn main:app" 2>/dev/null || true
echo "✅ Stopped."
