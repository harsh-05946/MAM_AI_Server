#!/bin/bash
# app/status_main_api.sh
PID_FILE=/home/ubuntu/app/main_api.pid

if pgrep -f "uvicorn main:app" > /dev/null; then
  echo "✅ Main Inference API is RUNNING."
  pgrep -af "uvicorn main:app"
  if [ -f "$PID_FILE" ]; then
    echo "PID file: $PID_FILE ($(cat "$PID_FILE"))"
  fi
  if command -v curl >/dev/null 2>&1; then
    echo "Health:"
    curl -s --connect-timeout 2 --max-time 5 "http://127.0.0.1:8000/health" || true
    echo
  fi
else
  echo "❌ Main Inference API is STOPPED."
fi
