#!/bin/bash
# app/status_asr_api.sh
PID_FILE=/home/ubuntu/app/asr_api.pid

if pgrep -f "uvicorn asr_api:app" > /dev/null; then
  echo "✅ ASR API (VibeVoice) is RUNNING."
  pgrep -af "uvicorn asr_api:app"
  if [ -f "$PID_FILE" ]; then
    echo "PID file: $PID_FILE ($(cat "$PID_FILE"))"
  fi
  if command -v curl >/dev/null 2>&1; then
    echo "Health:"
    curl -s "http://127.0.0.1:8001/health" || true
    echo
  fi
else
  echo "❌ ASR API (VibeVoice) is STOPPED."
fi
