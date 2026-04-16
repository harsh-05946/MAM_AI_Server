#!/bin/bash
# app/stop_asr_api.sh
echo "🛑 Stopping ASR API (VibeVoice)..."
PID_FILE=/home/ubuntu/app/asr_api.pid

terminate_pid() {
  local pid="$1"
  if ! ps -p "$pid" > /dev/null 2>&1; then
    return 0
  fi

  kill "$pid" 2>/dev/null || true
  sleep 2

  if ps -p "$pid" > /dev/null 2>&1; then
    echo "⚠️ PID $pid did not exit on SIGTERM; sending SIGKILL..."
    kill -9 "$pid" 2>/dev/null || true
    sleep 1
  fi
}

if [ -f "$PID_FILE" ]; then
  terminate_pid "$(cat "$PID_FILE")"
  rm -f "$PID_FILE"
fi

for pid in $(pgrep -f "uvicorn asr_api:app" 2>/dev/null || true); do
  terminate_pid "$pid"
done

if pgrep -f "uvicorn asr_api:app" > /dev/null 2>&1; then
  echo "❌ ASR API still running. Try with elevated permissions."
  exit 1
fi

echo "✅ Stopped."
