#!/bin/bash
# app/stop_main_api_a.sh
set -euo pipefail

PORT=${PORT:-8001}
pids=$(pgrep -f "uvicorn main:app.*--port ${PORT}" || true)
if [ -z "${pids}" ]; then
  echo "Main API A not running."
  exit 0
fi

echo "Stopping Main API A (pids: ${pids})"
kill ${pids}

