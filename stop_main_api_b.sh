#!/bin/bash
# app/stop_main_api_b.sh
set -euo pipefail

PORT=${PORT:-8002}
pids=$(pgrep -f "uvicorn main:app.*--port ${PORT}" || true)
if [ -z "${pids}" ]; then
  echo "Main API B not running."
  exit 0
fi

echo "Stopping Main API B (pids: ${pids})"
kill ${pids}

