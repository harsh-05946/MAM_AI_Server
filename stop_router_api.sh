#!/bin/bash
# app/stop_router_api.sh
set -euo pipefail

PORT=${PORT:-9000}
pids=$(pgrep -f "uvicorn router_api:app.*--port ${PORT}" || true)
if [ -z "${pids}" ]; then
  echo "Router API not running."
  exit 0
fi

echo "Stopping Router API (pids: ${pids})"
kill ${pids}

