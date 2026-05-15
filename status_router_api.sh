#!/bin/bash
# app/status_router_api.sh
set -euo pipefail

PORT=${PORT:-9000}
if pgrep -f "uvicorn router_api:app.*--port ${PORT}" > /dev/null; then
  echo "Router API running on port ${PORT}"
  exit 0
fi
echo "Router API not running on port ${PORT}"
exit 1

