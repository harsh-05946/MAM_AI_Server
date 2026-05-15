#!/bin/bash
# app/status_main_api_a.sh
set -euo pipefail

PORT=${PORT:-8001}
if pgrep -f "uvicorn main:app.*--port ${PORT}" > /dev/null; then
  echo "Main API A running on port ${PORT}"
  exit 0
fi
echo "Main API A not running on port ${PORT}"
exit 1

