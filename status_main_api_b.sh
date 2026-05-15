#!/bin/bash
# app/status_main_api_b.sh
set -euo pipefail

PORT=${PORT:-8002}
if pgrep -f "uvicorn main:app.*--port ${PORT}" > /dev/null; then
  echo "Main API B running on port ${PORT}"
  exit 0
fi
echo "Main API B not running on port ${PORT}"
exit 1

