#!/bin/bash
# app/stop_nginx_main_api.sh
set -euo pipefail

RUN_DIR=/home/ubuntu/app/nginx-run
PID_FILE=${RUN_DIR}/nginx.pid

if [ ! -f "${PID_FILE}" ]; then
  echo "NGINX not running (no pid file)."
  exit 0
fi

pid="$(cat "${PID_FILE}")"
if ! kill -0 "${pid}" 2>/dev/null; then
  echo "NGINX pid file exists but process not running."
  rm -f "${PID_FILE}"
  exit 0
fi

echo "Stopping NGINX (pid ${pid})"
kill "${pid}"

