#!/bin/bash
# app/start_nginx_main_api.sh

set -euo pipefail

cd /home/ubuntu/MAM_AI_Server || exit 1

CONF=/home/ubuntu/MAM_AI_Server/nginx_main_api.conf
RUN_DIR=/home/ubuntu/MAM_AI_Server/nginx-run
PID_FILE=${RUN_DIR}/nginx.pid

mkdir -p "${RUN_DIR}"

if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "NGINX already running (pid $(cat "${PID_FILE}"))."
  exit 0
fi

echo "Starting NGINX on :8000 -> router :9000"
nginx -c "${CONF}" -p "${RUN_DIR}"

