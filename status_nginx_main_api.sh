#!/bin/bash
# app/status_nginx_main_api.sh
set -euo pipefail

PID_FILE=/home/ubuntu/MAM_AI_Server/nginx-run/nginx.pid
if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "NGINX running (pid $(cat "${PID_FILE}"))"
  exit 0
fi
echo "NGINX not running"
exit 1

