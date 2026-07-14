#!/bin/bash
# Stop the single AI instance and helpers.
# Refuses shutdown while a campaign is RUNNING unless --force.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="$PROJECT_DIR/run"
FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

cd "$PROJECT_DIR" || exit 1

if [[ -f "$PROJECT_DIR/runtime_reports/current/current_run.json" ]]; then
  state="$(python3 -c 'import json; print(json.load(open("runtime_reports/current/current_run.json")).get("state",""))' 2>/dev/null || true)"
  if [[ "$state" == "RUNNING" && "$FORCE" -ne 1 ]]; then
    echo "ERROR: campaign is RUNNING. Finalize it first, or pass --force." >&2
    exit 1
  fi
fi

stop_pidfile() {
  local name="$1"
  local pid_file="$RUN_DIR/${name}.pid"
  if [[ ! -f "$pid_file" ]]; then
    echo "$name: not running"
    return 0
  fi
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Stopping $name (PID $pid)..."
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "Force killing $name (PID $pid)"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$pid_file"
  echo "$name stopped"
}

stop_pidfile "throughput-reporter"
stop_pidfile "host-monitor"
stop_pidfile "gpu-monitor"
stop_pidfile "ai-optimized"
echo "Done."
