#!/bin/bash
# Status for the single optimized AI instance.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="$PROJECT_DIR/run"
REPORT="$PROJECT_DIR/runtime_reports/current/AI_THROUGHPUT_REPORT.md"

cd "$PROJECT_DIR" || exit 1

show_pid() {
  local name="$1"
  local pid_file="$RUN_DIR/${name}.pid"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "$name: RUNNING pid=$pid"
      return 0
    fi
  fi
  echo "$name: STOPPED"
}

show_pid "ai-optimized"
show_pid "gpu-monitor"
show_pid "host-monitor"
show_pid "throughput-reporter"

echo
if [[ -f runtime_reports/current/current_run.json ]]; then
  echo "current_run.json:"
  cat runtime_reports/current/current_run.json
  echo
fi

if curl -s --max-time 5 http://127.0.0.1:9001/health >/tmp/ai_health.json 2>/dev/null; then
  echo "health/liveness (9001):"
  python3 -m json.tool </tmp/ai_health.json | head -n 20
else
  echo "health (9001): unreachable"
fi

if curl -s -o /tmp/ai_ready.json -w '%{http_code}' --max-time 5 http://127.0.0.1:9001/ready >/tmp/ai_ready_code.txt 2>/dev/null; then
  code="$(cat /tmp/ai_ready_code.txt 2>/dev/null || echo '?')"
  echo "ready (9001) HTTP ${code}:"
  python3 -m json.tool </tmp/ai_ready.json | head -n 40
else
  echo "ready (9001): unreachable"
fi

echo
echo "report: $REPORT"
if [[ -f "$REPORT" ]]; then
  head -n 40 "$REPORT"
fi
