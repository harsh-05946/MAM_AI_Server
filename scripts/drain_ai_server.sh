#!/bin/bash
# Graceful drain: mark not ready, wait for inflight /process to finish, then stop.
# Usage: bash scripts/drain_ai_server.sh [--force]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="$PROJECT_DIR/run"
PORT="${AI_INTERNAL_PORT:-9001}"
PUBLIC_PORT="${AI_PUBLIC_PORT:-9000}"
DRAIN_SECONDS="${AI_GRACEFUL_DRAIN_SECONDS:-30}"
FORCE_SECONDS="${AI_FORCE_SHUTDOWN_SECONDS:-90}"
FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

cd "$PROJECT_DIR" || exit 1

base="http://127.0.0.1:${PORT}"
alt="http://127.0.0.1:${PUBLIC_PORT}"

echo "Beginning drain via POST /internal/drain ..."
if ! curl -sf -X POST --max-time 5 "${base}/internal/drain" >/tmp/ai_drain.json 2>/dev/null; then
  if ! curl -sf -X POST --max-time 5 "${alt}/internal/drain" >/tmp/ai_drain.json 2>/dev/null; then
    echo "WARN: drain endpoint unreachable; continuing with stop." >&2
  fi
fi
if [[ -f /tmp/ai_drain.json ]]; then
  cat /tmp/ai_drain.json
  echo
fi

echo "Waiting for /ready=503 and inflight drain (up to ${DRAIN_SECONDS}s)..."
elapsed=0
while (( elapsed < DRAIN_SECONDS )); do
  code="$(curl -s -o /tmp/ai_ready.json -w '%{http_code}' --max-time 3 "${base}/ready" 2>/dev/null || true)"
  inflight="$(python3 -c 'import json,sys; print(json.load(open("/tmp/ai_ready.json")).get("inflight_requests", "?"))' 2>/dev/null || echo "?")"
  echo "  ready_http=${code} inflight=${inflight} t=${elapsed}s"
  if [[ "$code" == "503" ]] && [[ "$inflight" == "0" || "$inflight" == "?" ]]; then
    break
  fi
  sleep 1
  elapsed=$((elapsed + 1))
done

if [[ "$FORCE" -eq 1 ]]; then
  echo "Force stop requested."
  bash "$SCRIPT_DIR/stop_single_ai_optimized.sh" --force
else
  echo "Stopping AI server..."
  # Prefer force after drain window so a stuck campaign cannot block shutdown forever.
  if ! bash "$SCRIPT_DIR/stop_single_ai_optimized.sh" --force; then
    echo "stop failed; retrying with SIGKILL window ${FORCE_SECONDS}s" >&2
    sleep 1
    bash "$SCRIPT_DIR/stop_single_ai_optimized.sh" --force || true
  fi
fi
echo "Drain complete."
