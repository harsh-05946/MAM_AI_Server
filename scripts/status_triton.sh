#!/bin/bash
# Status for local Triton + feature flags (Phase 2).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTAINER_NAME="${TRITON_CONTAINER_NAME:-mam-triton}"
HTTP_URL="${TRITON_HTTP_URL:-http://127.0.0.1:8001}"

echo "=== Docker container ==="
if command -v docker >/dev/null 2>&1; then
  docker ps -a --filter "name=^${CONTAINER_NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true
else
  echo "Docker not installed"
fi

echo
echo "=== HTTP probes ($HTTP_URL) ==="
for path in /v2/health/live /v2/health/ready /v2; do
  code="$(curl -s -o /tmp/triton_probe.json -w '%{http_code}' --connect-timeout 2 --max-time 3 "${HTTP_URL}${path}" 2>/dev/null || echo 000)"
  echo "$path -> HTTP $code"
done

echo
echo "=== FastAPI /internal/runtime triton block (if AI up) ==="
if curl -sf --max-time 3 http://127.0.0.1:9001/internal/runtime >/tmp/ai_runtime.json 2>/dev/null; then
  python3 - <<'PY'
import json
from pathlib import Path
h = json.loads(Path("/tmp/ai_runtime.json").read_text())
print(json.dumps(h.get("triton", {}), indent=2))
PY
else
  echo "AI :9001 unreachable (or still starting)"
fi

# Prefer project venv if present for router unit view
PYTHON=""
if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  PYTHON="$PROJECT_DIR/.venv/bin/python"
elif [[ -x "$PROJECT_DIR/.venv-cu12/bin/python" ]]; then
  PYTHON="$PROJECT_DIR/.venv-cu12/bin/python"
fi
if [[ -n "$PYTHON" ]]; then
  echo
  echo "=== Router snapshot ==="
  PYTHONPATH="$PROJECT_DIR" "$PYTHON" -c "from runtime.triton_router import triton_runtime_status; import json; print(json.dumps(triton_runtime_status(), indent=2))"
fi
