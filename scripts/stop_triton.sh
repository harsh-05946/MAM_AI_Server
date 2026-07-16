#!/bin/bash
# Stop local Triton container started by start_triton.sh.

set -euo pipefail

CONTAINER_NAME="${TRITON_CONTAINER_NAME:-mam-triton}"

DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  if sudo docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
  else
    echo "Docker not reachable; nothing to stop."
    exit 0
  fi
fi

if "${DOCKER[@]}" ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  "${DOCKER[@]}" rm -f "$CONTAINER_NAME" >/dev/null
  echo "Stopped/removed $CONTAINER_NAME"
else
  echo "No container named $CONTAINER_NAME"
fi
