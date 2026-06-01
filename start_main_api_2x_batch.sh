#!/bin/bash
# Start main API with same URLs and 2x batch limits.

set -euo pipefail

cd /home/ubuntu/MAM_AI_Server || exit 1
source /home/ubuntu/MAM_AI_Server/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

# Online-by-default for simple HF cache flow (opt-in offline).
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export ALLOW_HF_FALLBACK=${ALLOW_HF_FALLBACK:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Keep URLs unchanged; only lift per-request batch caps for throughput retests.
export FACE_BATCH_MAX=$(( ${FACE_BATCH_MAX:-8} * 2 ))
export EMOTION_BATCH_MAX=$(( ${EMOTION_BATCH_MAX:-32} * 2 ))
export SCENE_BATCH_MAX=$(( ${SCENE_BATCH_MAX:-8} * 2 ))
export RAM_BATCH_MAX=$(( ${RAM_BATCH_MAX:-8} * 2 ))
export QWEN_BATCH_MAX=$(( ${QWEN_BATCH_MAX:-10} * 2 ))
export SARVAM_BATCH_MAX=$(( ${SARVAM_BATCH_MAX:-10} * 2 ))

if pgrep -f "uvicorn main:app.*--port ${PORT}" > /dev/null; then
  echo "Main API already running on port $PORT. Skipping start."
  exit 0
fi

echo "🚀 Starting Main Inference API on port $PORT with 2x batch caps..."
echo "FACE_BATCH_MAX=$FACE_BATCH_MAX EMOTION_BATCH_MAX=$EMOTION_BATCH_MAX SCENE_BATCH_MAX=$SCENE_BATCH_MAX"
echo "RAM_BATCH_MAX=$RAM_BATCH_MAX QWEN_BATCH_MAX=$QWEN_BATCH_MAX SARVAM_BATCH_MAX=$SARVAM_BATCH_MAX"
exec uvicorn main:app --host "$HOST" --port "$PORT" --workers 1
