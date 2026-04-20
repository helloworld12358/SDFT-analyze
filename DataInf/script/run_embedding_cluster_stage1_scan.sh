#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${EMBED_CLUSTER_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${EMBED_CLUSTER_PYTHON:-python}"
OUTPUT_ROOT="${EMBED_CLUSTER_OUTPUT_ROOT:-}"
TRAIN_DATASET="${EMBED_CLUSTER_TRAIN_DATASET:-}"
INCLUDE_BASE="${EMBED_CLUSTER_INCLUDE_BASE:-0}"
TASKS="${EMBED_CLUSTER_TASKS:-alpaca_eval,gsm8k,humaneval,multiarith,openfunction}"
SAMPLES_PER_TASK="${EMBED_CLUSTER_SAMPLES_PER_TASK:-100}"
SEED="${EMBED_CLUSTER_SEED:-42}"
LAYERS="${EMBED_CLUSTER_LAYERS:-all}"
BATCH_SIZE="${EMBED_CLUSTER_BATCH_SIZE:-8}"
MAX_LENGTH="${EMBED_CLUSTER_MAX_LENGTH:-1024}"
DEVICE="${EMBED_CLUSTER_DEVICE:-auto}"
PREFER_AUTO_ON_FAIL="${EMBED_CLUSTER_PREFER_AUTO_ON_FAIL:-1}"

ARGS=(
  --datainf_root "$DATAINF_ROOT"
  --tasks "$TASKS"
  --samples_per_task "$SAMPLES_PER_TASK"
  --seed "$SEED"
  --layers "$LAYERS"
  --batch_size "$BATCH_SIZE"
  --max_length "$MAX_LENGTH"
  --device "$DEVICE"
)

if [ -n "$OUTPUT_ROOT" ]; then
  ARGS+=(--output_root "$OUTPUT_ROOT")
fi
if [ "$INCLUDE_BASE" = "1" ]; then
  ARGS+=(--include_base)
fi
if [ "$PREFER_AUTO_ON_FAIL" = "1" ]; then
  ARGS+=(--prefer_auto_on_fail)
fi
if [ -n "$TRAIN_DATASET" ]; then
  ARGS+=(--train_dataset "$TRAIN_DATASET")
else
  ARGS+=(--all_train_datasets)
fi

"$PYTHON_BIN" "$SCRIPT_DIR/embedding_cluster_01_epoch5_layer_scan.py" "${ARGS[@]}" "$@"

