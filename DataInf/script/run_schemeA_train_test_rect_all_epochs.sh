#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${SCHEMEA_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${SCHEMEA_PYTHON:-python}"
GPU_IDS="${SCHEMEA_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
PAIR_WORKERS="${SCHEMEA_PAIRWISE_WORKERS:-0}"
PAIR_TIMEOUT="${SCHEMEA_PAIR_TIMEOUT_SEC:-0}"
COMPUTE_MISSING_TRAIN_GRADS="${SCHEMEA_COMPUTE_MISSING_TRAIN_GRADS:-1}"
TRAIN_GRAD_BS="${SCHEMEA_TRAIN_GRAD_BATCH_SIZE:-8}"
TRAIN_GRAD_MAXLEN="${SCHEMEA_TRAIN_GRAD_MAXLEN:-1024}"
TRAIN_GRAD_MAXSAMPLES="${SCHEMEA_TRAIN_GRAD_MAXSAMPLES:-0}"

# Split mode controls (can also be passed as positional args):
#   $1 => epoch (epoch_0/epoch_1/epoch_5/all)
#   $2 => method (sft/sdft/both)
#   $3 => train_dataset (optional; empty means all)
TARGET_EPOCH="${1:-${SCHEMEA_RECT_EPOCH:-all}}"
TARGET_METHOD="${2:-${SCHEMEA_RECT_METHOD:-both}}"
TARGET_TRAIN_DATASET="${3:-${SCHEMEA_RECT_TRAIN_DATASET:-}}"
USER_EXTRA_ARGS=()
if [ $# -ge 4 ]; then
  USER_EXTRA_ARGS=("${@:4}")
fi

EXTRA_ARGS=()
if [ "$COMPUTE_MISSING_TRAIN_GRADS" = "1" ]; then EXTRA_ARGS+=(--compute_missing_train_grads); fi
if [ "$TRAIN_GRAD_MAXSAMPLES" != "0" ]; then EXTRA_ARGS+=(--train_grad_max_samples "$TRAIN_GRAD_MAXSAMPLES"); fi

case "$TARGET_EPOCH" in
  all|epoch_0|epoch_1|epoch_5) ;;
  *)
    echo "Invalid epoch: $TARGET_EPOCH (expected: all|epoch_0|epoch_1|epoch_5)" >&2
    exit 1
    ;;
esac

case "$TARGET_METHOD" in
  both|sft|sdft) ;;
  *)
    echo "Invalid method: $TARGET_METHOD (expected: both|sft|sdft)" >&2
    exit 1
    ;;
esac

if [ -n "$TARGET_TRAIN_DATASET" ]; then
  EXTRA_ARGS+=(--train_dataset "$TARGET_TRAIN_DATASET")
else
  EXTRA_ARGS+=(--all_train_datasets)
fi

if [ "$TARGET_EPOCH" = "all" ]; then
  EXTRA_ARGS+=(--all_epochs)
else
  EXTRA_ARGS+=(--epoch "$TARGET_EPOCH")
fi

EXTRA_ARGS+=(--method "$TARGET_METHOD")

echo "[schemeA_rect] epoch=$TARGET_EPOCH method=$TARGET_METHOD train_dataset=${TARGET_TRAIN_DATASET:-ALL}"
echo "[schemeA_rect] gpu_ids=$GPU_IDS workers=$PAIR_WORKERS timeout=$PAIR_TIMEOUT"

"$PYTHON_BIN" "$SCRIPT_DIR/schemeA_10_train_test_rect.py" \
  --datainf_root "$DATAINF_ROOT" \
  --gpu_ids "$GPU_IDS" \
  --num_workers "$PAIR_WORKERS" \
  --pair_timeout_sec "$PAIR_TIMEOUT" \
  --train_grad_batch_size "$TRAIN_GRAD_BS" \
  --train_grad_max_length "$TRAIN_GRAD_MAXLEN" \
  "${EXTRA_ARGS[@]}" \
  "${USER_EXTRA_ARGS[@]}"
