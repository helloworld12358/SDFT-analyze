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

EXTRA_ARGS=()
if [ "$COMPUTE_MISSING_TRAIN_GRADS" = "1" ]; then EXTRA_ARGS+=(--compute_missing_train_grads); fi
if [ "$TRAIN_GRAD_MAXSAMPLES" != "0" ]; then EXTRA_ARGS+=(--train_grad_max_samples "$TRAIN_GRAD_MAXSAMPLES"); fi

"$PYTHON_BIN" "$SCRIPT_DIR/schemeA_10_train_test_rect.py" \
  --datainf_root "$DATAINF_ROOT" \
  --all_train_datasets \
  --all_epochs \
  --method both \
  --gpu_ids "$GPU_IDS" \
  --num_workers "$PAIR_WORKERS" \
  --pair_timeout_sec "$PAIR_TIMEOUT" \
  --train_grad_batch_size "$TRAIN_GRAD_BS" \
  --train_grad_max_length "$TRAIN_GRAD_MAXLEN" \
  "${EXTRA_ARGS[@]}" \
  "$@"

