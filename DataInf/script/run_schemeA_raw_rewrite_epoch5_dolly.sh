#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${SCHEMEA_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${SCHEMEA_PYTHON:-python}"
GPU_IDS="${SCHEMEA_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
PAIR_WORKERS="${SCHEMEA_PAIRWISE_WORKERS:-0}"
PAIR_TIMEOUT="${SCHEMEA_PAIR_TIMEOUT_SEC:-0}"
"$PYTHON_BIN" "$SCRIPT_DIR/schemeA_06_raw_rewrite_decompose.py" --datainf_root "$DATAINF_ROOT" --train_dataset dolly --epoch epoch_5 --feature_method sdft --oracle_mode mixed --gpu_ids "$GPU_IDS" --num_workers "$PAIR_WORKERS" --pair_timeout_sec "$PAIR_TIMEOUT" "$@"

