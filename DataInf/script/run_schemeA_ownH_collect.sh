#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${SCHEMEA_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${SCHEMEA_PYTHON:-python}"

DEFAULT_EXISTING_ROOT="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/SDFT-analysis/DataInf/result"
EXISTING_ROOTS="${SCHEMEA_EXISTING_RESULT_ROOTS:-$DEFAULT_EXISTING_ROOT}"
ALLOW_RECOMPUTE_MISSING="${SCHEMEA_ALLOW_RECOMPUTE_MISSING:-1}"
GPU_IDS="${SCHEMEA_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
PAIR_WORKERS="${SCHEMEA_PAIRWISE_WORKERS:-0}"
PAIR_TIMEOUT="${SCHEMEA_PAIR_TIMEOUT_SEC:-0}"

if [[ "$ALLOW_RECOMPUTE_MISSING" == "1" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/schemeA_02_ownH_task_gram.py" \
    --datainf_root "$DATAINF_ROOT" \
    --all_train_datasets \
    --all_epochs \
    --method both \
    --existing_result_roots "$EXISTING_ROOTS" \
    --gpu_ids "$GPU_IDS" \
    --num_workers "$PAIR_WORKERS" \
    --pair_timeout_sec "$PAIR_TIMEOUT" \
    --allow_recompute_missing \
    "$@"
else
  "$PYTHON_BIN" "$SCRIPT_DIR/schemeA_02_ownH_task_gram.py" \
    --datainf_root "$DATAINF_ROOT" \
    --all_train_datasets \
    --all_epochs \
    --method both \
    --existing_result_roots "$EXISTING_ROOTS" \
    --gpu_ids "$GPU_IDS" \
    --num_workers "$PAIR_WORKERS" \
    --pair_timeout_sec "$PAIR_TIMEOUT" \
    "$@"
fi
