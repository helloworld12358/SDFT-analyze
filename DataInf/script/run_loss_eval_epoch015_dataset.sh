#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${LOSS_EVAL_DATAINF_ROOT:-${SCHEMEA_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}}"
PYTHON_BIN="${LOSS_EVAL_PYTHON:-${SCHEMEA_PYTHON:-python}}"
TRAIN_DATASET="${1:-${LOSS_EVAL_TRAIN_DATASET:-}}"
if [ -z "$TRAIN_DATASET" ]; then
  echo "Usage: bash run_loss_eval_epoch015_dataset.sh <train_dataset>" >&2
  echo "or set LOSS_EVAL_TRAIN_DATASET=<train_dataset>" >&2
  exit 1
fi

# Default: compute both methods and all 3 epochs.
METHODS="${LOSS_EVAL_METHODS:-sft,sdft}"
EPOCHS="${LOSS_EVAL_EPOCHS:-epoch_0,epoch_1,epoch_5}"
TASKS="${LOSS_EVAL_TASKS:-alpaca_eval,gsm8k,humaneval,multiarith,openfunction}"

# For H200 4-GPU nodes, use a larger default batch size.
BATCH_SIZE="${LOSS_EVAL_BATCH_SIZE:-16}"
# <=0 means no truncation in python.
MAX_LENGTH="${LOSS_EVAL_MAX_LENGTH:-0}"
MAX_SAMPLES="${LOSS_EVAL_MAX_SAMPLES:-0}"
AUTO_BATCH_SIZE="${LOSS_EVAL_AUTO_BATCH_SIZE:-1}"
BATCH_PROBE_MAX_BS="${LOSS_EVAL_BATCH_PROBE_MAX_BS:-64}"
BATCH_PROBE_BATCHES="${LOSS_EVAL_BATCH_PROBE_BATCHES:-1}"

# Prefer SCHEMEA GPU env if already set in your workflow.
GPU_IDS_CSV="${LOSS_EVAL_GPU_IDS:-${SCHEMEA_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}}"
IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
  GPU_IDS=(0)
fi

METHODS_LIST=()
IFS=',' read -r -a _M <<< "$METHODS"
for x in "${_M[@]}"; do
  x="$(echo "$x" | xargs)"
  [ -z "$x" ] && continue
  if [ "$x" = "sft" ] || [ "$x" = "sdft" ]; then
    METHODS_LIST+=("$x")
  fi
done
if [ "${#METHODS_LIST[@]}" -eq 0 ]; then
  METHODS_LIST=(sft sdft)
fi

EPOCHS_LIST=()
IFS=',' read -r -a _E <<< "$EPOCHS"
for x in "${_E[@]}"; do
  x="$(echo "$x" | xargs)"
  [ -z "$x" ] && continue
  EPOCHS_LIST+=("$x")
done
if [ "${#EPOCHS_LIST[@]}" -eq 0 ]; then
  EPOCHS_LIST=(epoch_0 epoch_1 epoch_5)
fi

LOG_DIR="$SCRIPT_DIR/logs_loss_eval"
mkdir -p "$LOG_DIR"

declare -a COMBOS
for m in "${METHODS_LIST[@]}"; do
  for e in "${EPOCHS_LIST[@]}"; do
    COMBOS+=("$m|$e")
  done
done

echo "[loss_eval] train_dataset=$TRAIN_DATASET"
echo "[loss_eval] methods=${METHODS_LIST[*]}"
echo "[loss_eval] epochs=${EPOCHS_LIST[*]}"
echo "[loss_eval] tasks=$TASKS"
echo "[loss_eval] gpu_ids=${GPU_IDS[*]}"
echo "[loss_eval] batch_size=$BATCH_SIZE"
echo "[loss_eval] max_length=$MAX_LENGTH (<=0 means no truncation)"
echo "[loss_eval] auto_batch_size=$AUTO_BATCH_SIZE probe_max_bs=$BATCH_PROBE_MAX_BS probe_batches=$BATCH_PROBE_BATCHES"

idx=0
fail=0
while [ $idx -lt ${#COMBOS[@]} ]; do
  declare -a PIDS=()
  declare -a LOGS=()
  batch_start=$idx
  for ((slot=0; slot<${#GPU_IDS[@]}; slot++)); do
    if [ $idx -ge ${#COMBOS[@]} ]; then
      break
    fi
    combo="${COMBOS[$idx]}"
    method="${combo%%|*}"
    epoch="${combo##*|}"
    gpu="${GPU_IDS[$slot]}"
    ts="$(date +%s)"
    log="$LOG_DIR/loss_eval_${TRAIN_DATASET}_${method}_${epoch}_gpu${gpu}_${ts}.log"

    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      EXTRA_ARGS=()
      if [ -n "${LOSS_EVAL_OUTPUT_ROOT:-}" ]; then EXTRA_ARGS+=(--output_root "$LOSS_EVAL_OUTPUT_ROOT"); fi
      if [ -n "${LOSS_EVAL_BASE_MODEL_PATH:-}" ]; then EXTRA_ARGS+=(--base_model_path "$LOSS_EVAL_BASE_MODEL_PATH"); fi
      if [ -n "${LOSS_EVAL_DATA_ROOT:-}" ]; then EXTRA_ARGS+=(--data_root "$LOSS_EVAL_DATA_ROOT"); fi
      if [ "$MAX_SAMPLES" != "0" ]; then EXTRA_ARGS+=(--max_samples "$MAX_SAMPLES"); fi
      if [ "${LOSS_EVAL_PREFER_AUTO_ON_FAIL:-0}" = "1" ]; then EXTRA_ARGS+=(--prefer_auto_on_fail); fi
      if [ "${LOSS_EVAL_VERBOSE:-0}" = "1" ]; then EXTRA_ARGS+=(--verbose); fi
      if [ "$AUTO_BATCH_SIZE" = "1" ]; then EXTRA_ARGS+=(--auto_batch_size); fi
      "$PYTHON_BIN" "$SCRIPT_DIR/loss_eval_01_compute_loss_tables.py" \
        --datainf_root "$DATAINF_ROOT" \
        --train_dataset "$TRAIN_DATASET" \
        --methods "$method" \
        --epochs "$epoch" \
        --tasks "$TASKS" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        --batch_probe_max_bs "$BATCH_PROBE_MAX_BS" \
        --batch_probe_batches "$BATCH_PROBE_BATCHES" \
        --device "cuda:0" \
        "${EXTRA_ARGS[@]}"
    ) >"$log" 2>&1 &

    PIDS+=("$!")
    LOGS+=("$log")
    idx=$((idx + 1))
    sleep 1
  done

  for ((j=0; j<${#PIDS[@]}; j++)); do
    pid="${PIDS[$j]}"
    log="${LOGS[$j]}"
    if wait "$pid"; then
      echo "[OK] pid=$pid log=$log"
    else
      echo "[ERR] pid=$pid log=$log"
      tail -n 40 "$log" || true
      fail=1
    fi
  done
  echo "[loss_eval] batch finished: start_index=$batch_start"
done

if [ "$fail" -ne 0 ]; then
  echo "[loss_eval] completed with failures for train_dataset=$TRAIN_DATASET" >&2
  exit 1
fi

echo "[loss_eval] completed successfully for train_dataset=$TRAIN_DATASET"
