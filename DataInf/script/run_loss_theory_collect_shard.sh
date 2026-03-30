#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${LOSS_THEORY_DATAINF_ROOT:-${SCHEMEA_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}}"
PYTHON_BIN="${LOSS_THEORY_PYTHON:-${SCHEMEA_PYTHON:-python}}"

GPU_IDS="${LOSS_THEORY_GPU_IDS:-${SCHEMEA_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}}"
SHARD_COUNT="${LOSS_THEORY_SHARD_COUNT:-7}"
SHARD_INDEX="${LOSS_THEORY_SHARD_INDEX:-${1:-0}}"

EPOCHS="${LOSS_THEORY_EPOCHS:-epoch_1,epoch_5}"
METHODS="${LOSS_THEORY_METHODS:-sft,sdft}"
TASKS="${LOSS_THEORY_TASKS:-alpaca_eval,gsm8k,humaneval,multiarith,openfunction}"

BATCH_SIZE="${LOSS_THEORY_BATCH_SIZE:-8}"
MAX_LENGTH="${LOSS_THEORY_MAX_LENGTH:-0}"
MAX_SAMPLES_PER_TASK="${LOSS_THEORY_MAX_SAMPLES_PER_TASK:-0}"
TOKEN_SAMPLE_RATE="${LOSS_THEORY_TOKEN_SAMPLE_RATE:-0.01}"
TOKEN_SAMPLE_CAP="${LOSS_THEORY_TOKEN_SAMPLE_CAP_PER_SAMPLE:-64}"
SEQ_PROBE_RATE="${LOSS_THEORY_SEQ_PROBE_SAMPLE_RATE:-0.002}"
SEQ_PROBE_MAXTOK="${LOSS_THEORY_SEQ_PROBE_MAX_TOKENS:-512}"
FLUSH_EVERY="${LOSS_THEORY_FLUSH_EVERY:-64}"
SEED="${LOSS_THEORY_SEED:-42}"

LOG_DIR="${SCRIPT_DIR}/logs_loss_theory_collect"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [ "${#GPU_ARRAY[@]}" -eq 0 ]; then
  GPU_ARRAY=(0)
fi

echo "[loss_theory_collect] shard=${SHARD_INDEX}/${SHARD_COUNT} epochs=${EPOCHS} methods=${METHODS} tasks=${TASKS}"
echo "[loss_theory_collect] gpu_ids=${GPU_IDS} batch_size=${BATCH_SIZE} max_length=${MAX_LENGTH}"
echo "[loss_theory_collect] local_gpu_count=${#GPU_ARRAY[@]}"

declare -a PIDS=()
for ((g=0; g<${#GPU_ARRAY[@]}; g++)); do
  gpu="${GPU_ARRAY[$g]}"
  EFFECTIVE_SHARD_COUNT=$((SHARD_COUNT * ${#GPU_ARRAY[@]}))
  EFFECTIVE_SHARD_INDEX=$((SHARD_INDEX + g * SHARD_COUNT))
  log="${LOG_DIR}/collect_sh${SHARD_INDEX}_gpu${gpu}_$(date +%s).log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_BIN" "$SCRIPT_DIR/loss_theory_01_forward_collect.py" \
      --datainf_root "$DATAINF_ROOT" \
      --methods "$METHODS" \
      --epochs "$EPOCHS" \
      --tasks "$TASKS" \
      --batch_size "$BATCH_SIZE" \
      --max_length "$MAX_LENGTH" \
      --max_samples_per_task "$MAX_SAMPLES_PER_TASK" \
      --token_sample_rate "$TOKEN_SAMPLE_RATE" \
      --token_sample_cap_per_sample "$TOKEN_SAMPLE_CAP" \
      --seq_probe_sample_rate "$SEQ_PROBE_RATE" \
      --seq_probe_max_tokens "$SEQ_PROBE_MAXTOK" \
      --flush_every "$FLUSH_EVERY" \
      --shard_count "$EFFECTIVE_SHARD_COUNT" \
      --shard_index "$EFFECTIVE_SHARD_INDEX" \
      --seed "$SEED" \
      --device "cuda:0"
  ) >"$log" 2>&1 &
  PIDS+=("$!")
  sleep 1
done

fail=0
for pid in "${PIDS[@]}"; do
  if wait "$pid"; then
    echo "[OK] pid=$pid"
  else
    echo "[ERR] pid=$pid"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "[loss_theory_collect] completed with failures" >&2
  exit 1
fi
echo "[loss_theory_collect] completed successfully"
