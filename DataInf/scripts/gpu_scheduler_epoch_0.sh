# 保存为: gpu_scheduler_epoch_0.sh
#!/usr/bin/env bash
set -euo pipefail

# gpu_scheduler_epoch_0.sh
# 批量运行所有组合 (models × {epoch_0} × {sdft,sft})，顺序调用 pairwise_tasks_epoch_0.sh
# Usage: ./gpu_scheduler_epoch_0.sh
# Or run a single combo: ./gpu_scheduler_epoch_0.sh <MODEL> <EPOCH> <METHOD>

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PAIRWISE_SCRIPT="${SCRIPT_DIR}/pairwise_tasks_epoch_0.sh"

# 如果给了三个参数，则只运行该组合
if [ $# -ge 3 ]; then
  MODEL_ARG="$1"
  EPOCH_ARG="$2"
  METHOD_ARG="$3"
  RESULTS_DIR=$("$PAIRWISE_SCRIPT" "$MODEL_ARG" "$EPOCH_ARG" "$METHOD_ARG")
  echo "COMBO DONE: ${MODEL_ARG}/${EPOCH_ARG}/${METHOD_ARG} -> ${RESULTS_DIR}"
  exit 0
fi

# 批量组合配置（7 个模型 × 2 个 method）
MODELS=(gsm8k openfunction magicoder alpaca dolly lima openhermes)
METHODS=(sdft sft)
EPOCH="epoch_0"
MIN_FREE_PCT=15
SLEEP_SEC=3

for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    echo "START COMBO: ${model}/${EPOCH}/${method}"
    RESULTS_DIR=$("$PAIRWISE_SCRIPT" "$model" "$EPOCH" "$method" "$MIN_FREE_PCT" "$SLEEP_SEC")
    echo "COMBO DONE: ${model}/${EPOCH}/${method} -> ${RESULTS_DIR}"
    sleep 1
  done
done

echo "ALL COMBOS FINISHED"
