#!/usr/bin/env bash
set -euo pipefail

# gpu_scheduler.sh
# 批量串行运行多个组合。每个组合通过调用 pairwise_tasks.sh（同目录下）执行。
# Fail-fast: 出现任何错误立刻退出。

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PAIRWISE_SCRIPT="${SCRIPT_DIR}/pairwise_tasks.sh"

# 若提供三个参数则只运行该组合；否则按预设批量组合运行
if [ $# -ge 3 ]; then
  MODEL_ARG="$1"
  EPOCH_ARG="$2"
  METHOD_ARG="$3"
  shift 3
  echo "Running single combo: ${MODEL_ARG}/${EPOCH_ARG}/${METHOD_ARG}"
  result_DIR=$("$PAIRWISE_SCRIPT" "$MODEL_ARG" "$EPOCH_ARG" "$METHOD_ARG")
  echo "COMBO DONE: ${MODEL_ARG}/${EPOCH_ARG}/${METHOD_ARG} -> ${result_DIR}"
  exit 0
fi

# 批量组合（顺序执行）
MODELS=(gsm8k openfunction magicoder alpaca dolly lima openhermes)
METHODS=(sdft sft)
EPOCHS="epoch_1"

for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    for epoch in "${EPOCHS[@]}"; do
      echo "START COMBO: ${model}/${epoch}/${method}"
      result_DIR=$("$PAIRWISE_SCRIPT" "$model" "$epoch" "$method")
      echo "COMBO DONE: ${model}/${epoch}/${method} -> ${result_DIR}"
      sleep 1
    done
  done
done

echo "ALL COMBOS FINISHED"
