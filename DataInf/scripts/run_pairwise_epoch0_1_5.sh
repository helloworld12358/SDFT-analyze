#!/usr/bin/env bash
set -euo pipefail

# run_pairwise_epoch0_1_5.sh
# 顺序运行：models x epochs(epoch_0,epoch_1,epoch_5) x methods(sdft,sft)
# 每个组合调用 compute_pairwise_from_grads_tagged.py 生成矩阵并写入 txt（内积，默认不 normalize）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="python"
PY_SCRIPT="${DATAINF_ROOT}/src/compute_pairwise_from_grads_tagged.py"

MODELS=( "gsm8k" "openfunction" "magicoder" "alpaca" "dolly" "lima" "openhermes" )
EPOCHS=( "epoch_0" "epoch_1" "epoch_5" )
METHODS=( "sdft" "sft" )

# 默认数据集列表（与脚本内部一致）
DATASET_NAMES=( "alpaca_eval" "gsm8k" "humaneval" "multiarith" "openfunction" )

for model in "${MODELS[@]}"; do
  for epoch in "${EPOCHS[@]}"; do
    for method in "${METHODS[@]}"; do
      echo "START COMBO: ${model}/${epoch}/${method}"
      # 调用 python 脚本（不使用 normalize，默认 float64）
      "$PYTHON" "$PY_SCRIPT" \
        --datainf_root "$DATAINF_ROOT" \
        --model "$model" \
        --epoch "$epoch" \
        --method "$method" \
        --dataset_names "$(IFS=,; echo "${DATASET_NAMES[*]}")" \
        --verbose
      # python 返回的最后一行是生成的 txt 的绝对路径（脚本会打印）
      echo "FINISHED COMBO: ${model}/${epoch}/${method}"
      echo "----------------------------------------"
    done
  done
done

echo "ALL DONE."
