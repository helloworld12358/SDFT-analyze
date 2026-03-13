#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------------------
# 批量计算 HSIC（交叉）脚本
#
# 依次对以下数据集运行 HSIC_QA.py：
#   gsm8k, openfunction, magicoder, alcapa, dolly, lima, openhermes
#
# 假设：
#   - 本脚本放在 Mutual-Information/scripts/ 下
#   - HSIC_QA.py 放在 Mutual-Information/ 下
#   - 结果会写入 Mutual-Information/results/hsic/<dataset>/
# -------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# BASE_DIR 指向 Mutual-Information 根目录
BASE_DIR="$(dirname "$SCRIPT_DIR")"
# Python 脚本在上一级目录
PYTHON_SCRIPT="${BASE_DIR}/HSIC_QA.py"

# 要处理的数据集列表
datasets=(
  "gsm8k"
  "openfunction"
  "magicoder"
  "alpaca"
  "dolly"
  "lima"
  "openhermes"
)

for ds in "${datasets[@]}"; do
  echo "=============================================="
  echo "开始计算 HSIC cross for dataset: ${ds}"

  if ! python3.12 "$PYTHON_SCRIPT" "$ds"; then
    echo "⚠️  HSIC 计算失败: ${ds}, 跳过后续操作"
    continue
  fi

  echo "✅ 完成: ${ds}"
  echo
done

echo "所有数据集 HSIC 交叉计算完成！"
