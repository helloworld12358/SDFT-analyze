#!/usr/bin/env bash
set -euo pipefail

################################################################################
# run_all_hsic.sh
#
# 依次对以下数据集调用 compute_hsic_distilled.py：
#   gsm8k, openfunction, magicoder, alpaca, dolly, lima, openhermes
#
# 假定本脚本与 compute_hsic_distilled.py 位于同一目录下。
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 上一级目录，即 Mutual-Information
BASE_DIR="$(dirname "$SCRIPT_DIR")"
# compute_hsic_distilled.py 的路径（同目录下）
HSIC_SCRIPT="${BASE_DIR}/compute_hsic_distilled.py"

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
  echo "============================================================"
  echo ">> Running HSIC computation for dataset: ${ds}"
  if python3 "${HSIC_SCRIPT}" "${ds}"; then
    echo ">> [OK] ${ds} completed."
  else
    echo ">> [ERROR] ${ds} failed, skipping."
  fi
  echo
done

echo "All HSIC computations done!"
