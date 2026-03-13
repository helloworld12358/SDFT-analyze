#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# 批量计算两分布熵差（ΔH）的脚本
#
# 对每个数据集，自动读取：
#   P: <DATA_DIR>/<dataset>/<dataset>_train.json
#   Q: <DATA_DIR>/<dataset>/distilled_<dataset>.json
# 然后调用 continuous_entropy_diff.py 计算 H(P) − H(Q)
# 并将结果写入相应目录。
# -----------------------------------------------------------------------------

# 获取当前脚本所在目录（Mutual-Information/scripts）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 上一级目录，即 Mutual-Information
BASE_DIR="$(dirname "$SCRIPT_DIR")"
# entropy_diff 脚本的绝对路径
PYTHON_SCRIPT="${BASE_DIR}/continuous_entropy_diff.py"
# 数据根目录
DATA_DIR="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/data"

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
  # 定义 P 和 Q 的路径
  # P: 训练集文件
  file_p="${DATA_DIR}/${ds}/${ds}_train.json"
  # Q: 蒸馏集文件
  file_q="${DATA_DIR}/${ds}/distilled_${ds}.json"

  echo "=============================="
  echo "处理数据集：${ds}"
  echo "  P (train) 文件：${file_p}"
  echo "  Q (distilled) 文件：${file_q}"

  # 如果任一文件不存在则跳过
  if [ ! -f "$file_p" ] || [ ! -f "$file_q" ]; then
    echo "跳过：缺少 P 或 Q 文件 → $file_p 或 $file_q"
    continue
  fi

  # 调用 entropy_diff 脚本
  # --file_p 表示分布 P 的路径
  # --file_q 表示分布 Q 的路径
  if ! python3.12 "$PYTHON_SCRIPT" --file_p "$file_p" --file_q "$file_q"; then
    echo "⚠️  处理失败：${ds}，已跳过"
    continue
  fi

  echo
done

echo "所有熵差计算完成！"

