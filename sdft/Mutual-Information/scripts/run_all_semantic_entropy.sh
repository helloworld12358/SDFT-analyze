#!/usr/bin/env bash
set -euo pipefail

# 获取当前脚本所在目录（Mutual-Information/scripts）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 上一级目录，即 Mutual-Information
BASE_DIR="$(dirname "$SCRIPT_DIR")"
# semantic_entropy.py 的绝对路径
PYTHON_SCRIPT="${BASE_DIR}/semantic_entropy.py"
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
  for mode in "train" "distilled"; do
    # 构造输入文件路径
    if [ "$mode" = "train" ]; then
      input_json="${DATA_DIR}/${ds}/${ds}_train.json"
    else
      input_json="${DATA_DIR}/${ds}/distilled_${ds}.json"
    fi

    # 如果文件不存在则跳过
    if [ ! -f "$input_json" ]; then
      echo "跳过：文件不存在 → $input_json"
      continue
    fi

    echo "=============================="
    echo "处理中：${ds} (${mode})"
    echo "输入文件：$input_json"

    # 调用 semantic_entropy.py，出错时只跳过当前项
    if ! python "$PYTHON_SCRIPT" --json_file "$input_json"; then
      echo "⚠️ 处理失败：${ds}/${mode}，已跳过"
      continue
    fi

    echo
  done
done

echo "所有语义熵计算完成！"

