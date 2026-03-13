#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------------------
# 批量计算多组预测分布微分熵差（ΔH）的脚本（带显存自适应）
#
# 对每个模型预测（P: sdft，Q: sft）和每个数据集/测试集组合，
# 自动读取：
#   P: <SDFT_ROOT>/predictions/<dataset>/sdft/<testset>/generated_predictions.jsonl
#   Q: <SDFT_ROOT>/predictions/<dataset>/sft/<testset_q>/generated_predictions.jsonl
# 其中 testset_q = testset，除非 testset == "inference_sdft"，此时 testset_q="inference_sft"。
# 每次调用 continuous_entropy_diff_test.py 时带上 --batch_size，
# 若出现显存 OOM (“out of memory”) 错误，则将 batch_size 砍半重试，直至成功或 batch_size<1。
# -------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# BASE_DIR 指向 Mutual-Information 根目录
BASE_DIR="$(dirname "$SCRIPT_DIR")"
# SDFT_ROOT 回到 sdft 目录（Mutual-Information 的上一层）
SDFT_ROOT="$(dirname "$BASE_DIR")"
PYTHON_SCRIPT="${BASE_DIR}/continuous_entropy_diff_test.py"

# 预测结果根目录，修正为 sdft/predictions
PRED_ROOT="${SDFT_ROOT}/predictions"
# 初始 batch_size
INIT_BS=16

# 要处理的数据集
datasets=(
  "gsm8k"
  "openfunction"
  "magicoder"
  "alpaca"
  "dolly"
  "lima"
  "openhermes"
)

# 要处理的测试集
testsets=(
  "advbench-jailbreak"
  "advbench-raw"
  "alpaca_eval"
  "gsm8k"
  "multiarith"
  "openfunction"
  "inference_sdft"
)

for ds in "${datasets[@]}"; do
  for ts in "${testsets[@]}"; do
    # 定义 P 分布（sdft）路径
    file_p="${PRED_ROOT}/${ds}/sdft/${ts}/generated_predictions.jsonl"
    # 定义 Q 分布（sft）路径
    if [ "$ts" = "inference_sdft" ]; then
      ts_q="inference_sft"
    else
      ts_q="$ts"
    fi
    file_q="${PRED_ROOT}/${ds}/sft/${ts_q}/generated_predictions.jsonl"

    echo "=============================================="
    echo "数据集：${ds}  测试集：${ts}"
    echo "  P 文件：${file_p}"
    echo "  Q 文件：${file_q}"

    if [ ! -f "$file_p" ] || [ ! -f "$file_q" ]; then
      echo "跳过：缺少 P 或 Q 文件"
      continue
    fi

    # 自适应 batch_size
    BS=$INIT_BS
    while true; do
      echo ">>> 尝试 batch_size=${BS}"
      # 捕获 stderr 和 stdout
      output=$(python3.12 "$PYTHON_SCRIPT" \
                   --file_p "$file_p" \
                   --file_q "$file_q" \
                   --batch_size "$BS" 2>&1)
      ret=$?
      if [ $ret -eq 0 ]; then
        echo "✅ 计算成功（batch_size=${BS}）"
        break
      fi

      # 检查是否为显存 OOM 错误
      if echo "$output" | grep -iq "out of memory"; then
        BS=$(( BS / 2 ))
        if [ $BS -lt 1 ]; then
          echo "❌ batch_size 已降至 <1，无法继续重试，已跳过"
          break
        fi
        echo "⚠️  显存不足，batch_size 降级为 ${BS} 再试"
        continue
      else
        echo "❌ 计算失败，错误信息："
        echo "$output"
        break
      fi
    done

    echo
  done
done

echo "所有微分熵差计算完成！"
