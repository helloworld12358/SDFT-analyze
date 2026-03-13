#!/usr/bin/env bash
# run_per_file_stats.sh
# 按文件逐个调用 per_file_token_stats.py 以保证每个文件内部按自身频率降序绘图
# 使用前请确保：
#  - per_file_token_stats.py 在当前目录或修改 PY_SCRIPT 路径
#  - 已安装依赖：pip install transformers matplotlib tqdm

set -euo pipefail

# ---- 配置区（按需修改） ----
BASE_DATA_DIR="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/data"
BASE_RESULT_BASE="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/results"
MODEL_PATH="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/model/Llama-2-7b-chat-hf"
PY_SCRIPT="batch_token_stats_aligned.py"   # 若不在当前目录请写绝对路径

DATASETS=( "gsm8k" "openfunction" "magicoder" "alpaca" "dolly" "lima" "openhermes" )
file1_template="{dataset}_train.json"
file2_template="distilled_{dataset}.json"

TOP_N_LOG=100   # log 中显示前 100 token
TOP_M_PLOT=100  # 绘图展示该文件内部前 100 token（按文件内部频率降序）

# ----------------------------------------

for ds in "${DATASETS[@]}"; do
  data_dir="${BASE_DATA_DIR}/${ds}"
  result_dir="${BASE_RESULT_BASE}/${ds}"
  mkdir -p "${result_dir}"

  file1="${data_dir}/${file1_template//\{dataset\}/${ds}}"
  file2="${data_dir}/${file2_template//\{dataset\}/${ds}}"

  if [ -f "${file1}" ]; then
    echo "[INFO] 处理 ${file1}"
    python3 "${PY_SCRIPT}" --data-file "${file1}" --model-path "${MODEL_PATH}" --result-dir "${result_dir}" --top-n-log "${TOP_N_LOG}" --top-m-plot "${TOP_M_PLOT}"
  else
    echo "[WARN] 未找到 ${file1} ，跳过。"
  fi

  if [ -f "${file2}" ]; then
    echo "[INFO] 处理 ${file2}"
    python3 "${PY_SCRIPT}" --data-file "${file2}" --model-path "${MODEL_PATH}" --result-dir "${result_dir}" --top-n-log "${TOP_N_LOG}" --top-m-plot "${TOP_M_PLOT}"
  else
    echo "[WARN] 未找到 ${file2} ，跳过。"
  fi
done

echo "[ALL DONE] 结果已保存到 ${BASE_RESULT_BASE}/*（按 dataset 子目录）"
