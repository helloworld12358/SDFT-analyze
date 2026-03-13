#!/bin/bash
set -euo pipefail

# 单节点单进程运行版本（使用单张 GPU）
# - 如果你想要用多卡并行，请用原来的 torchrun 脚本 / 或把本脚本中 python 调用改回 torchrun --nproc_per_node=N。
# - 仅把运行方式改为单进程，脚本其余行为（参数、输出目录、merge 调用）保持不变。

MODEL_PATH="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/model/Llama-2-7b-chat-hf"
ADAPTER_DIR="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/epoch1_checkpoints/alpaca/sft"
TRAIN_DATASET="gsm8k"
OUTPUT_FOLDER="analysis/alpaca/gradient_analysis_gsm8ktest"
PYTHON_SCRIPT="analyze_gradients_llama_factory.py"
MERGE_SCRIPT="merge_gradients.py"

# 单进程只使用一张 GPU（若要换到 GPU 1,2 等，修改为 "1" 或 "2"）
CUDA_VISIBLE_DEVICES="0"

# 这些与单进程运行无关，但保留为记录
NPROC_PER_NODE=1
MASTER_PORT=12345

MAX_STEPS=2
SPLIT="test"

EXTRA_TRAIN_ARGS=( --per_device_train_batch_size 2 --gradient_accumulation_steps 4 --bf16 --ddp_find_unused_parameters False )

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# 清理并准备输出目录（注意：会删除同名目录）
rm -rf "${OUTPUT_FOLDER}"
mkdir -p "${OUTPUT_FOLDER}"

# ---------------------------
# 使用单进程 python 启动（single-process）
# ---------------------------
CMD=( python "${PYTHON_SCRIPT}"
      --output_dir "${OUTPUT_FOLDER}"
      --max_steps "${MAX_STEPS}"
      --split "${SPLIT}"
      --model_name_or_path "${MODEL_PATH}"
      --dataset "${TRAIN_DATASET}_train"
      --template gsm8k
)

if [[ -n "${ADAPTER_DIR}" ]]; then
  CMD+=( --adapter_name_or_path "${ADAPTER_DIR}" )
fi

for a in "${EXTRA_TRAIN_ARGS[@]}"; do
  CMD+=( "${a}" )
done

echo "=== Running gradient analysis (single-process) ==="
echo "Model: ${MODEL_PATH}"
if [[ -n "${ADAPTER_DIR}" ]]; then
  echo "Adapter: ${ADAPTER_DIR}"
else
  echo "Adapter: (none)"
fi
echo "Dataset: ${TRAIN_DATASET}_train"
echo "Split: ${SPLIT}"
echo "Output dir: ${OUTPUT_FOLDER}"
echo "Max steps: ${MAX_STEPS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Extra train args: ${EXTRA_TRAIN_ARGS[*]}"
echo "-------------------------------------------------"

# run (single-process)
"${CMD[@]}"

EXITCODE=$?
if [ "${EXITCODE}" -ne 0 ]; then
    echo "[Error] ${PYTHON_SCRIPT} exited with code ${EXITCODE}"
    exit "${EXITCODE}"
fi

echo "Gradient analysis finished successfully. Results are saved in ${OUTPUT_FOLDER}."
echo "Now starting merge step..."

if [[ ! -f "${MERGE_SCRIPT}" ]]; then
  echo "[Error] merge script not found at ${MERGE_SCRIPT}. Please place merge_gradients.py alongside this script or update MERGE_SCRIPT path."
  exit 2
fi

python "${MERGE_SCRIPT}" --output_dir "${OUTPUT_FOLDER}"
MERGE_EXIT=$?
if [ "${MERGE_EXIT}" -ne 0 ]; then
  echo "[Error] merge_gradients.py failed with exit code ${MERGE_EXIT}"
  exit "${MERGE_EXIT}"
fi

echo "Merge completed. Merged outputs are located at: ${OUTPUT_FOLDER}/merged"
