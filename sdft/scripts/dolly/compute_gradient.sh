#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "${SCRIPT_DIR}/../model" ] && [ -d "${SCRIPT_DIR}/../data" ]; then
  SDFT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  SDFT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
set -euo pipefail
# 分析 LoRA 梯度的启动脚本（调用 analyze_gradients_llama_factory.py）
# 输出到终端（不重定向到文件）。
#
# 默认：单机 4 卡（CUDA_VISIBLE_DEVICES="0,1,2,3"），torchrun --nproc_per_node=4
# 若需要跨多台机器，请参见脚本末尾说明。

# ---------------------------
# 基本配置（请按实际改路径）
# ---------------------------
MODEL_PATH="${SDFT_ROOT}/model/Llama-2-7b-chat-hf"
# 如果不使用 adapter（只提供 base model），请留空 ADAPTER_DIR=""
ADAPTER_DIR="${SDFT_ROOT}/epoch1_checkpoints/dolly/sft"
TRAIN_DATASET="dolly"
OUTPUT_FOLDER="analysis/${TRAIN_DATASET}/gradient_analysis"
# python 脚本文件名（确保与实际文件名一致）
PYTHON_SCRIPT="analyze_gradients_llama_factory.py"

# 合并脚本路径（请确保 merge_gradients.py 在同一目录或给出绝对路径）
MERGE_SCRIPT="merge_gradients.py"

# ---------------------------
# 运行/设备配置（默认单机 4 卡）
# ---------------------------
CUDA_VISIBLE_DEVICES="0,1,2,3"      # 这里设置要用的 GPU id 列表（本机 GPU）
NPROC_PER_NODE=4                    # torchrun 的 --nproc_per_node（等于 GPU 数）
MASTER_PORT=12345

# ---------------------------
# 分析超参数（按需修改）
# ---------------------------
SPLIT="train"

# ---------------------------
# 训练超参数（传给 llmtuner/get_train_args）
# 这里以数组形式提供，会追加到命令行
# 示例：--per_device_train_batch_size 2 --gradient_accumulation_steps 4 --bf16
# 我额外加入 ddp_find_unused_parameters False 以匹配 llmtuner 对 LoRA+DDP 的建议
# ---------------------------
EXTRA_TRAIN_ARGS=( --per_device_train_batch_size 2 --gradient_accumulation_steps 4 --bf16 --ddp_find_unused_parameters False )

# ---------------------------
# 环境变量
# ---------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# 可选 NCCL 调优（按需打开）
# export NCCL_DEBUG=INFO
# export NCCL_P2P_LEVEL=NVL
# export NCCL_IB_HCA=mlx5_0

# ---------------------------
# 准备输出目录
# ---------------------------
rm -rf "${OUTPUT_FOLDER}"
mkdir -p "${OUTPUT_FOLDER}"

# ---------------------------
# 构建命令行参数
# ---------------------------
CMD=( torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" "${PYTHON_SCRIPT}"
      --output_dir "${OUTPUT_FOLDER}"
      --split "${SPLIT}"
      --model_name_or_path "${MODEL_PATH}"
      --dataset "${TRAIN_DATASET}_train"
      --template alpaca
)

# 如果指定了 adapter，则传入 --adapter_name_or_path
if [[ -n "${ADAPTER_DIR}" ]]; then
  CMD+=( --adapter_name_or_path "${ADAPTER_DIR}" )
else
  # 不传 adapter 时脚本会自动注入 lora_target=q_proj,v_proj
  echo "[Info] No adapter specified — analyze script will auto-inject lora_target=q_proj,v_proj."
fi

# 将 extra train args 追加到命令
for a in "${EXTRA_TRAIN_ARGS[@]}"; do
  CMD+=( "${a}" )
done

# ---------------------------
# 打印信息并运行（只输出到终端）
# ---------------------------
echo "=== Running gradient analysis ==="
echo "Model: ${MODEL_PATH}"
if [[ -n "${ADAPTER_DIR}" ]]; then
  echo "Adapter: ${ADAPTER_DIR}"
else
  echo "Adapter: (none)"
fi
echo "Dataset: ${TRAIN_DATASET}_train"
echo "Output dir: ${OUTPUT_FOLDER}"
echo "Torchrun nproc_per_node: ${NPROC_PER_NODE}, master_port: ${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Extra train args: ${EXTRA_TRAIN_ARGS[*]}"
echo "-------------------------------------------------"

# 执行命令（输出到终端）
"${CMD[@]}"

EXITCODE=$?
if [ "${EXITCODE}" -ne 0 ]; then
    echo "[Error] ${PYTHON_SCRIPT} exited with code ${EXITCODE}"
    exit "${EXITCODE}"
fi

echo "Gradient analysis finished successfully. Results are saved in ${OUTPUT_FOLDER}."
echo "Now starting merge step..."

# ---------------------------
# 调用合并脚本
# ---------------------------
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


