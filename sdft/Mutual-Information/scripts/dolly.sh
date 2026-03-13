#!/bin/bash
set -e

################################################################################
# inference_two_adapters_torchrun.sh
#
# 对同一训练集 dolly_train 依次使用：
#   1) SD-FT 微调模型（checkpoints/dolly/sdft）
#   2) SFT    微调模型（checkpoints/dolly/sft）
# 执行推理并将结果分别保存，无任何评估。
################################################################################

# ———— 配置 ——————————————————————————————————————————————————————————————————————————
# 基础模型与数据集
MODEL_PATH="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/model/Llama-2-7b-chat-hf"
DATASET="dolly_train"
TEMPLATE="alpaca"

# 两个 Adapter 路径
ADAPTER_SDFT="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/checkpoints/dolly/sdft"
ADAPTER_SFT="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/checkpoints/dolly/sft"

# 推理参数
BATCH_SIZE=8
MAX_SAMPLES=9999999999999
CUDA_VISIBLE_DEVICES="0,1,2,3"

# 输出目录
OUT_SDFT="predictions/dolly/sdft/inference_sdft"
OUT_SFT="predictions/dolly/sft/inference_sft"

################################################################################
# 准备
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
mkdir -p "${OUT_SDFT}" "${OUT_SFT}"

################################################################################
# 1) 使用 SD-FT Adapter 推理
echo ">>> [1/2] Inference with SD-FT adapter on ${DATASET}"
torchrun --nproc_per_node=4 main.py \
  --stage sft \
  --model_name_or_path "${MODEL_PATH}" \
  --adapter_name_or_path "${ADAPTER_SDFT}" \
  --do_predict \
  --dataset "${DATASET}" \
  --template "${TEMPLATE}" \
  --output_dir "${OUT_SDFT}" \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --max_samples ${MAX_SAMPLES} \
  --predict_with_generate \
  --overwrite_cache \
  --fp16

echo "  → Saved to ${OUT_SDFT}/generated_predictions.jsonl"
echo

################################################################################
# 2) 使用 SFT Adapter 推理
echo ">>> [2/2] Inference with SFT adapter on ${DATASET}"
torchrun --nproc_per_node=4 main.py \
  --stage sft \
  --model_name_or_path "${MODEL_PATH}" \
  --adapter_name_or_path "${ADAPTER_SFT}" \
  --do_predict \
  --dataset "${DATASET}" \
  --template "${TEMPLATE}" \
  --output_dir "${OUT_SFT}" \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --max_samples ${MAX_SAMPLES} \
  --predict_with_generate \
  --overwrite_cache \
  --fp16

echo "  → Saved to ${OUT_SFT}/generated_predictions.jsonl"
echo
echo ">>> Both inferences completed."

# ─── 3) 计算 HSIC ────────────────────────────────────────────────────────────
echo ">>> [3/3] Computing HSIC for distilled vs SD-FT and train vs SFT"
python Mutual-Information/revalue.py dolly
echo ">>> HSIC computation completed. See results/dolly_hsic.log"