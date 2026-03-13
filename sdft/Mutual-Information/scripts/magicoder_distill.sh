#!/usr/bin/env bash
set -euo pipefail

################################################################################
# inference_two_adapters_torchrun.sh
#
# Performs inference on the distilled_magicoder.json dataset using:
#   1) SD-FT fine-tuned model (checkpoints/magicoder/sdft)
#   2) SFT fine-tuned model (checkpoints/magicoder/sft)
# Saves the results separately and computes HSIC.
################################################################################

# --- Configurations ---
# Base model and input JSON file
MODEL_PATH="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/model/Llama-2-7b-chat-hf"
DATASET="distilled_magicoder"
TEMPLATE="alpaca"

# Adapter paths (made absolute)
ADAPTER_SDFT="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/checkpoints/magicoder/sdft"
ADAPTER_SFT="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/checkpoints/magicoder/sft"

# Inference parameters
BATCH_SIZE=8
MAX_SAMPLES=9999999999999
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Output directories (updated for distilled dataset)
OUT_SDFT="predictions/magicoder/sdft/inference_distilled_sdft"
OUT_SFT="predictions/magicoder/sft/inference_distilled_sft"

################################################################################
# Preparation
mkdir -p "${OUT_SDFT}" "${OUT_SFT}"

################################################################################
# 1) Inference with SD-FT Adapter
echo ">>> [1/2] Inference with SD-FT adapter on distilled_magicoder"
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
# 2) Inference with SFT Adapter
echo ">>> [2/2] Inference with SFT adapter on distilled_magicoder"
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

################################################################################
# 3) Compute HSIC
echo ">>> [3/3] Computing HSIC for distilled vs SD-FT and distilled vs SFT"
python Mutual-Information/compute_hsic_distilled.py magicoder
echo ">>> HSIC computation completed. See results/hsic/magicoder/distilled_magicoder_hsic.log"