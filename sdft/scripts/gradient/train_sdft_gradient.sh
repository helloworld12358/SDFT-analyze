#!/bin/bash
set -euo pipefail

MODEL_PATH="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/model/Llama-2-7b-chat-hf"
PYTHON_SCRIPT="analyze_gradients_llama_factory.py"
MERGE_SCRIPT="merge_gradients.py"

CUDA_VISIBLE_DEVICES="0,1,2,3"
NPROC_PER_NODE=4
MASTER_PORT=12345
SPLIT="train"
EXTRA_TRAIN_ARGS=( --per_device_train_batch_size 2 --gradient_accumulation_steps 4 --bf16 --ddp_find_unused_parameters False )

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

DATASETS=("gsm8k" "openfunction" "magicoder" "alpaca" "dolly" "lima" "openhermes")

for TRAIN_DATASET in "${DATASETS[@]}"; do
    echo "=== Processing dataset: ${TRAIN_DATASET} ==="
    
    OUTPUT_FOLDER="analysis/${TRAIN_DATASET}/gradient_analysis_original_sdft"
    
    if [[ "${TRAIN_DATASET}" == "gsm8k" ]]; then
        TEMPLATE="gsm8k"
    else
        TEMPLATE="alpaca"
    fi
    
    rm -rf "${OUTPUT_FOLDER}"
    mkdir -p "${OUTPUT_FOLDER}"
    
    CMD=( torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" "${PYTHON_SCRIPT}"
          --output_dir "${OUTPUT_FOLDER}"
          --split "${SPLIT}"
          --model_name_or_path "${MODEL_PATH}"
          --dataset "distilled_${TRAIN_DATASET}"
          --template "${TEMPLATE}"
    )
    
    for a in "${EXTRA_TRAIN_ARGS[@]}"; do
        CMD+=( "${a}" )
    done
    
    echo "Model: ${MODEL_PATH}"
    echo "Dataset: distilled_${TRAIN_DATASET}"
    echo "Template: ${TEMPLATE}"
    echo "Output dir: ${OUTPUT_FOLDER}"
    echo "-------------------------------------------------"
    
    "${CMD[@]}"
    
    EXITCODE=$?
    if [ "${EXITCODE}" -ne 0 ]; then
        echo "[Error] ${PYTHON_SCRIPT} failed for ${TRAIN_DATASET} with code ${EXITCODE}"
        exit "${EXITCODE}"
    fi
    
    echo "Gradient analysis for ${TRAIN_DATASET} finished successfully."
    echo "Now starting merge step for ${TRAIN_DATASET}..."
    
    if [[ ! -f "${MERGE_SCRIPT}" ]]; then
        echo "[Error] merge script not found at ${MERGE_SCRIPT}"
        exit 2
    fi
    
    python "${MERGE_SCRIPT}" --output_dir "${OUTPUT_FOLDER}"
    MERGE_EXIT=$?
    if [ "${MERGE_EXIT}" -ne 0 ]; then
        echo "[Error] merge_gradients.py failed for ${TRAIN_DATASET} with exit code ${MERGE_EXIT}"
        exit "${MERGE_EXIT}"
    fi
    
    echo "Merge completed for ${TRAIN_DATASET}. Results at: ${OUTPUT_FOLDER}/merged"
    echo "=== ${TRAIN_DATASET} processing completed ==="
    echo ""
done

echo "All datasets processed successfully!"