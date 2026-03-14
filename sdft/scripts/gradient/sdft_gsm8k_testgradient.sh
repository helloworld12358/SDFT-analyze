#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "${SCRIPT_DIR}/../model" ] && [ -d "${SCRIPT_DIR}/../data" ]; then
  SDFT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  SDFT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
set -euo pipefail

MODEL_PATH="${SDFT_ROOT}/model/Llama-2-7b-chat-hf"
PYTHON_SCRIPT="analyze_gradients_llama_factory.py"
MERGE_SCRIPT="merge_gradients.py"

CUDA_VISIBLE_DEVICES="0,1,2,3"
NPROC_PER_NODE=4
MASTER_PORT=12345
SPLIT="test"
EXTRA_TRAIN_ARGS=( --per_device_train_batch_size 2 --gradient_accumulation_steps 4 --bf16 --ddp_find_unused_parameters False )

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

ADAPTERS=("gsm8k" "openfunction" "magicoder" "alpaca" "dolly" "lima" "openhermes")

for ADAPTER_NAME in "${ADAPTERS[@]}"; do
    echo "=== Processing adapter: ${ADAPTER_NAME} ==="
    
    TRAIN_DATASET="gsm8k"
    OUTPUT_FOLDER="analysis/${ADAPTER_NAME}/gradient_analysis_original_sdft_gsm8ktest"
    
    rm -rf "${OUTPUT_FOLDER}"
    mkdir -p "${OUTPUT_FOLDER}"
    
    CMD=( torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" "${PYTHON_SCRIPT}"
          --output_dir "${OUTPUT_FOLDER}"
          --split "${SPLIT}"
          --model_name_or_path "${MODEL_PATH}"
          --dataset "${TRAIN_DATASET}_test"
          --template gsm8k
    )
    
    for a in "${EXTRA_TRAIN_ARGS[@]}"; do
        CMD+=( "${a}" )
    done
    
    echo "=== Running gradient analysis ==="
    echo "Model: ${MODEL_PATH}"
    echo "Dataset: ${TRAIN_DATASET}_test"
    echo "Output dir: ${OUTPUT_FOLDER}"
    echo "Torchrun nproc_per_node: ${NPROC_PER_NODE}, master_port: ${MASTER_PORT}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "Extra train args: ${EXTRA_TRAIN_ARGS[*]}"
    echo "-------------------------------------------------"
    
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
    echo "=== ${ADAPTER_NAME} processing completed ==="
    echo ""
done

echo "All adapters processed successfully!"