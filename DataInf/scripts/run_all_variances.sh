#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

PYTHON_SCRIPT="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/DataInf/src/compute_loss_variance.py"
BASE_MODEL="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/model/Llama-2-7b-chat-hf"
CHECKPOINT_ROOT_EPOCH1="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/epoch1_checkpoints"
CHECKPOINT_ROOT="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/checkpoints"
DATA_ROOT="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/data"
OUTPUT_ROOT="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/DataInf/results"

MODEL="Llama2_7b"
MODEL_SHORT="Llama2_7b"
EPOCHS=("epoch_0" "epoch_1" "epoch_5")
METHODS=("sdft" "sft")
DATASETS="alpaca_eval,gsm8k,humaneval,multiarith,openfunction"

BATCH_SIZE=8
PYTHON_EXEC="python"
LOG_ROOT="${SCRIPT_DIR}/logs_variance"
mkdir -p "$LOG_ROOT"
mkdir -p "$OUTPUT_ROOT"

if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Error: python script not found at $PYTHON_SCRIPT" >&2
  exit 1
fi

TASKS=()
for EPOCH in "${EPOCHS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    ARGS=(--base_model_path "$BASE_MODEL" --checkpoint_root_epoch1 "$CHECKPOINT_ROOT_EPOCH1" --checkpoint_root "$CHECKPOINT_ROOT" --model_name_short "$MODEL_SHORT" --model "$MODEL" --epoch "$EPOCH" --method "$METHOD" --data_root "$DATA_ROOT" --datasets "$DATASETS" --batch_size "$BATCH_SIZE" --max_length 1024 --output_root "$OUTPUT_ROOT" --verbose)
    TASKS+=("$(printf "%q " "${ARGS[@]}")")
  done
done

TOTAL=${#TASKS[@]}
GPU_COUNT=4 

echo "Total tasks: $TOTAL"
echo "Available GPUs: $GPU_COUNT (Using strategy: Batch Execution)"

for ((i=0; i<TOTAL; i+=GPU_COUNT)); do
    echo "=========================================================="
    echo "Starting Batch starting from task index $i"
    
    # 这一轮循环负责启动 GPU_COUNT 个任务
    for ((g=0; g<GPU_COUNT; g++)); do
        task_idx=$((i + g))
        
        # 如果任务索引超过了总任务数，就停止分发
        if [ $task_idx -ge $TOTAL ]; then
            break
        fi

        LOGFILE="${LOG_ROOT}/task_${task_idx}_gpu${g}_$(date +%s).log"
        ARGS_STR="${TASKS[$task_idx]}"
        
        echo "[LAUNCH] Task $task_idx on GPU $g -> Log: $LOGFILE"
        
        # 核心：后台执行，不等待立即返回
        eval "CUDA_VISIBLE_DEVICES=$g $PYTHON_EXEC \"$PYTHON_SCRIPT\" $ARGS_STR > \"$LOGFILE\" 2>&1 &"
    done

    echo "Waiting for current batch to finish..."
    # 核心：等待当前所有后台任务结束后，再进行下一轮循环
    wait
    echo "Batch finished."
done

echo "All tasks finished. Results placed under $OUTPUT_ROOT"