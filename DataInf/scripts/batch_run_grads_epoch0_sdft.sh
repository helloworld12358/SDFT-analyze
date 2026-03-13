#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_SCRIPT="${SCRIPT_DIR}/../src/save_avg_grad.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python file not found at $PYTHON_SCRIPT"
    exit 1
fi

BASE_MODEL="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/model/Llama-2-7b-chat-hf"
LORA_TYPE="sdft"
DATA_ROOT="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/data"
OUTPUT_ROOT="../output_grads"

BATCH_SIZE=8
NUM_GPUS=4
PYTHON_EXEC="python"
# 这里的 LOG_ROOT 仅作为最后的保底（fallback），或者用来在脚本同级生成结构
LOG_ROOT="./logs_run_save_avg_grad"
mkdir -p "$LOG_ROOT"
mkdir -p "$OUTPUT_ROOT"

MODELS=("alpaca" "gsm8k" "openfunction" "magicoder" "dolly" "lima" "openhermes")

declare -A DATASETS
DATASETS["gsm8k"]="${DATA_ROOT}/gsm8k/gsm8k_test.json"
DATASETS["openfunction"]="${DATA_ROOT}/openfunction/openfunction_test.json"
DATASETS["humaneval"]="${DATA_ROOT}/humanevalpack_test.jsonl"
DATASETS["multiarith"]="${DATA_ROOT}/multiarith_test.json"
DATASETS["alpaca_eval"]="${DATA_ROOT}/alpaca_eval.json"

TASK_ARGS=()
TASK_DESCS=()
TASK_OUTPUTS=()

echo "Generating task list..."

for MODEL_NAME in "${MODELS[@]}"; do
    LORA_PATH=""
    # 保持你 epoch_0 的原有逻辑
    CURRENT_OUTPUT_DIR="${OUTPUT_ROOT}/epoch_0/${LORA_TYPE}/${MODEL_NAME}"
    mkdir -p "$CURRENT_OUTPUT_DIR"

    for DATA_NAME in "${!DATASETS[@]}"; do
        DATA_PATH="${DATASETS[$DATA_NAME]}"
        OUTPUT_FILE="${CURRENT_OUTPUT_DIR}/${DATA_NAME}.pt"
        if [ ! -f "$DATA_PATH" ]; then
            echo "Skip: dataset file not found: $DATA_PATH"
            continue
        fi

        ARGS="--base_model_path $BASE_MODEL --dataset_path $DATA_PATH --output_path $OUTPUT_FILE --batch_size $BATCH_SIZE --lora_target q_proj,v_proj"
        TASK_ARGS+=("$ARGS")
        TASK_DESCS+=("Model:$MODEL_NAME | Data:$DATA_NAME | LORA:INIT_AT_RUNTIME")
        TASK_OUTPUTS+=("$OUTPUT_FILE")
    done
done

TOTAL_TASKS=${#TASK_ARGS[@]}
echo "Total Tasks: $TOTAL_TASKS | Batch Size: $BATCH_SIZE | GPUs: $NUM_GPUS"
if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "No tasks to run. Exiting."
    exit 0
fi

# ------------ 核心修改开始 ------------
SHARED_LOG_DIR=""
if [ ${#TASK_OUTPUTS[@]} -gt 0 ]; then
    SHARED_LOG_DIR=$("$PYTHON_EXEC" -c 'import os,sys
paths = sys.argv[1:]
if not paths:
    print("")
else:
    dirs = [os.path.dirname(p) for p in paths]
    try:
        print(os.path.commonpath(dirs))
    except Exception:
        print(dirs[0] if dirs else "")' "${TASK_OUTPUTS[@]}")
fi

# 修改点 1: 无论是否找到共享目录，都在后面追加 logs_run_save_avg_grad 子目录
# 这样日志就会被收纳进子文件夹，而不是散落在 ../output_grads/epoch_0/sft/ 根目录下
if [ -z "$SHARED_LOG_DIR" ]; then
    LOG_DIR="${LOG_ROOT}/logs_run_save_avg_grad"
else
    LOG_DIR="${SHARED_LOG_DIR}/logs_run_save_avg_grad"
fi

# 修改点 2: 创建目录失败时的回退逻辑也保持一致
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
    echo "Warning: cannot create log dir $LOG_DIR, falling back to local LOG_ROOT"
    LOG_DIR="${LOG_ROOT}/logs_run_save_avg_grad"
    mkdir -p "$LOG_DIR"
fi
# ------------ 核心修改结束 ------------

echo "Logs will be saved to: $LOG_DIR"

trap 'echo "Received interrupt; killing child processes..."; pkill -P $$ || true; exit 1' SIGINT SIGTERM

i=0
while [ $i -lt $TOTAL_TASKS ]; do
    echo ">>> Launching Batch starting at index $i"
    PIDS=()
    LOG_FILES=()
    for ((gpu_slot=0; gpu_slot<NUM_GPUS && i<TOTAL_TASKS; gpu_slot++)); do
        gpu_id=$gpu_slot
        ARGS="${TASK_ARGS[$i]}"
        DESC="${TASK_DESCS[$i]}"
        OUTPATH="${TASK_OUTPUTS[$i]}"

        timestamp=$(date +%s)
        safe_desc=$(echo "$DESC" | tr ' /:' '_' | tr -s '_')
        out_dirname=$(dirname "$OUTPATH")
        out_dir_base=$(basename "$out_dirname")
        LOG_FILE="${LOG_DIR}/task_${i}_${out_dir_base}_${safe_desc}_${timestamp}.log"
        LOG_FILES+=("$LOG_FILE")

        echo "[GPU ${gpu_id}] Launching task #$i : $DESC"
        CUDA_VISIBLE_DEVICES=${gpu_id} \
            $PYTHON_EXEC "$PYTHON_SCRIPT" $ARGS \
            > "$LOG_FILE" 2>&1 &

        pid=$!
        PIDS+=($pid)
        echo "[GPU ${gpu_id}] PID $pid -> log: $LOG_FILE"
        ((i++))
        sleep 5
    done

    echo ">>> Waiting for current batch (PIDs: ${PIDS[*]}) ..."
    for idx in "${!PIDS[@]}"; do
        pid=${PIDS[$idx]}
        logfile=${LOG_FILES[$idx]}
        if wait "$pid"; then
            echo "[OK] PID $pid finished. Log: $logfile"
        else
            exit_code=$?
            echo -e "\033[31m[ERROR] PID $pid exited with code $exit_code. See log: $logfile\033[0m"
        fi
    done

    echo ">>> Batch Finished."
    echo "--------------------------------------------------------"
done

echo "All Done. Logs stored in $LOG_DIR"