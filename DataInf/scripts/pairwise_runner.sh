#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATAINF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SDFT_ROOT="$(cd "${DATAINF_ROOT}/../sdft" && pwd 2>/dev/null || echo "${DATAINF_ROOT}/../sdft")"

MODEL_NAME="alpaca"
EPOCH_TAG="epoch_5"
LORA_METHOD="sdft"

CKPT_DIR_NAME="checkpoints"
TRAIN_DATA_FILE="distilled_alpaca.json"

DATASET_NAMES=(
    "alpaca_eval"
    "gsm8k"
    "humaneval"
    "multiarith"
    "openfunction"
)

BASE_MODEL="${SDFT_ROOT}/model/Llama-2-7b-chat-hf"
LORA_CHECKPOINT="${SDFT_ROOT}/${CKPT_DIR_NAME}/${MODEL_NAME}/${LORA_METHOD}"
TRAIN_DATA="${SDFT_ROOT}/data/${MODEL_NAME}/${TRAIN_DATA_FILE}"

GRADS_BASE_DIR="${DATAINF_ROOT}/output_grads/${EPOCH_TAG}/${LORA_METHOD}/${MODEL_NAME}"

GRADS=()
for name in "${DATASET_NAMES[@]}"; do
    GRADS+=("${GRADS_BASE_DIR}/${name}.pt")
done

NUM_GRADS=${#GRADS[@]}
NUM_GPUS=4
TASKS_PER_GPU=2       # <=2 as requested (3 may OOM)
MAX_CONCURRENT=$((NUM_GPUS * TASKS_PER_GPU))
PYTHON="python"
HESSIAN_PY="${DATAINF_ROOT}/src/calc_dataset_similarity.py"
ASSEMBLE_PY="${DATAINF_ROOT}/src/assemble_matrix.py"

FINAL_ROOT="${DATAINF_ROOT}/results/${MODEL_NAME}/${EPOCH_TAG}/${LORA_METHOD}"
RESULTS_DIR="${FINAL_ROOT}/pairwise_results"
LOG_DIR="${FINAL_ROOT}/logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

GRADS_LIST_FILE="${FINAL_ROOT}/grads_list.txt"
> "$GRADS_LIST_FILE"
for p in "${GRADS[@]}"; do echo "$p" >> "$GRADS_LIST_FILE"; done

echo "------------ Configuration ------------"
echo "DataInf root: $DATAINF_ROOT"
echo "Model:        $MODEL_NAME"
echo "Epoch:        $EPOCH_TAG"
echo "LoRA method:  $LORA_METHOD"
echo "Checkpoints:  $LORA_CHECKPOINT"
echo "Grads Dir:    $GRADS_BASE_DIR"
echo "Num Grads:    $NUM_GRADS"
echo "Results root: $FINAL_ROOT"
echo "NUM_GPUS:     $NUM_GPUS"
echo "TASKS_PER_GPU:$TASKS_PER_GPU"
echo "MAX_CONC:     $MAX_CONCURRENT"
echo "---------------------------------------"

# bookkeeping structures
declare -a pids=()                   # active pids
declare -A PID_GPU=()                # pid -> gpu id mapping
declare -a GPU_OCC                   # occupancy per gpu
for ((g=0; g<NUM_GPUS; g++)); do GPU_OCC[g]=0; done

# helper: cleanup finished pids and update occupancy
cleanup_finished() {
    local new_pids=()
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
        else
            # reap and get exit code
            if wait "$pid"; then
                rc=0
            else
                rc=$?
            fi
            local gpu=${PID_GPU[$pid]}
            echo "[CLEANUP] PID $pid finished (rc=$rc), freeing GPU $gpu slot."
            # decrement occupancy safely (ensure non-negative)
            if [ "${GPU_OCC[$gpu]}" -gt 0 ]; then
                GPU_OCC[$gpu]=$((GPU_OCC[$gpu]-1))
            else
                GPU_OCC[$gpu]=0
            fi
            unset PID_GPU["$pid"]
        fi
    done
    pids=("${new_pids[@]}")
}

# find a gpu with occupancy < TASKS_PER_GPU; returns it via echo, or empty if none
find_free_gpu() {
    for ((g=0; g<NUM_GPUS; g++)); do
        if [ "${GPU_OCC[$g]}" -lt "$TASKS_PER_GPU" ]; then
            echo "$g"
            return 0
        fi
    done
    echo ""
    return 1
}

task_idx=0

for ((i=0;i<NUM_GRADS;i++)); do
  for ((j=i;j<NUM_GRADS;j++)); do
    out_json="${RESULTS_DIR}/sim_${DATASET_NAMES[$i]}_${DATASET_NAMES[$j]}.json"
    log_file="${LOG_DIR}/sim_${DATASET_NAMES[$i]}_${DATASET_NAMES[$j]}.log"

    # wait until there is a free GPU slot
    while true; do
        cleanup_finished
        free_gpu=$(find_free_gpu)
        if [ -n "$free_gpu" ]; then
            gpu_id=$free_gpu
            break
        fi
        # no free slot yet, sleep a bit then retry
        sleep 2
    done

    echo "[TASK $task_idx] ${DATASET_NAMES[$i]} vs ${DATASET_NAMES[$j]} -> GPU $gpu_id (occ=${GPU_OCC[$gpu_id]})"
    CUDA_VISIBLE_DEVICES=${gpu_id} \
      $PYTHON "$HESSIAN_PY" \
        --base_model_path "$BASE_MODEL" \
        --lora_path "$LORA_CHECKPOINT" \
        --train_dataset_path "$TRAIN_DATA" \
        --grad1_path "${GRADS[$i]}" \
        --grad2_path "${GRADS[$j]}" \
        --out_path "$out_json" \
        > "$log_file" 2>&1 &

    pid=$!
    pids+=("$pid")
    PID_GPU[$pid]=$gpu_id
    GPU_OCC[$gpu_id]=$((GPU_OCC[$gpu_id]+1))
    echo "[LAUNCHED] PID $pid on GPU $gpu_id (new occ=${GPU_OCC[$gpu_id]})"

    task_idx=$((task_idx+1))

    # small stagger to avoid simultaneous spikes
    sleep 1
  done
done

# wait for all remaining children to finish
while [ "${#pids[@]}" -gt 0 ]; do
    cleanup_finished
    if [ "${#pids[@]}" -gt 0 ]; then
        sleep 2
    fi
done

# Assemble matrix
OUT_CSV="${RESULTS_DIR}/pairwise_matrix_${MODEL_NAME}_${EPOCH_TAG}_${LORA_METHOD}.csv"
OUT_NPY="${RESULTS_DIR}/pairwise_matrix_${MODEL_NAME}_${EPOCH_TAG}_${LORA_METHOD}.npy"
$PYTHON "$ASSEMBLE_PY" --grads_list "$GRADS_LIST_FILE" --results_dir "$RESULTS_DIR" --out_csv "$OUT_CSV" --out_npy "$OUT_NPY"

echo "Done. Results in: $RESULTS_DIR"
