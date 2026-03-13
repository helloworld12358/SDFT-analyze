# 保存为: pairwise_tasks_epoch_0.sh
#!/usr/bin/env bash
set -euo pipefail

# pairwise_tasks_epoch_0.sh
# 单组合 pairwise 计算（不写 tasks.txt，不传 lora checkpoint，默认 epoch_0）
# Usage: ./pairwise_tasks_epoch_0.sh [MODEL_NAME] [EPOCH_TAG] [LORA_METHOD] [MIN_FREE_PCT] [SLEEP_SEC]
# Defaults: MODEL_NAME=alpaca EPOCH_TAG=epoch_0 LORA_METHOD=sdft MIN_FREE_PCT=15 SLEEP_SEC=3

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATAINF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SDFT_ROOT="$(cd "${DATAINF_ROOT}/../sdft" && pwd 2>/dev/null || echo "${DATAINF_ROOT}/../sdft")"

MODEL_NAME="${1:-alpaca}"
EPOCH_TAG="${2:-epoch_0}"
LORA_METHOD="${3:-sdft}"
MIN_FREE_PCT="${4:-15}"
SLEEP_SEC="${5:-3}"

CKPT_DIR_NAME="checkpoints"
if [ "$EPOCH_TAG" = "epoch_1" ]; then
  CKPT_DIR_NAME="epoch1_checkpoints"
fi

if [ "$LORA_METHOD" = "sft" ]; then
  TRAIN_DATA_FILE="${MODEL_NAME}_train.json"
else
  TRAIN_DATA_FILE="distilled_${MODEL_NAME}.json"
fi

DATASET_NAMES=( "alpaca_eval" "gsm8k" "humaneval" "multiarith" "openfunction" )

BASE_MODEL="${SDFT_ROOT}/model/Llama-2-7b-chat-hf"
TRAIN_DATA="${SDFT_ROOT}/data/${MODEL_NAME}/${TRAIN_DATA_FILE}"

GRADS_BASE_DIR="${DATAINF_ROOT}/output_grad/${EPOCH_TAG}/${LORA_METHOD}/${MODEL_NAME}"
FINAL_ROOT="${DATAINF_ROOT}/result/${MODEL_NAME}/${EPOCH_TAG}/${LORA_METHOD}"
result_DIR="${FINAL_ROOT}/pairwise_result"
LOG_DIR="${FINAL_ROOT}/logs"
mkdir -p "$result_DIR" "$LOG_DIR"

HESSIAN_PY="${DATAINF_ROOT}/src/calc_dataset_similarity.py"
ASSEMBLE_PY="${DATAINF_ROOT}/src/assemble_matrix.py"
PYTHON="python"

# 构建内存中的任务列表（不写磁盘；注意这里不传 --lora_path）
CMDS=()
for i in "${!DATASET_NAMES[@]}"; do
  for j in $(seq "$i" $((${#DATASET_NAMES[@]} - 1))); do
    di=${DATASET_NAMES[$i]}
    dj=${DATASET_NAMES[$j]}
    grad_i="${GRADS_BASE_DIR}/${di}.pt"
    grad_j="${GRADS_BASE_DIR}/${dj}.pt"
    out_json="${result_DIR}/sim_${di}_${dj}.json"
    log_file="${LOG_DIR}/sim_${di}_${dj}.log"
    cmd="$PYTHON \"$HESSIAN_PY\" --base_model_path \"$BASE_MODEL\" --train_dataset_path \"$TRAIN_DATA\" --grad1_path \"$grad_i\" --grad2_path \"$grad_j\" --out_path \"$out_json\""
    CMDS+=("${cmd}||${log_file}")
  done
done

NUM_TASKS=${#CMDS[@]}
if [ "$NUM_TASKS" -eq 0 ]; then
  echo "No tasks to run: no grads found under $GRADS_BASE_DIR" >&2
  exit 1
fi

command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found" >&2; exit 1; }
NGPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$NGPUS" -le 0 ]; then
  echo "No GPUs detected" >&2
  exit 1
fi

get_free_pct_array() {
  nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | \
  awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); total=$1; used=$2; free=total-used; printf "%.2f\n", free/total*100}'
}

PIDS=()
declare -A PID_GPU
declare -A PID_IDX

task_idx=0

cleanup_and_exit() {
  rc="$1"
  for p in "${PIDS[@]:-}"; do
    if [ -n "$p" ] && [[ "$p" =~ ^[0-9]+$ ]]; then
      kill -9 "$p" 2>/dev/null || true
    fi
  done
  exit "$rc"
}
trap 'cleanup_and_exit 1' INT TERM

# 主调度循环：当 GPU 的 free_pct >= MIN_FREE_PCT 时，直接在该 GPU 上启动新进程（不限制并发数）
while [ $task_idx -lt $NUM_TASKS ] || [ ${#PIDS[@]} -gt 0 ]; do
  # 回收已结束的子进程（fail-fast：任何非零退出都会导致 cleanup 并退出）
  new_pids=()
  for pid in "${PIDS[@]}"; do
    if [ -z "$pid" ]; then
      echo "ERROR: empty pid in PIDS" >&2
      cleanup_and_exit 1
    fi
    if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
      echo "ERROR: non-numeric pid in PIDS: '$pid'" >&2
      cleanup_and_exit 1
    fi
    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")
    else
      wait "$pid"
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "ERROR: child PID $pid exited rc=$rc" >&2
        cleanup_and_exit $rc
      fi
      unset PID_GPU["$pid"]
      unset PID_IDX["$pid"]
    fi
  done
  PIDS=("${new_pids[@]}")

  if [ $task_idx -lt $NUM_TASKS ]; then
    readarray -t FREE_PCTS < <(get_free_pct_array)
    launched=0
    for ((g=0; g<NGPUS && task_idx<NUM_TASKS; g++)); do
      free_pct=${FREE_PCTS[$g]}
      enough=$(awk -v fp="$free_pct" -v min="$MIN_FREE_PCT" 'BEGIN{ print (fp+0 >= min) ? 1 : 0 }')
      if [ "$enough" -eq 1 ]; then
        entry="${CMDS[$task_idx]}"
        cmd="${entry%%||*}"
        logf="${entry##*||}"
        mkdir -p "$(dirname "$logf")"
        CUDA_VISIBLE_DEVICES=$g bash -c "$cmd" > "$logf" 2>&1 &
        pid=$!
        if [ -z "$pid" ] || ! [[ "$pid" =~ ^[0-9]+$ ]]; then
          echo "ERROR: failed to launch task_idx=$task_idx on GPU $g" >&2
          echo "COMMAND: $cmd" >&2
          cleanup_and_exit 1
        fi
        PIDS+=("$pid")
        PID_GPU[$pid]=$g
        PID_IDX[$pid]=$task_idx
        task_idx=$((task_idx+1))
        launched=1
        sleep 0.1
      fi
    done
    if [ $launched -eq 0 ]; then
      sleep "$SLEEP_SEC"
    fi
  else
    sleep 1
  fi
done

# 聚合结果（通过 process substitution 传递 grads 列表）
OUT_CSV="${result_DIR}/pairwise_matrix_${MODEL_NAME}_${EPOCH_TAG}_${LORA_METHOD}.csv"
OUT_NPY="${result_DIR}/pairwise_matrix_${MODEL_NAME}_${EPOCH_TAG}_${LORA_METHOD}.npy"
$PYTHON "$ASSEMBLE_PY" --grads_list <(for n in "${DATASET_NAMES[@]}"; do echo "${GRADS_BASE_DIR}/${n}.pt"; done) --result_dir "$result_DIR" --out_csv "$OUT_CSV" --out_npy "$OUT_NPY"

# 最终输出（最小化）
echo "$result_DIR"
