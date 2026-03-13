#!/usr/bin/env bash
set -euo pipefail

# pairwise_tasks.sh
# 对单个组合 (MODEL_NAME EPOCH_TAG LORA_METHOD) 执行 pairwise 计算并汇总矩阵
# Fail-fast: 一旦启动/回收子进程发现异常即退出并打印错误。

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATAINF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SDFT_ROOT="$(cd "${DATAINF_ROOT}/../sdft" && pwd 2>/dev/null || echo "${DATAINF_ROOT}/../sdft")"

MODEL_NAME="${1:-alpaca}"
EPOCH_TAG="${2:-epoch_5}"
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
LORA_CHECKPOINT="${SDFT_ROOT}/${CKPT_DIR_NAME}/${MODEL_NAME}/${LORA_METHOD}"
TRAIN_DATA="${SDFT_ROOT}/data/${MODEL_NAME}/${TRAIN_DATA_FILE}"

GRADS_BASE_DIR="${DATAINF_ROOT}/output_grads/${EPOCH_TAG}/${LORA_METHOD}/${MODEL_NAME}"
FINAL_ROOT="${DATAINF_ROOT}/results/${MODEL_NAME}/${EPOCH_TAG}/${LORA_METHOD}"
RESULTS_DIR="${FINAL_ROOT}/pairwise_results"
LOG_DIR="${FINAL_ROOT}/logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

HESSIAN_PY="${DATAINF_ROOT}/src/calc_dataset_similarity.py"
ASSEMBLE_PY="${DATAINF_ROOT}/src/assemble_matrix.py"
PYTHON="python"

# 构建内存中的任务列表（不写磁盘）
CMDS=()
for i in "${!DATASET_NAMES[@]}"; do
  for j in $(seq "$i" $((${#DATASET_NAMES[@]} - 1))); do
    di=${DATASET_NAMES[$i]}
    dj=${DATASET_NAMES[$j]}
    grad_i="${GRADS_BASE_DIR}/${di}.pt"
    grad_j="${GRADS_BASE_DIR}/${dj}.pt"
    out_json="${RESULTS_DIR}/sim_${di}_${dj}.json"
    log_file="${LOG_DIR}/sim_${di}_${dj}.log"
    cmd="$PYTHON \"$HESSIAN_PY\" --base_model_path \"$BASE_MODEL\" --lora_path \"$LORA_CHECKPOINT\" --train_dataset_path \"$TRAIN_DATA\" --grad1_path \"$grad_i\" --grad2_path \"$grad_j\" --out_path \"$out_json\""
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

# 当子进程失败或 pid 非法时立即失败
trap 'for p in "${PIDS[@]:-}"; do kill -9 "$p" 2>/dev/null || true; done; exit 1' INT TERM

while [ $task_idx -lt $NUM_TASKS ] || [ ${#PIDS[@]} -gt 0 ]; do
  # 回收子进程：对于每个 pid，先检查它是否是合法数字；若不是，直接报错退出
  new_pids=()

  # <- 修改点：use "${PIDS[@]}" rather than "${PIDS[@]:-}" to avoid an empty-string iteration
  for pid in "${PIDS[@]}"; do
    if [ -z "$pid" ]; then
      echo "ERROR: found empty pid in PIDS array" >&2
      exit 1
    fi
    if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
      echo "ERROR: found non-numeric pid in PIDS array: '$pid'" >&2
      exit 1
    fi

    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")
    else
      # wait 返回子进程 exit code；fail-fast：非 0 即刻退出并打印上下文
      wait "$pid"
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "ERROR: child PID $pid (task_index=${PID_IDX[$pid]:-unknown}) exited with rc=$rc" >&2
        exit $rc
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
        # 启动后立即检查 pid 合法性；若 pid 非数字或为空则立即失败并打印命令
        if [ -z "$pid" ] || ! [[ "$pid" =~ ^[0-9]+$ ]]; then
          echo "ERROR: failed to launch task_idx=$task_idx on GPU $g" >&2
          echo "COMMAND: $cmd" >&2
          exit 1
        fi
        PIDS+=("$pid")
        PID_GPU[$pid]=$g
        PID_IDX[$pid]=$task_idx
        task_idx=$((task_idx+1))
        launched=1
        sleep 0.2
      fi
    done
    if [ $launched -eq 0 ]; then
      sleep "$SLEEP_SEC"
    fi
  else
    sleep 1
  fi
done

# 聚合结果
OUT_CSV="${RESULTS_DIR}/pairwise_matrix_${MODEL_NAME}_${EPOCH_TAG}_${LORA_METHOD}.csv"
OUT_NPY="${RESULTS_DIR}/pairwise_matrix_${MODEL_NAME}_${EPOCH_TAG}_${LORA_METHOD}.npy"
$PYTHON "$ASSEMBLE_PY" --grads_list <(for n in "${DATASET_NAMES[@]}"; do echo "${GRADS_BASE_DIR}/${n}.pt"; done) --results_dir "$RESULTS_DIR" --out_csv "$OUT_CSV" --out_npy "$OUT_NPY"

# 成功时只打印结果目录
echo "$RESULTS_DIR"
