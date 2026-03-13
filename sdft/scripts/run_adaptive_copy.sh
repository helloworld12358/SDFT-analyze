#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BASE_MODEL="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/model/Llama-2-7b-chat-hf"
CHECKPOINTS_ROOT="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/checkpoints"
PY_SCRIPT="${SCRIPT_DIR}/theorem_experiment_random.py"   # 修改后的 Python 脚本路径

OUTPUT_BASE="${PROJECT_ROOT}/experiment_results_random"
mkdir -p "$OUTPUT_BASE"

# GPUs to use (物理 GPU id 列表) — 现在只暴露 0,1,2
GPUS=(0 1 2 3)
N_GPUS=${#GPUS[@]}

# 是否启用 reuse_base_model（在每个 GPU 的进程内部）
REUSE_BASE_MODEL=true

TRAIN_DOMAINS=( "gsm8k" "openfunction" "magicoder" "alpaca" "dolly" "lima" "openhermes" )
CKPT_TYPES=( "sft" "sdft" )

declare -A TEST_PATHS
TEST_PATHS["gsm8k"]="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/data/gsm8k/gsm8k_test.json"
TEST_PATHS["openfunction"]="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/data/openfunction/openfunction_test.json"
TEST_PATHS["humaneval"]="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/data/humanevalpack_test.jsonl"
TEST_PATHS["multiarith"]="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/data/multiarith_test.json"
TEST_PATHS["alpaca_eval"]="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/data/alpaca_eval.json"

echo "[$(date +%T)] Starting per-GPU queue runner (GPUS: ${GPUS[*]})"
echo "[$(date +%T)] Results -> $OUTPUT_BASE"

# pre-checks
if [ ! -d "$BASE_MODEL" ]; then echo "ERROR: Base model missing: $BASE_MODEL"; exit 1; fi
if [ ! -d "$CHECKPOINTS_ROOT" ]; then echo "ERROR: Checkpoints root missing: $CHECKPOINTS_ROOT"; exit 1; fi
if [ ! -f "$PY_SCRIPT" ]; then echo "ERROR: Python script missing: $PY_SCRIPT"; exit 1; fi
for key in "${!TEST_PATHS[@]}"; do
  if [ ! -f "${TEST_PATHS[$key]}" ]; then echo "ERROR: Test dataset missing: ${TEST_PATHS[$key]}"; exit 1; fi
done
for domain in "${TRAIN_DOMAINS[@]}"; do
  for ctype in "${CKPT_TYPES[@]}"; do
    ckpt_dir="${CHECKPOINTS_ROOT}/${domain}/${ctype}"
    if [ ! -d "$ckpt_dir" ]; then
      echo "ERROR: required checkpoint dir missing: $ckpt_dir"
      exit 1
    fi
  done
done

# ---------------------------------------------------------------------------
# is_valid_result: 结果文件存在且包含 delta_norm 且不含 error 标志
# 返回 0 表示有效（跳过），返回 1 表示无效或不存在（需运行）
# ---------------------------------------------------------------------------
is_valid_result() {
  local f="$1"
  [ -f "$f" ] && [ -s "$f" ] || return 1
  python3 - "$f" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    if d.get("error"):
        sys.exit(1)
    if "delta_norm" not in d:
        sys.exit(1)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

# build task list (array of "ckpt_dir|test_file|out_path|tmp_dir")
TASKS=()
SKIPPED_COUNT=0
TOTAL_CANDIDATE=0

for domain in "${TRAIN_DOMAINS[@]}"; do
  for ctype in "${CKPT_TYPES[@]}"; do
    ckpt_dir="${CHECKPOINTS_ROOT}/${domain}/${ctype}"
    for test_key in "${!TEST_PATHS[@]}"; do
      test_file="${TEST_PATHS[$test_key]}"
      out_fn="${domain}_${ctype}__${test_key}.json"
      out_path="${OUTPUT_BASE}/${out_fn}"
      tmp_dir="${OUTPUT_BASE}/${domain}_${ctype}__${test_key}_tmp"

      TOTAL_CANDIDATE=$((TOTAL_CANDIDATE+1))
      if is_valid_result "${out_path}"; then
        echo "[$(date +%T)] SKIP (already valid): ${domain}/${ctype} -> ${test_key} (out=${out_path})"
        SKIPPED_COUNT=$((SKIPPED_COUNT+1))
        # ensure tmp dir removed if any stale
        rm -rf "${tmp_dir}" 2>/dev/null || true
        continue
      fi

      TASKS+=( "${ckpt_dir}|${test_file}|${out_path}|${tmp_dir}" )
    done
  done
done

total_tasks=${#TASKS[@]}
echo "[$(date +%T)] Total candidate tasks: ${TOTAL_CANDIDATE}; queued: ${total_tasks}; skipped: ${SKIPPED_COUNT}"

# per-GPU PID and task index
declare -a GPU_PID
declare -a GPU_TASKIDX
for (( i=0; i< N_GPUS; i++ )); do
  GPU_PID[i]=""
  GPU_TASKIDX[i]=-1
done

TASK_PTR=0
ABORT=0

# trap: 确保脚本退出时尝试 kill 子进程
trap ' 
  echo "[$(date +%T)] Trapping exit: killing child tasks...";
  for (( _g=0; _g< N_GPUS; _g++ )); do
    pid=${GPU_PID[$_g]}
    if [ -n "$pid" ]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
' EXIT

# helper: launch a task on gpu_index
launch_on_gpu() {
  local gpu_index="$1"
  local task_idx="$2"
  IFS='|' read -r ckpt_dir test_file out_path task_tmp_dir <<< "${TASKS[$task_idx]}"

  echo "[$(date +%T)] Launching task ${task_idx}/${total_tasks} on GPU ${GPUS[$gpu_index]}: ckpt=${ckpt_dir}, data=${test_file}"

  # ensure tmp dir exists for task
  mkdir -p "${task_tmp_dir}"

  (
    set +e
    # 清除可能残留的 torchrun/DDP env variables，避免 Python 误以为处于 torchrun 环境并尝试 init process group
    # 还保留 LANG 等常用环境变量
    env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT \
      CUDA_VISIBLE_DEVICES="${GPUS[$gpu_index]}" \
      python3 "${PY_SCRIPT}" \
        --base_model_path "${BASE_MODEL}" \
        --lora_checkpoint_path "${ckpt_dir}" \
        --dataset_path "${test_file}" \
        --output_path "${out_path}" \
        --tmp_dir "${task_tmp_dir}" \
        --n_runs 5 \
        --device "cuda:0" $( [ "${REUSE_BASE_MODEL}" = true ] && echo "--reuse_base_model" || echo "" )
    rc=$?

    # clean tmp_dir after task regardless of rc
    rm -rf "${task_tmp_dir}" 2>/dev/null || true

    if [ $rc -ne 0 ]; then
      echo "[$(date +%T)] ERROR: task ${task_idx} on GPU ${GPUS[$gpu_index]} failed (rc=${rc})." >&2
      exit $rc
    fi
    echo "[$(date +%T)] DONE: task ${task_idx} on GPU ${GPUS[$gpu_index]} completed."
    exit 0
  ) &

  GPU_PID[$gpu_index]=$!
  GPU_TASKIDX[$gpu_index]=$task_idx
  echo "[$(date +%T)] PID ${GPU_PID[$gpu_index]} assigned to GPU ${GPUS[$gpu_index]}"
}

# if nothing to do, exit early
if [ ${total_tasks} -eq 0 ]; then
  echo "[$(date +%T)] No tasks to run (all candidate tasks were already completed). Exiting."
  exit 0
fi

# main loop: fill GPUs initially
for (( g=0; g< N_GPUS && TASK_PTR < total_tasks; g++ )); do
  launch_on_gpu "$g" "$TASK_PTR"
  TASK_PTR=$((TASK_PTR+1))
done

# monitor and dispatch remaining tasks as GPUs become free
while true; do
  # check abort flag
  if [ $ABORT -ne 0 ]; then
    echo "[$(date +%T)] ABORTING: an error occurred previously. Killing running tasks..."
    for (( g=0; g< N_GPUS; g++ )); do
      pid=${GPU_PID[$g]}
      if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
      fi
    done
    wait
    exit 1
  fi

  # check if all tasks assigned and all GPUs idle => finish
  all_done=1
  for (( g=0; g< N_GPUS; g++ )); do
    pid=${GPU_PID[$g]}
    if [ -n "$pid" ]; then
      if kill -0 "$pid" 2>/dev/null; then
        all_done=0
      else
        # process finished -> collect exit status
        wait "$pid"
        rc=$?
        GPU_PID[$g]=""
        finished_idx=${GPU_TASKIDX[$g]}
        GPU_TASKIDX[$g]=-1
        if [ $rc -ne 0 ]; then
          ABORT=1
          break
        fi
        # if there are remaining tasks, launch next on this gpu
        if [ $TASK_PTR -lt $total_tasks ]; then
          launch_on_gpu "$g" "$TASK_PTR"
          TASK_PTR=$((TASK_PTR+1))
          all_done=0
        fi
      fi
    fi
  done

  if [ $ABORT -ne 0 ]; then
    echo "[$(date +%T)] A task failed. Aborting all."
    # kill remaining
    for (( g=0; g< N_GPUS; g++ )); do
      pid=${GPU_PID[$g]}
      if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
      fi
    done
    wait
    exit 1
  fi

  if [ $all_done -eq 1 ] && [ $TASK_PTR -ge $total_tasks ]; then
    echo "[$(date +%T)] All tasks completed successfully."
    break
  fi

  sleep 2
done

# final wait just in case
wait

echo "[$(date +%T)] Finished. Results saved to ${OUTPUT_BASE}"