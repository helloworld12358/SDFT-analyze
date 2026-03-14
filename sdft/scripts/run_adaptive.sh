#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "${SCRIPT_DIR}/../model" ] && [ -d "${SCRIPT_DIR}/../data" ]; then
  SDFT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  SDFT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

LOG_DIR="${PROJECT_ROOT}/tmp_logs"
mkdir -p "$LOG_DIR"

BASE_MODEL="${SDFT_ROOT}/model/Llama-2-7b-chat-hf"
CHECKPOINTS_ROOT="${SDFT_ROOT}/checkpoints"
PY_SCRIPT="${SCRIPT_DIR}/theorem_experiment_lasttry.py"

OUTPUT_BASE="${PROJECT_ROOT}/experiment_results"
mkdir -p "$OUTPUT_BASE"

GPU_IDS=(0 1 2 3)

NEED_MEM=25000
MEM_DECREASE_THRESHOLD=10000
LAUNCH_WAIT_TIMEOUT=60

TRAIN_DOMAINS=( "gsm8k" "openfunction" "magicoder" "alpaca" "dolly" "lima" "openhermes" )
CKPT_TYPES=( "sft" "sdft" )

declare -A TEST_PATHS
TEST_PATHS["gsm8k"]="${SDFT_ROOT}/data/gsm8k/gsm8k_test.json"
TEST_PATHS["openfunction"]="${SDFT_ROOT}/data/openfunction/openfunction_test.json"
TEST_PATHS["humaneval"]="${SDFT_ROOT}/data/humanevalpack_test.jsonl"
TEST_PATHS["multiarith"]="${SDFT_ROOT}/data/multiarith_test.json"
TEST_PATHS["alpaca_eval"]="${SDFT_ROOT}/data/alpaca_eval.json"

LOG_FILE="${LOG_DIR}/run_adaptive_full_$$.log"
JOB_FILE="${LOG_DIR}/run_adaptive_full_jobs_$$.tsv"
: > "$LOG_FILE"
: > "$JOB_FILE"

echo "[$(date +%T)] Starting run_adaptive_full" | tee -a "$LOG_FILE"
echo "[$(date +%T)] Logs are saved in: $LOG_DIR" | tee -a "$LOG_FILE"
echo "[$(date +%T)] Results will be saved to: $OUTPUT_BASE" | tee -a "$LOG_FILE"

if [ ! -d "$BASE_MODEL" ]; then echo "ERROR: Base model missing: $BASE_MODEL" | tee -a "$LOG_FILE"; exit 1; fi
if [ ! -d "$CHECKPOINTS_ROOT" ]; then echo "ERROR: Checkpoints root missing: $CHECKPOINTS_ROOT" | tee -a "$LOG_FILE"; exit 1; fi
if [ ! -f "$PY_SCRIPT" ]; then echo "ERROR: Python script missing: $PY_SCRIPT" | tee -a "$LOG_FILE"; exit 1; fi
for key in "${!TEST_PATHS[@]}"; do
  if [ ! -f "${TEST_PATHS[$key]}" ]; then echo "ERROR: Test dataset missing: ${TEST_PATHS[$key]}" | tee -a "$LOG_FILE"; exit 1; fi
done

TASKS=()
for domain in "${TRAIN_DOMAINS[@]}"; do
  for ctype in "${CKPT_TYPES[@]}"; do
    ckpt_dir="${CHECKPOINTS_ROOT}/${domain}/${ctype}"
    if [ ! -d "$ckpt_dir" ]; then
      echo "ERROR: required checkpoint dir missing: $ckpt_dir" | tee -a "$LOG_FILE"
      exit 1
    fi
    for test_key in "${!TEST_PATHS[@]}"; do
      test_file="${TEST_PATHS[$test_key]}"
      TASKS+=("${ckpt_dir}|${test_file}|${test_key}")
    done
  done
done

NUM_TASKS=${#TASKS[@]}
echo "[$(date +%T)] Generated ${NUM_TASKS} tasks" | tee -a "$LOG_FILE"

get_free_mem() {
  local gid="$1"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gid" 2>/dev/null | head -n1 | tr -d ' '
}

record_job() {
  local pid="$1"; local gpu="$2"
  echo -e "${pid}\t${gpu}" >> "$JOB_FILE"
}

cleanup_jobfile() {
  local tmp="${JOB_FILE}.tmp"
  : > "$tmp"
  if [ -f "$JOB_FILE" ]; then
    while read -r line; do
      pid=$(echo "$line" | awk '{print $1}')
      gpu=$(echo "$line" | awk '{print $2}')
      if kill -0 "$pid" 2>/dev/null; then
        echo -e "${pid}\t${gpu}" >> "$tmp"
      fi
    done < "$JOB_FILE"
    mv "$tmp" "$JOB_FILE"
  fi
}

count_running_on_gpu() {
  cleanup_jobfile
  if [ ! -f "$JOB_FILE" ]; then echo 0; return; fi
  awk -v g="$1" '$2==g {cnt++} END{print (cnt+0)}' "$JOB_FILE"
}

launch_and_wait() {
  local gpu="$1"
  local index="$2"

  IFS='|' read -r ckpt_dir test_file test_key <<< "${TASKS[$index]}"
  
  local ckpt_name=$(basename "$ckpt_dir")
  local parent_dir=$(dirname "$ckpt_dir")
  local domain_name=$(basename "$parent_dir")
  
  local result_filename="${domain_name}_${ckpt_name}__${test_key}.json"
  local full_output_path="${OUTPUT_BASE}/${result_filename}"

  echo "[$(date +%T)] Launching task idx=$index" | tee -a "$LOG_FILE"
  echo "  CKPT: $domain_name / $ckpt_name" | tee -a "$LOG_FILE"
  echo "  TARGET: $full_output_path" | tee -a "$LOG_FILE"
  
  initial_free=$(get_free_mem "$gpu")
  if ! [[ "$initial_free" =~ ^[0-9]+$ ]]; then initial_free=0; fi

  CUDA_VISIBLE_DEVICES="$gpu" python3 "$PY_SCRIPT" \
    --base_model_path "$BASE_MODEL" \
    --lora_checkpoint_path "$ckpt_dir" \
    --dataset_path "$test_file" \
    --output_path "$full_output_path" \
    --device "cuda:0" >> "${LOG_FILE}.gpu${gpu}" 2>&1 &

  pid=$!
  echo "[$(date +%T)] Started PID $pid on visible GPU $gpu" | tee -a "$LOG_FILE"
  record_job "$pid" "$gpu"

  waited=0
  while [ "$waited" -lt "$LAUNCH_WAIT_TIMEOUT" ]; do
    sleep 1
    waited=$((waited+1))
    cur_free=$(get_free_mem "$gpu")
    if ! [[ "$cur_free" =~ ^[0-9]+$ ]]; then cur_free=0; fi
    diff=$((initial_free - cur_free))
    if [ "$diff" -ge "$MEM_DECREASE_THRESHOLD" ]; then
      echo "[$(date +%T)] Detected mem decrease on GPU $gpu: ${diff} MB" >> "$LOG_FILE"
      break
    fi
  done

  if [ "$waited" -ge "$LAUNCH_WAIT_TIMEOUT" ]; then
    echo "[$(date +%T)] Warning: timeout waiting for mem decrease on GPU $gpu (PID $pid)" >> "$LOG_FILE"
  fi
}

task_idx=0
while [ "$task_idx" -lt "$NUM_TASKS" ]; do
  launched_in_pass=0
  for gpu in "${GPU_IDS[@]}"; do
    if [ "$task_idx" -ge "$NUM_TASKS" ]; then break; fi
    free_mem=$(get_free_mem "$gpu")
    if ! [[ "$free_mem" =~ ^[0-9]+$ ]]; then free_mem=0; fi

    allowed_slots=0
    if [ "$free_mem" -gt "$NEED_MEM" ]; then
      allowed_slots=$(( free_mem / NEED_MEM ))
    fi
    running=$(count_running_on_gpu "$gpu")
    if [ "$allowed_slots" -ge 1 ] && [ "$running" -lt "$allowed_slots" ]; then
      launch_and_wait "$gpu" "$task_idx"
      task_idx=$((task_idx + 1))
      launched_in_pass=1
    fi
  done

  if [ "$launched_in_pass" -eq 0 ]; then
    sleep 5
  else
    sleep 1
  fi
done

echo "[$(date +%T)] All tasks dispatched. Waiting for remaining jobs..." | tee -a "$LOG_FILE"
while true; do
  cleanup_jobfile
  if [ ! -f "$JOB_FILE" ]; then break; fi
  remaining=$(wc -l < "$JOB_FILE" 2>/dev/null || echo 0)
  if [ "$remaining" -le 0 ]; then break; fi
  echo "[$(date +%T)] Waiting for $remaining running tasks..." | tee -a "$LOG_FILE"
  sleep 15
done

echo "[$(date +%T)] All jobs finished. Results saved to $OUTPUT_BASE" | tee -a "$LOG_FILE"
rm -f "$JOB_FILE"