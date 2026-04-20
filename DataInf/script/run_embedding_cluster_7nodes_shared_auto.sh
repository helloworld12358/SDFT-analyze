#!/usr/bin/env bash
set -euo pipefail

# 7-node shared-storage automatic pipeline:
# - Stage1 shard scan (epoch5 layer scan)
# - Barrier wait for all shards
# - Stage2 layer selection (leader only, exactly once)
# - Barrier wait for stage2 done
# - Stage3 shard plotting (t-SNE)
# - Optional final barrier (leader waits all stage3 done)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${EMBED_CLUSTER_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${EMBED_CLUSTER_PYTHON:-python}"
OUTPUT_ROOT="${EMBED_CLUSTER_OUTPUT_ROOT:-$DATAINF_ROOT/results/embedding_cluster}"

WORKER_ID="${EMBED_CLUSTER_WORKER_ID:-}"
NUM_WORKERS="${EMBED_CLUSTER_NUM_WORKERS:-7}"
DATASET_LIST_CSV="${EMBED_CLUSTER_DATASET_LIST:-alpaca,dolly,gsm8k,lima,magicoder,openfunction,openhermes}"

STAGE_TIMEOUT_SEC="${EMBED_CLUSTER_STAGE_TIMEOUT_SEC:-172800}" # default 48h
POLL_SEC="${EMBED_CLUSTER_POLL_SEC:-20}"

if [ -z "$WORKER_ID" ]; then
  echo "ERROR: EMBED_CLUSTER_WORKER_ID is required (0-based)."
  echo "Example: export EMBED_CLUSTER_WORKER_ID=0"
  exit 2
fi

if ! [[ "$WORKER_ID" =~ ^[0-9]+$ ]]; then
  echo "ERROR: EMBED_CLUSTER_WORKER_ID must be non-negative integer, got: $WORKER_ID"
  exit 2
fi
if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: EMBED_CLUSTER_NUM_WORKERS must be positive integer, got: $NUM_WORKERS"
  exit 2
fi
if [ "$NUM_WORKERS" -lt 1 ]; then
  echo "ERROR: EMBED_CLUSTER_NUM_WORKERS must be >= 1"
  exit 2
fi
if [ "$WORKER_ID" -ge "$NUM_WORKERS" ]; then
  echo "ERROR: EMBED_CLUSTER_WORKER_ID ($WORKER_ID) must be < EMBED_CLUSTER_NUM_WORKERS ($NUM_WORKERS)"
  exit 2
fi

IFS=',' read -r -a DATASETS <<< "$DATASET_LIST_CSV"
if [ "${#DATASETS[@]}" -eq 0 ]; then
  echo "ERROR: dataset list is empty"
  exit 2
fi

COORD_DIR="${EMBED_CLUSTER_COORD_DIR:-$OUTPUT_ROOT/_coord}"
mkdir -p "$COORD_DIR"

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] [worker=$WORKER_ID] $*"; }

touch_ok() {
  local f="$1"
  mkdir -p "$(dirname "$f")"
  printf "worker=%s\ntime=%s\nhost=%s\n" "$WORKER_ID" "$(ts)" "$(hostname)" > "$f"
}

wait_for_files_count() {
  local glob_pat="$1"
  local expected="$2"
  local fail_flag="$3"
  local timeout="$4"
  local start_ts now elapsed cnt
  start_ts="$(date +%s)"
  while true; do
    if [ -f "$fail_flag" ]; then
      log "Detected fail flag: $fail_flag"
      return 1
    fi
    # shellcheck disable=SC2086
    cnt=$(find "$COORD_DIR" -maxdepth 1 -type f -name "$glob_pat" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$cnt" -ge "$expected" ]; then
      log "Barrier satisfied: pattern=$glob_pat count=$cnt expected=$expected"
      return 0
    fi
    now="$(date +%s)"
    elapsed=$((now - start_ts))
    if [ "$elapsed" -ge "$timeout" ]; then
      log "Timeout waiting barrier: pattern=$glob_pat count=$cnt expected=$expected timeout=$timeout"
      return 1
    fi
    log "Waiting barrier: pattern=$glob_pat count=$cnt expected=$expected elapsed=${elapsed}s"
    sleep "$POLL_SEC"
  done
}

run_stage1_for_dataset() {
  local ds="$1"
  local okf="$COORD_DIR/stage1_done_${ds}.ok"
  [ -f "$okf" ] && { log "Stage1 already marked done for $ds, skip."; return 0; }
  log "Stage1 start for dataset=$ds"
  EMBED_CLUSTER_TRAIN_DATASET="$ds" \
  EMBED_CLUSTER_OUTPUT_ROOT="$OUTPUT_ROOT" \
  EMBED_CLUSTER_DATAINF_ROOT="$DATAINF_ROOT" \
  EMBED_CLUSTER_PYTHON="$PYTHON_BIN" \
  bash "$SCRIPT_DIR/run_embedding_cluster_stage1_scan.sh"
  touch_ok "$okf"
  log "Stage1 done for dataset=$ds"
}

run_stage3_for_dataset() {
  local ds="$1"
  local okf="$COORD_DIR/stage3_done_${ds}.ok"
  [ -f "$okf" ] && { log "Stage3 already marked done for $ds, skip."; return 0; }
  log "Stage3 start for dataset=$ds"
  EMBED_CLUSTER_TRAIN_DATASET="$ds" \
  EMBED_CLUSTER_OUTPUT_ROOT="$OUTPUT_ROOT" \
  EMBED_CLUSTER_DATAINF_ROOT="$DATAINF_ROOT" \
  EMBED_CLUSTER_PYTHON="$PYTHON_BIN" \
  bash "$SCRIPT_DIR/run_embedding_cluster_stage3_tsne.sh"
  touch_ok "$okf"
  log "Stage3 done for dataset=$ds"
}

run_stage2_leader_once() {
  local lockdir="$COORD_DIR/stage2.lock"
  local okf="$COORD_DIR/stage2_done.ok"
  local failf="$COORD_DIR/stage2_failed.flag"

  if [ -f "$okf" ]; then
    log "Stage2 already done by other worker, skip."
    return 0
  fi

  if mkdir "$lockdir" 2>/dev/null; then
    # acquired lock
    log "Acquired stage2 lock; running stage2 layer selection."
    {
      EMBED_CLUSTER_OUTPUT_ROOT="$OUTPUT_ROOT" \
      EMBED_CLUSTER_DATAINF_ROOT="$DATAINF_ROOT" \
      EMBED_CLUSTER_PYTHON="$PYTHON_BIN" \
      bash "$SCRIPT_DIR/run_embedding_cluster_stage2_select_layers.sh"
    } && {
      touch_ok "$okf"
      log "Stage2 completed."
    } || {
      touch_ok "$failf"
      log "Stage2 failed."
      return 1
    }
  else
    log "Stage2 lock already taken by another worker; waiting stage2 done."
  fi
  return 0
}

# worker shard assignment
MY_DATASETS=()
for i in "${!DATASETS[@]}"; do
  if [ $((i % NUM_WORKERS)) -eq "$WORKER_ID" ]; then
    MY_DATASETS+=("${DATASETS[$i]}")
  fi
done

if [ "${#MY_DATASETS[@]}" -eq 0 ]; then
  log "No dataset assigned to this worker (worker_id=$WORKER_ID num_workers=$NUM_WORKERS). Exiting."
  exit 0
fi

log "Assigned datasets: ${MY_DATASETS[*]}"
log "DATAINF_ROOT=$DATAINF_ROOT"
log "OUTPUT_ROOT=$OUTPUT_ROOT"

STAGE1_FAIL="$COORD_DIR/stage1_failed.flag"
STAGE2_FAIL="$COORD_DIR/stage2_failed.flag"
STAGE3_FAIL="$COORD_DIR/stage3_failed.flag"

for ds in "${MY_DATASETS[@]}"; do
  if ! run_stage1_for_dataset "$ds"; then
    touch_ok "$STAGE1_FAIL"
    log "Stage1 failed on dataset=$ds"
    exit 1
  fi
done

# Wait all stage1 done (expected by dataset count, not worker count)
if ! wait_for_files_count "stage1_done_*.ok" "${#DATASETS[@]}" "$STAGE1_FAIL" "$STAGE_TIMEOUT_SEC"; then
  log "Stage1 global barrier failed."
  exit 1
fi

if ! run_stage2_leader_once; then
  exit 1
fi
if ! wait_for_files_count "stage2_done.ok" 1 "$STAGE2_FAIL" "$STAGE_TIMEOUT_SEC"; then
  log "Stage2 barrier failed."
  exit 1
fi

for ds in "${MY_DATASETS[@]}"; do
  if ! run_stage3_for_dataset "$ds"; then
    touch_ok "$STAGE3_FAIL"
    log "Stage3 failed on dataset=$ds"
    exit 1
  fi
done

# optional final barrier
if [ "${EMBED_CLUSTER_WAIT_ALL_STAGE3:-1}" = "1" ]; then
  if ! wait_for_files_count "stage3_done_*.ok" "${#DATASETS[@]}" "$STAGE3_FAIL" "$STAGE_TIMEOUT_SEC"; then
    log "Stage3 global barrier failed."
    exit 1
  fi
fi

log "All assigned tasks finished successfully."

