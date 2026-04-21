#!/usr/bin/env bash
set -euo pipefail

# Launch 4 local workers on one node (one worker per GPU), reusing shared coord state.
# Useful when you only have one server but want to fully utilize 4 GPUs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${EMBED_CLUSTER_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${EMBED_CLUSTER_PYTHON:-python}"
OUTPUT_ROOT="${EMBED_CLUSTER_OUTPUT_ROOT:-$DATAINF_ROOT/results/embedding_cluster}"
LOCAL_GPUS_CSV="${EMBED_CLUSTER_LOCAL_GPUS:-0,1,2,3}"
NUM_WORKERS="${EMBED_CLUSTER_NUM_WORKERS:-4}"
LOG_DIR="${EMBED_CLUSTER_LOCAL_LOG_DIR:-$OUTPUT_ROOT/_coord}"

IFS=',' read -r -a GPU_LIST <<< "$LOCAL_GPUS_CSV"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  echo "ERROR: EMBED_CLUSTER_LOCAL_GPUS is empty"
  exit 2
fi
if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: EMBED_CLUSTER_NUM_WORKERS must be positive integer"
  exit 2
fi
if [ "$NUM_WORKERS" -lt 1 ]; then
  echo "ERROR: EMBED_CLUSTER_NUM_WORKERS must be >=1"
  exit 2
fi
if [ "$NUM_WORKERS" -gt "${#GPU_LIST[@]}" ]; then
  echo "WARN: NUM_WORKERS=$NUM_WORKERS but only ${#GPU_LIST[@]} GPUs listed; using ${#GPU_LIST[@]} workers."
  NUM_WORKERS="${#GPU_LIST[@]}"
fi

mkdir -p "$LOG_DIR"

echo "[single-node-4gpu] DATAINF_ROOT=$DATAINF_ROOT"
echo "[single-node-4gpu] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[single-node-4gpu] NUM_WORKERS=$NUM_WORKERS"
echo "[single-node-4gpu] GPU_LIST=${GPU_LIST[*]}"

pids=()
for ((wid=0; wid<NUM_WORKERS; wid++)); do
  gpu="${GPU_LIST[$wid]}"
  logf="$LOG_DIR/local_worker${wid}.log"
  echo "[single-node-4gpu] start worker=$wid gpu=$gpu log=$logf"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export EMBED_CLUSTER_DEVICE="cuda:0"
    export EMBED_CLUSTER_WORKER_ID="$wid"
    export EMBED_CLUSTER_NUM_WORKERS="$NUM_WORKERS"
    export EMBED_CLUSTER_DATAINF_ROOT="$DATAINF_ROOT"
    export EMBED_CLUSTER_PYTHON="$PYTHON_BIN"
    export EMBED_CLUSTER_OUTPUT_ROOT="$OUTPUT_ROOT"
    bash "$SCRIPT_DIR/run_embedding_cluster_7nodes_shared_auto.sh"
  ) >"$logf" 2>&1 &
  pids+=($!)
done

on_sig() {
  echo "[single-node-4gpu] received interrupt, terminating workers..."
  for pid in "${pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
trap on_sig INT TERM

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done

if [ "$rc" -ne 0 ]; then
  echo "[single-node-4gpu] at least one worker failed. Check logs under $LOG_DIR"
  exit 1
fi

echo "[single-node-4gpu] all workers completed successfully."

