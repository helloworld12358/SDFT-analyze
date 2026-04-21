#!/usr/bin/env bash
set -euo pipefail

# Epoch0 + QA(full) + answer-last-token embedding clustering
# Single node multi-GPU launcher.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${EMBED_QA0_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${EMBED_QA0_PYTHON:-python}"
OUTPUT_ROOT="${EMBED_QA0_OUTPUT_ROOT:-$DATAINF_ROOT/results/embedding_cluster_epoch0_qa_answer_lasttok}"

FAMILIES_CSV="${EMBED_QA0_FAMILIES:-sft,sdft}"
SEEDS_CSV="${EMBED_QA0_SEEDS:-42,43,44}"
TRAIN_DATASETS_CSV="${EMBED_QA0_TRAIN_DATASETS:-gsm8k,openfunction,magicoder,alpaca,dolly,lima,openhermes}"
SAMPLES_PER_CLASS="${EMBED_QA0_SAMPLES_PER_CLASS:-500}"
LAYERS="${EMBED_QA0_LAYERS:-all}"

MAX_LENGTH="${EMBED_QA0_MAX_LENGTH:-1024}"
BATCH_SIZE="${EMBED_QA0_BATCH_SIZE:-0}"
MAX_PROBE_BATCH="${EMBED_QA0_MAX_PROBE_BATCH:-256}"
DISABLE_AUTO_TUNE_BATCH="${EMBED_QA0_DISABLE_AUTO_TUNE_BATCH:-0}"

DISABLE_TSNE="${EMBED_QA0_DISABLE_TSNE:-0}"
TSNE_LAYERS="${EMBED_QA0_TSNE_LAYERS:-}"
TSNE_TOP_K_LAYERS="${EMBED_QA0_TSNE_TOP_K_LAYERS:-3}"
TSNE_PERPLEXITY="${EMBED_QA0_TSNE_PERPLEXITY:-30}"
TSNE_N_ITER="${EMBED_QA0_TSNE_N_ITER:-1000}"
TSNE_LEARNING_RATE="${EMBED_QA0_TSNE_LEARNING_RATE:-auto}"
TSNE_INIT="${EMBED_QA0_TSNE_INIT:-pca}"
TSNE_METRIC="${EMBED_QA0_TSNE_METRIC:-euclidean}"
PCA_DIM="${EMBED_QA0_PCA_DIM:-50}"

LOCAL_GPUS_CSV="${EMBED_QA0_LOCAL_GPUS:-0,1,2,3}"
RUN_SUMMARY="${EMBED_QA0_RUN_SUMMARY:-1}"
ALLOW_REUSE_OUTPUT="${EMBED_QA0_ALLOW_REUSE_OUTPUT:-1}"
SKIP_DONE="${EMBED_QA0_SKIP_DONE:-1}"

if [ -d "$OUTPUT_ROOT" ] && [ "$ALLOW_REUSE_OUTPUT" != "1" ]; then
  echo "ERROR: output exists and EMBED_QA0_ALLOW_REUSE_OUTPUT=0"
  echo "  $OUTPUT_ROOT"
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT/_coord"

IFS=',' read -r -a FAMILIES <<< "$FAMILIES_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
IFS=',' read -r -a GPUS <<< "$LOCAL_GPUS_CSV"

if [ "${#FAMILIES[@]}" -eq 0 ] || [ "${#SEEDS[@]}" -eq 0 ] || [ "${#GPUS[@]}" -eq 0 ]; then
  echo "ERROR: empty families/seeds/gpus."
  exit 2
fi

echo "[epoch0-qa] DATAINF_ROOT=$DATAINF_ROOT"
echo "[epoch0-qa] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[epoch0-qa] FAMILIES=${FAMILIES[*]}"
echo "[epoch0-qa] SEEDS=${SEEDS[*]}"
echo "[epoch0-qa] GPUS=${GPUS[*]}"

max_workers="${#GPUS[@]}"
running=0
failures=0

declare -a pids

job_done() {
  local family="$1"
  local seed="$2"
  local f1="$OUTPUT_ROOT/jobs/$family/seed_$seed/layer_metrics_${family}_seed${seed}.csv"
  local f2="$OUTPUT_ROOT/jobs/$family/seed_$seed/layer_metrics_${family}_seed${seed}.json"
  [ -s "$f1" ] && [ -s "$f2" ]
}

launch_one() {
  local family="$1"
  local seed="$2"
  local gpu="$3"
  local logf="$OUTPUT_ROOT/_coord/job_${family}_seed${seed}.log"
  echo "[epoch0-qa] launch family=$family seed=$seed gpu=$gpu log=$logf"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_BIN" "$SCRIPT_DIR/embedding_cluster_epoch0_qa_01_run_job.py" \
      --datainf_root "$DATAINF_ROOT" \
      --output_root "$OUTPUT_ROOT" \
      --family "$family" \
      --seed "$seed" \
      --train_datasets "$TRAIN_DATASETS_CSV" \
      --samples_per_class "$SAMPLES_PER_CLASS" \
      --layers "$LAYERS" \
      --batch_size "$BATCH_SIZE" \
      --max_length "$MAX_LENGTH" \
      --max_probe_batch "$MAX_PROBE_BATCH" \
      --device "cuda:0" \
      --tsne_top_k_layers "$TSNE_TOP_K_LAYERS" \
      --tsne_perplexity "$TSNE_PERPLEXITY" \
      --tsne_n_iter "$TSNE_N_ITER" \
      --tsne_learning_rate "$TSNE_LEARNING_RATE" \
      --tsne_init "$TSNE_INIT" \
      --tsne_metric "$TSNE_METRIC" \
      --pca_dim "$PCA_DIM" \
      ${TSNE_LAYERS:+--tsne_layers "$TSNE_LAYERS"} \
      $([ "$DISABLE_AUTO_TUNE_BATCH" = "1" ] && echo "--disable_auto_tune_batch") \
      $([ "$DISABLE_TSNE" = "1" ] && echo "--disable_tsne")
  ) >"$logf" 2>&1 &
  pids+=($!)
  running=$((running + 1))
}

wait_one() {
  if [ "$running" -gt 0 ]; then
    if wait -n; then
      :
    else
      echo "[epoch0-qa] one job failed; waiting remaining jobs..."
      failures=$((failures + 1))
    fi
    running=$((running - 1))
  fi
}

gpu_idx=0
for family in "${FAMILIES[@]}"; do
  family="$(echo "$family" | xargs)"
  [ -z "$family" ] && continue
  for seed in "${SEEDS[@]}"; do
    seed="$(echo "$seed" | xargs)"
    [ -z "$seed" ] && continue
    if [ "$SKIP_DONE" = "1" ] && job_done "$family" "$seed"; then
      echo "[epoch0-qa] skip done job family=$family seed=$seed"
      continue
    fi
    while [ "$running" -ge "$max_workers" ]; do
      wait_one
    done
    gpu="${GPUS[$((gpu_idx % max_workers))]}"
    gpu_idx=$((gpu_idx + 1))
    launch_one "$family" "$seed" "$gpu"
  done
done

while [ "$running" -gt 0 ]; do
  wait_one
done

if [ "$failures" -eq 0 ] && [ "$RUN_SUMMARY" = "1" ]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/embedding_cluster_epoch0_qa_02_summary.py" \
    --datainf_root "$DATAINF_ROOT" \
    --output_root "$OUTPUT_ROOT"
fi

if [ "$failures" -gt 0 ]; then
  echo "[epoch0-qa] done with failures=$failures"
  exit 1
fi

echo "[epoch0-qa] done."
