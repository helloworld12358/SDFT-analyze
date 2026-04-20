#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${EMBED_CLUSTER_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${EMBED_CLUSTER_PYTHON:-python}"
OUTPUT_ROOT="${EMBED_CLUSTER_OUTPUT_ROOT:-}"
TRAIN_DATASET="${EMBED_CLUSTER_TRAIN_DATASET:-}"
INCLUDE_BASE="${EMBED_CLUSTER_INCLUDE_BASE:-1}"
TASKS="${EMBED_CLUSTER_TASKS:-alpaca_eval,gsm8k,humaneval,multiarith,openfunction}"
SAMPLES_PER_TASK="${EMBED_CLUSTER_SAMPLES_PER_TASK:-100}"
SEED="${EMBED_CLUSTER_SEED:-42}"
PLOT_LAYERS="${EMBED_CLUSTER_PLOT_LAYERS:-}"
TOP_K_LAYERS="${EMBED_CLUSTER_TOP_K_LAYERS:-3}"
BATCH_SIZE="${EMBED_CLUSTER_BATCH_SIZE:-8}"
MAX_LENGTH="${EMBED_CLUSTER_MAX_LENGTH:-1024}"
DEVICE="${EMBED_CLUSTER_DEVICE:-auto}"
PREFER_AUTO_ON_FAIL="${EMBED_CLUSTER_PREFER_AUTO_ON_FAIL:-1}"
TSNE_PERPLEXITY="${EMBED_CLUSTER_TSNE_PERPLEXITY:-30}"
TSNE_N_ITER="${EMBED_CLUSTER_TSNE_N_ITER:-1000}"
TSNE_LEARNING_RATE="${EMBED_CLUSTER_TSNE_LEARNING_RATE:-auto}"
TSNE_INIT="${EMBED_CLUSTER_TSNE_INIT:-pca}"
TSNE_METRIC="${EMBED_CLUSTER_TSNE_METRIC:-euclidean}"
PCA_DIM="${EMBED_CLUSTER_PCA_DIM:-50}"

ARGS=(
  --datainf_root "$DATAINF_ROOT"
  --tasks "$TASKS"
  --samples_per_task "$SAMPLES_PER_TASK"
  --seed "$SEED"
  --top_k_layers "$TOP_K_LAYERS"
  --batch_size "$BATCH_SIZE"
  --max_length "$MAX_LENGTH"
  --device "$DEVICE"
  --tsne_perplexity "$TSNE_PERPLEXITY"
  --tsne_n_iter "$TSNE_N_ITER"
  --tsne_learning_rate "$TSNE_LEARNING_RATE"
  --tsne_init "$TSNE_INIT"
  --tsne_metric "$TSNE_METRIC"
  --pca_dim "$PCA_DIM"
)

if [ -n "$OUTPUT_ROOT" ]; then
  ARGS+=(--output_root "$OUTPUT_ROOT")
fi
if [ "$INCLUDE_BASE" = "1" ]; then
  ARGS+=(--include_base)
fi
if [ "$PREFER_AUTO_ON_FAIL" = "1" ]; then
  ARGS+=(--prefer_auto_on_fail)
fi
if [ -n "$TRAIN_DATASET" ]; then
  ARGS+=(--train_dataset "$TRAIN_DATASET")
else
  ARGS+=(--all_train_datasets)
fi
if [ -n "$PLOT_LAYERS" ]; then
  ARGS+=(--layers "$PLOT_LAYERS")
fi

"$PYTHON_BIN" "$SCRIPT_DIR/embedding_cluster_03_plot_selected_layers_tsne.py" "${ARGS[@]}" "$@"

