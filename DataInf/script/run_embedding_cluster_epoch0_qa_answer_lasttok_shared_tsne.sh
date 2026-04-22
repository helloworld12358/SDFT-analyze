#!/usr/bin/env bash
set -euo pipefail

# Re-run epoch0 QA answer-last-token t-SNE on SHARED layers
# so SFT/SDFT are directly comparable on exactly the same layers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${EMBED_QA0_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${EMBED_QA0_PYTHON:-python}"

# Existing run (with all-layer metrics already computed)
BASE_OUTPUT_ROOT="${EMBED_QA0_BASE_OUTPUT_ROOT:-$DATAINF_ROOT/results/embedding_cluster_epoch0_qa_answer_lasttok}"
# New output root for aligned-layer t-SNE (avoid overwriting old results)
OUTPUT_ROOT="${EMBED_QA0_OUTPUT_ROOT:-${BASE_OUTPUT_ROOT}_shared_tsne}"

SHARED_LAYERS="${EMBED_QA0_SHARED_LAYERS:-}"  # optional explicit like 16,17,20
SHARED_POLICY="${EMBED_QA0_SHARED_POLICY:-common_quality}" # common_quality|delta_focus|balanced
SHARED_TOP_K="${EMBED_QA0_SHARED_TOP_K:-3}"
SHARED_MIN_LAYER="${EMBED_QA0_SHARED_MIN_LAYER:-1}"
SHARED_MAX_LAYER="${EMBED_QA0_SHARED_MAX_LAYER:-32}"

mkdir -p "$OUTPUT_ROOT/_coord"

if [ -z "${SHARED_LAYERS}" ]; then
  SHARED_LAYERS="$("$PYTHON_BIN" "$SCRIPT_DIR/embedding_cluster_epoch0_qa_02b_pick_shared_layers.py" \
    --output_root "$BASE_OUTPUT_ROOT" \
    --policy "$SHARED_POLICY" \
    --top_k "$SHARED_TOP_K" \
    --min_layer "$SHARED_MIN_LAYER" \
    --max_layer "$SHARED_MAX_LAYER" \
    --out_json "$OUTPUT_ROOT/_coord/shared_layers.json" \
    --out_csv "$OUTPUT_ROOT/_coord/shared_layers_table.csv" \
    --out_txt "$OUTPUT_ROOT/_coord/shared_layers.txt")"
fi

echo "[shared-tsne] BASE_OUTPUT_ROOT=$BASE_OUTPUT_ROOT"
echo "[shared-tsne] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[shared-tsne] SHARED_LAYERS=$SHARED_LAYERS"

export EMBED_QA0_OUTPUT_ROOT="$OUTPUT_ROOT"
export EMBED_QA0_LAYERS="$SHARED_LAYERS"
export EMBED_QA0_TSNE_LAYERS="$SHARED_LAYERS"
export EMBED_QA0_SKIP_DONE="${EMBED_QA0_SKIP_DONE:-0}"
export EMBED_QA0_RUN_SUMMARY="${EMBED_QA0_RUN_SUMMARY:-1}"
export EMBED_QA0_DISABLE_TSNE=0

bash "$SCRIPT_DIR/run_embedding_cluster_epoch0_qa_answer_lasttok_single_node_4gpu.sh"

echo "[shared-tsne] done."
