#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${EMBED_CLUSTER_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${EMBED_CLUSTER_PYTHON:-python}"
OUTPUT_ROOT="${EMBED_CLUSTER_OUTPUT_ROOT:-}"
TOP_K_LAYERS="${EMBED_CLUSTER_TOP_K_LAYERS:-3}"

ARGS=(
  --datainf_root "$DATAINF_ROOT"
  --top_k_layers "$TOP_K_LAYERS"
)
if [ -n "$OUTPUT_ROOT" ]; then
  ARGS+=(--output_root "$OUTPUT_ROOT")
fi

"$PYTHON_BIN" "$SCRIPT_DIR/embedding_cluster_02_select_layers.py" "${ARGS[@]}" "$@"

