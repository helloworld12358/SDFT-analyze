#!/usr/bin/env bash
set -euo pipefail

# 4-task variant (exclude multiarith), single-node 4-GPU launcher.
# Keeps previous 5-task results untouched by writing to a separate output root by default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${EMBED_CLUSTER_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${EMBED_CLUSTER_PYTHON:-python}"

# Fixed 4-task setting (exclude multiarith)
TASKS_FIXED="${EMBED_CLUSTER_TASKS_FIXED:-alpaca_eval,gsm8k,humaneval,openfunction}"

# Default output root is separate from previous 5-task experiment to avoid overwrite.
OUTPUT_ROOT_DEFAULT="$DATAINF_ROOT/results/embedding_cluster_4tasks_no_multiarith"
OUTPUT_ROOT="${EMBED_CLUSTER_OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}"

ALLOW_BASE_OUTPUT="${EMBED_CLUSTER_ALLOW_BASE_OUTPUT:-0}"
ALLOW_REUSE_OUTPUT="${EMBED_CLUSTER_ALLOW_REUSE_OUTPUT:-1}"

if [ "$ALLOW_BASE_OUTPUT" != "1" ] && [ "$OUTPUT_ROOT" = "$DATAINF_ROOT/results/embedding_cluster" ]; then
  echo "ERROR: OUTPUT_ROOT points to existing 5-task directory:"
  echo "  $OUTPUT_ROOT"
  echo "This script blocks it by default to avoid overwriting previous results."
  echo "Use a different EMBED_CLUSTER_OUTPUT_ROOT, or set EMBED_CLUSTER_ALLOW_BASE_OUTPUT=1 (not recommended)."
  exit 2
fi

if [ -d "$OUTPUT_ROOT" ] && [ "$ALLOW_REUSE_OUTPUT" != "1" ]; then
  echo "ERROR: OUTPUT_ROOT already exists and ALLOW_REUSE_OUTPUT=0"
  echo "  $OUTPUT_ROOT"
  echo "Set EMBED_CLUSTER_ALLOW_REUSE_OUTPUT=1 to resume/reuse, or change OUTPUT_ROOT."
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"

echo "[4tasks-no-multiarith] DATAINF_ROOT=$DATAINF_ROOT"
echo "[4tasks-no-multiarith] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[4tasks-no-multiarith] TASKS=$TASKS_FIXED"

EMBED_CLUSTER_DATAINF_ROOT="$DATAINF_ROOT" \
EMBED_CLUSTER_PYTHON="$PYTHON_BIN" \
EMBED_CLUSTER_OUTPUT_ROOT="$OUTPUT_ROOT" \
EMBED_CLUSTER_TASKS="$TASKS_FIXED" \
bash "$SCRIPT_DIR/run_embedding_cluster_single_node_4gpu.sh" "$@"

