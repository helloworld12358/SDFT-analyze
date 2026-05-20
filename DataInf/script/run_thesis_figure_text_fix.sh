#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash DataInf/script/run_thesis_figure_text_fix.sh
#
# Optional env vars:
#   THESIS_FIG_DATAINF_ROOT
#   THESIS_FIG_PYTHON
#   THESIS_FIG_OUTPUT_ROOT
#   THESIS_FIG_CHAPTER2_PIPELINE_SOURCE

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATAINF_ROOT="${THESIS_FIG_DATAINF_ROOT:-$REPO_ROOT/DataInf}"
PYTHON_BIN="${THESIS_FIG_PYTHON:-python}"
OUTPUT_ROOT="${THESIS_FIG_OUTPUT_ROOT:-$DATAINF_ROOT/results/thesis_figure_text_fix}"
CH2_SRC="${THESIS_FIG_CHAPTER2_PIPELINE_SOURCE:-}"

echo "[thesis_figure_text_fix] repo_root=$REPO_ROOT"
echo "[thesis_figure_text_fix] datainf_root=$DATAINF_ROOT"
echo "[thesis_figure_text_fix] output_root=$OUTPUT_ROOT"

ARGS=(
  --datainf_root "$DATAINF_ROOT"
  --output_root "$OUTPUT_ROOT"
)
if [[ -n "$CH2_SRC" ]]; then
  ARGS+=(--chapter2_pipeline_source "$CH2_SRC")
fi

"$PYTHON_BIN" "$SCRIPT_DIR/thesis_figure_text_fix.py" "${ARGS[@]}"

