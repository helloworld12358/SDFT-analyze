#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash DataInf/script/run_thesis_figure_text_fix_and_pack.sh
#
# Optional env vars:
#   THESIS_FIG_DATAINF_ROOT
#   THESIS_FIG_PYTHON
#   THESIS_FIG_OUTPUT_ROOT
#   THESIS_FIG_CHAPTER2_PIPELINE_SOURCE

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATAINF_ROOT="${THESIS_FIG_DATAINF_ROOT:-$REPO_ROOT/DataInf}"
OUTPUT_ROOT="${THESIS_FIG_OUTPUT_ROOT:-$DATAINF_ROOT/results/thesis_figure_text_fix}"

bash "$SCRIPT_DIR/run_thesis_figure_text_fix.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
PKG="$DATAINF_ROOT/results/thesis_figure_text_fix_bundle_${STAMP}.tar.gz"
tar -czf "$PKG" -C "$OUTPUT_ROOT" .
echo "$PKG"

