#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash DataInf/script/run_chapter4_paper_figures.sh
#
# Optional env:
#   CH4_PYTHON=python
#   CH4_OUTPUT_DIR=/abs/path/to/figures

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${CH4_PYTHON:-python}"
OUT_DIR="${CH4_OUTPUT_DIR:-$REPO_ROOT/figures}"

echo "[chapter4_figures] repo_root=$REPO_ROOT"
echo "[chapter4_figures] output_dir=$OUT_DIR"

"$PY" "$REPO_ROOT/DataInf/script/chapter4_make_paper_figures.py" \
  --output_dir "$OUT_DIR"

echo "[chapter4_figures] generated files:"
ls -lh "$OUT_DIR"/chapter4_* 2>/dev/null || true
