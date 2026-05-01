#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATAINF_ROOT="${CH3_DATAINF_ROOT:-$REPO_ROOT/DataInf}"
PYTHON_BIN="${CH3_PYTHON:-python}"
OUT_DIR="${CH3_OUTPUT_DIR:-}"

echo "[chapter3_pack] repo_root=$REPO_ROOT"
echo "[chapter3_pack] datainf_root=$DATAINF_ROOT"
if [[ -n "$OUT_DIR" ]]; then
  echo "[chapter3_pack] output_dir=$OUT_DIR"
  "$PYTHON_BIN" "$SCRIPT_DIR/chapter3_collect_latex_pack.py" --datainf_root "$DATAINF_ROOT" --output_dir "$OUT_DIR"
else
  "$PYTHON_BIN" "$SCRIPT_DIR/chapter3_collect_latex_pack.py" --datainf_root "$DATAINF_ROOT"
fi

