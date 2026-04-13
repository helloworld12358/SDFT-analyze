#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${LORA_NORM_DATAINF_ROOT:-${SCHEMEA_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}}"
PYTHON_BIN="${LORA_NORM_PYTHON:-${SCHEMEA_PYTHON:-python}}"
OUTPUT_ROOT="${LORA_NORM_OUTPUT_ROOT:-}"

echo "[lora_norm] datainf_root=${DATAINF_ROOT}"
echo "[lora_norm] output_root=${OUTPUT_ROOT:-<default results/lora_norm>}"

"$PYTHON_BIN" "$SCRIPT_DIR/lora_norm_01_collect.py" \
  --datainf_root "$DATAINF_ROOT" \
  --output_root "$OUTPUT_ROOT"

"$PYTHON_BIN" "$SCRIPT_DIR/lora_norm_02_summary_plot.py" \
  --datainf_root "$DATAINF_ROOT" \
  --output_root "$OUTPUT_ROOT"

echo "[lora_norm] done"

