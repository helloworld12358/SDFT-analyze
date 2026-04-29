#!/usr/bin/env bash
set -euo pipefail

# One-command runner for 5 figures:
# - each test task -> one big figure
# - each big figure contains 7 train-dataset bar subplots
# - each subplot has 6 bars: E0-SFT, E0-SDFT, E1-SFT, E1-SDFT, E5-SFT, E5-SDFT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${LOSS_EVAL_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${LOSS_EVAL_PYTHON:-python}"

TRAIN_DATASETS="${LOSS_EVAL_TRAIN_DATASETS:-gsm8k,openfunction,magicoder,alpaca,dolly,lima,openhermes}"
TASKS="${LOSS_EVAL_TASKS:-alpaca_eval,gsm8k,humaneval,multiarith,openfunction}"
OUT_DIR="${LOSS_EVAL_PANEL_OUTPUT_ROOT:-}"
MATRIX_CSV="${LOSS_EVAL_MATRIX_CSV:-}"
FMT="${LOSS_EVAL_PANEL_FORMAT:-pdf}"
YMIN="${LOSS_EVAL_PANEL_YMIN:-}"

EXTRA_ARGS=()
if [ -n "$OUT_DIR" ]; then EXTRA_ARGS+=(--output_dir "$OUT_DIR"); fi
if [ -n "$MATRIX_CSV" ]; then EXTRA_ARGS+=(--matrix_csv "$MATRIX_CSV"); fi
if [ -n "$YMIN" ]; then EXTRA_ARGS+=(--y_min "$YMIN"); fi

echo "[loss_eval_task_panels] datainf_root=$DATAINF_ROOT"
if [ -n "$OUT_DIR" ]; then echo "[loss_eval_task_panels] output_dir=$OUT_DIR"; fi
if [ -n "$MATRIX_CSV" ]; then echo "[loss_eval_task_panels] matrix_csv=$MATRIX_CSV"; fi
if [ -n "$YMIN" ]; then echo "[loss_eval_task_panels] y_min=$YMIN"; fi

"$PYTHON_BIN" "$SCRIPT_DIR/loss_eval_04_plot_testtask_7groups_bar.py" \
  --datainf_root "$DATAINF_ROOT" \
  --train_datasets "$TRAIN_DATASETS" \
  --tasks "$TASKS" \
  --format "$FMT" \
  "${EXTRA_ARGS[@]}" \
  "$@"
