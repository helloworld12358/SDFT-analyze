#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${LOSS_EVAL_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${LOSS_EVAL_PYTHON:-python}"
METHODS="${LOSS_EVAL_METHODS:-sft,sdft}"
EPOCHS="${LOSS_EVAL_EPOCHS:-epoch_0,epoch_1,epoch_5}"
TASKS="${LOSS_EVAL_TASKS:-alpaca_eval,gsm8k,humaneval,multiarith,openfunction}"
EXTRA_ARGS=()
if [ -n "${LOSS_EVAL_OUTPUT_ROOT:-}" ]; then EXTRA_ARGS+=(--output_root "$LOSS_EVAL_OUTPUT_ROOT"); fi
"$PYTHON_BIN" "$SCRIPT_DIR/loss_eval_02_final_summary.py" \
  --datainf_root "$DATAINF_ROOT" \
  --methods "$METHODS" \
  --epochs "$EPOCHS" \
  --tasks "$TASKS" \
  "${EXTRA_ARGS[@]}" \
  "$@"
"$PYTHON_BIN" "$SCRIPT_DIR/loss_eval_03_plot_curves.py" \
  --datainf_root "$DATAINF_ROOT" \
  --epochs "$EPOCHS" \
  --tasks "$TASKS" \
  "${EXTRA_ARGS[@]}"
