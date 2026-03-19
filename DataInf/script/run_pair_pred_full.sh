#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${PAIR_PRED_DATAINF_ROOT:-${SCHEMEA_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}}"
PYTHON_BIN="${PAIR_PRED_PYTHON:-${SCHEMEA_PYTHON:-python}}"

# Predictor mode:
#   manual | grid | lodo
PAIR_PRED_MODE="${PAIR_PRED_MODE:-grid}"
# Backward-compatible: if only PAIR_PRED_EPOCH is set, use it as feature epoch.
PAIR_PRED_FEATURE_EPOCH="${PAIR_PRED_FEATURE_EPOCH:-${PAIR_PRED_EPOCH:-epoch_1}}"
PAIR_PRED_LABEL_EPOCH="${PAIR_PRED_LABEL_EPOCH:-$PAIR_PRED_FEATURE_EPOCH}"
PAIR_PRED_ALPHA="${PAIR_PRED_ALPHA:-1.0}"
PAIR_PRED_BETA="${PAIR_PRED_BETA:-1.0}"
PAIR_PRED_GAMMA="${PAIR_PRED_GAMMA:-1.0}"
PAIR_PRED_GRID_VALUES="${PAIR_PRED_GRID_VALUES:--2,-1,-0.5,0,0.5,1,2}"

"$PYTHON_BIN" "$SCRIPT_DIR/pair_pred_00_inventory_and_loader.py" --datainf_root "$DATAINF_ROOT"
"$PYTHON_BIN" "$SCRIPT_DIR/pair_pred_01_extract_pair_features.py" \
  --datainf_root "$DATAINF_ROOT" \
  --feature_epoch "$PAIR_PRED_FEATURE_EPOCH" \
  --label_epoch "$PAIR_PRED_LABEL_EPOCH"
"$PYTHON_BIN" "$SCRIPT_DIR/pair_pred_02_score_predictor.py" \
  --datainf_root "$DATAINF_ROOT" \
  --feature_epoch "$PAIR_PRED_FEATURE_EPOCH" \
  --label_epoch "$PAIR_PRED_LABEL_EPOCH" \
  --mode "$PAIR_PRED_MODE" \
  --alpha "$PAIR_PRED_ALPHA" \
  --beta "$PAIR_PRED_BETA" \
  --gamma "$PAIR_PRED_GAMMA" \
  --grid_values "$PAIR_PRED_GRID_VALUES"
"$PYTHON_BIN" "$SCRIPT_DIR/pair_pred_03_final_summary.py" \
  --datainf_root "$DATAINF_ROOT" \
  --feature_epoch "$PAIR_PRED_FEATURE_EPOCH" \
  --label_epoch "$PAIR_PRED_LABEL_EPOCH"
