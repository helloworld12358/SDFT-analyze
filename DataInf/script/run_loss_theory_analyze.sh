#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${LOSS_THEORY_DATAINF_ROOT:-${SCHEMEA_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}}"
PYTHON_BIN="${LOSS_THEORY_PYTHON:-${SCHEMEA_PYTHON:-python}}"

echo "[loss_theory_analyze] datainf_root=${DATAINF_ROOT}"

"$PYTHON_BIN" "$SCRIPT_DIR/loss_theory_02_tail_shape.py" --datainf_root "$DATAINF_ROOT"
"$PYTHON_BIN" "$SCRIPT_DIR/loss_theory_03_mgf_check.py" --datainf_root "$DATAINF_ROOT"
"$PYTHON_BIN" "$SCRIPT_DIR/loss_theory_04_emp_bernstein.py" --datainf_root "$DATAINF_ROOT"
"$PYTHON_BIN" "$SCRIPT_DIR/loss_theory_05_robust_mean.py" --datainf_root "$DATAINF_ROOT"
"$PYTHON_BIN" "$SCRIPT_DIR/loss_theory_06_conditional.py" --datainf_root "$DATAINF_ROOT"
"$PYTHON_BIN" "$SCRIPT_DIR/loss_theory_07_len_ablation.py" --datainf_root "$DATAINF_ROOT"
"$PYTHON_BIN" "$SCRIPT_DIR/loss_theory_08_dependence.py" --datainf_root "$DATAINF_ROOT"
"$PYTHON_BIN" "$SCRIPT_DIR/loss_theory_09_final_report.py" --datainf_root "$DATAINF_ROOT"
"$PYTHON_BIN" "$SCRIPT_DIR/loss_theory_10_combo_matrix_tables.py" --datainf_root "$DATAINF_ROOT"

echo "[loss_theory_analyze] done"
