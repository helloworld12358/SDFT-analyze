#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAINF_ROOT="${SCHEMEA_DATAINF_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${SCHEMEA_PYTHON:-python}"
"$PYTHON_BIN" "$SCRIPT_DIR/schemeA_06_raw_rewrite_decompose.py" --datainf_root "$DATAINF_ROOT" --all_train_datasets --epoch epoch_5 --feature_method sdft --oracle_mode mixed "$@"
