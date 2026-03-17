#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/run_schemeA_crossH_all_epochs.sh"
"$SCRIPT_DIR/run_schemeA_mixedH_all_epochs.sh"
"$SCRIPT_DIR/run_schemeA_score_and_final_summary.sh"
