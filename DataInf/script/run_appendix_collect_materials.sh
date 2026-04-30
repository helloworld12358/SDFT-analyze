#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATAINF_ROOT="${APPENDIX_DATAINF_ROOT:-$REPO_ROOT/DataInf}"
PYTHON_BIN="${APPENDIX_PYTHON:-python}"
OUT_DIR="${APPENDIX_OUTPUT_DIR:-$DATAINF_ROOT/results/appendix_materials}"
COPY_TSNE="${APPENDIX_COPY_TSNE_SELECTED:-1}"

echo "[appendix_collect] repo_root=$REPO_ROOT"
echo "[appendix_collect] datainf_root=$DATAINF_ROOT"
echo "[appendix_collect] output_dir=$OUT_DIR"
echo "[appendix_collect] copy_tsne_selected=$COPY_TSNE"

"$PYTHON_BIN" "$SCRIPT_DIR/appendix_collect_materials.py" \
  --repo_root "$REPO_ROOT" \
  --output_dir "$OUT_DIR" \
  --copy_tsne_selected "$COPY_TSNE"
