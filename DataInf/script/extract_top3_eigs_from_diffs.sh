#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATAINF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# model / epoch lists (可按需修改)
MODELS=( "gsm8k" "openfunction" "magicoder" "alpaca" "dolly" "lima" "openhermes" )
EPOCHS=( "epoch_0" "epoch_1" "epoch_5" )

METHOD_SD="sdft"
METHOD_SF="sft"

PYTHON_BIN="python3"

for model in "${MODELS[@]}"; do
  for epoch in "${EPOCHS[@]}"; do
    echo ">>> PROCESS: ${model} / ${epoch}"

    DIFF_DIR="${DATAINF_ROOT}/result/${model}/${epoch}/${METHOD_SD}"
    DIFF_FILE="${DIFF_DIR}/diff_matrix_${model}_${epoch}_${METHOD_SF}_minus_${METHOD_SD}.npy"

    if [ ! -f "$DIFF_FILE" ]; then
      echo "SKIP: diff file not found: $DIFF_FILE" >&2
      continue
    fi

    OUT_DIR="${DIFF_DIR}/top3_eigs"
    mkdir -p "$OUT_DIR"

    timestamp=$(date +%s)
    OUT_TXT="${OUT_DIR}/top3_submatrix_eigs_${model}_${epoch}_${timestamp}.txt"

    echo "Reading diff: $DIFF_FILE -> writing top3 eigs to: $OUT_TXT"

    "$PYTHON_BIN" - <<PYCODE
import os,sys,math,cmath
import numpy as np

diff_path = r"${DIFF_FILE}"
out_path = r"${OUT_TXT}"

# load
D = np.load(diff_path)

# ensure at least 3x3
if D.ndim != 2 or D.shape[0] < 3 or D.shape[1] < 3:
    raise SystemExit(f"ERROR: diff matrix shape invalid for top3 extraction: {D.shape}")

# extract top-left 3x3
sub = D[:3, :3].astype(np.complex128)

# eigen decomposition (use eig because 3x3 may be non-symmetric)
w, v = np.linalg.eig(sub)

# L2-normalize eigenvector columns
for k in range(v.shape[1]):
    col = v[:, k]
    norm = np.linalg.norm(col)
    if norm == 0 or not np.isfinite(norm):
        # leave as-is
        continue
    v[:, k] = col / norm

# formatting helpers: fixed point (no scientific notation) with 8 decimals
def fmt_real(x):
    # if nearly integer, still print with decimals
    return "{:.8f}".format(float(x))

def fmt_complex(z):
    re = float(np.real(z))
    im = float(np.imag(z))
    if abs(im) < 1e-12:
        return fmt_real(re)
    else:
        # show as a+bi with fixed decimals
        sign = "+" if im >= 0 else "-"
        return "{:.8f}{}{:.8f}j".format(re, sign, abs(im))

# write to out file
with open(out_path, "w", encoding="utf-8") as f:
    f.write("Source diff file: {}\n".format(diff_path))
    f.write("Extracted submatrix: top-left 3x3 (values shown scaled exactly as stored)\n\n")

    f.write("Submatrix (3x3):\n")
    for row in sub:
        f.write("  " + ", ".join(fmt_complex(x) for x in row) + "\n")
    f.write("\n")

    f.write("Eigenvalues (corresponding order to eigenvectors below):\n")
    for idx, val in enumerate(w, start=1):
        f.write(f"  eig[{idx}]: {fmt_complex(val)}\n")
    f.write("\n")

    f.write("Eigenvectors (columns correspond to above eigenvalues). Each vector is L2-normalized.\n")
    for k in range(v.shape[1]):
        vec = v[:, k]
        f.write(f"  vec[{k+1}] = [" + ", ".join(fmt_complex(x) for x in vec) + "]\n")
    f.write("\n")

    # small diagnostics: norms
    f.write("Diagnostics: eigenvector L2 norms (should be 1.0 if normalized):\n")
    for k in range(v.shape[1]):
        nrm = np.linalg.norm(v[:, k])
        f.write(f"  norm vec[{k+1}] = {nrm:.8f}\n")

print(f"WROTE: {out_path}")
PYCODE

  done
done

echo "ALL DONE."
