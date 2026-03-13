#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATAINF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODELS=( "gsm8k" "openfunction" "magicoder" "alpaca" "dolly" "lima" "openhermes" )
EPOCHS=( "epoch_0" "epoch_1" "epoch_5" )
METHOD_SD="sdft"
METHOD_SF="sft"

PYTHON_BIN="python3"

for model in "${MODELS[@]}"; do
  for epoch in "${EPOCHS[@]}"; do
    echo ">>> PROCESS: ${model} / ${epoch}"

    DIR_SD="${DATAINF_ROOT}/result/${model}/${epoch}/${METHOD_SD}"
    DIR_SF="${DATAINF_ROOT}/result/${model}/${epoch}/${METHOD_SF}"

    FILE_SD_NPY="${DIR_SD}/pairwise_matrix_${model}_${epoch}_${METHOD_SD}.npy"
    FILE_SF_NPY="${DIR_SF}/pairwise_matrix_${model}_${epoch}_${METHOD_SF}.npy"

    if [ ! -f "$FILE_SD_NPY" ]; then
      echo "ERROR: missing sdft matrix: $FILE_SD_NPY" >&2
      exit 2
    fi
    if [ ! -f "$FILE_SF_NPY" ]; then
      echo "ERROR: missing sft matrix: $FILE_SF_NPY" >&2
      exit 3
    fi

    OUT_DIR="$DIR_SD"
    mkdir -p "$OUT_DIR"

    BASE_TAG="diff_matrix_${model}_${epoch}_${METHOD_SF}_minus_${METHOD_SD}"
    DIFF_NPY="${OUT_DIR}/${BASE_TAG}.npy"
    DIFF_CSV="${OUT_DIR}/${BASE_TAG}.csv"
    DIFF_TXT="${OUT_DIR}/${BASE_TAG}.txt"
    DIFF_EIGVALS_NPY="${OUT_DIR}/diff_eigenvalues_${model}_${epoch}.npy"
    DIFF_EIGVECS_NPY="${OUT_DIR}/diff_eigenvectors_${model}_${epoch}.npy"
    DIFF_EIG_TXT="${OUT_DIR}/diff_eigen_${model}_${epoch}.txt"
    SUMMARY_TXT="${OUT_DIR}/diff_summary_${model}_${epoch}.txt"

    # 使用未加单引号的 heredoc，确保 shell 变量在进入 python 前被展开
    "$PYTHON_BIN" - <<PYCODE
import sys, os
import numpy as np

sd_path = "${FILE_SD_NPY}"
sf_path = "${FILE_SF_NPY}"
out_npy = "${DIFF_NPY}"
out_csv = "${DIFF_CSV}"
out_txt = "${DIFF_TXT}"
eigvals_npy = "${DIFF_EIGVALS_NPY}"
eigvecs_npy = "${DIFF_EIGVECS_NPY}"
eig_txt = "${DIFF_EIG_TXT}"
summary_txt = "${SUMMARY_TXT}"

def l2_normalize_columns(mat):
    M = mat.copy()
    for k in range(M.shape[1]):
        col = M[:, k].astype(np.complex128)
        norm = np.linalg.norm(col)
        if norm == 0 or not np.isfinite(norm):
            continue
        M[:, k] = col / norm
    return M

def fmt_real(x):
    return f"{float(x):.8f}"

def fmt_complex(z):
    if abs(getattr(z, "imag", 0.0)) < 1e-12:
        return fmt_real(z.real)
    else:
        return f"({z.real:.8f}{z.imag:+.8f}j)"

A = np.load(sd_path)
B = np.load(sf_path)

if A.shape != B.shape:
    print("ERROR: shape mismatch: sdft {}, sft {}".format(A.shape, B.shape), file=sys.stderr)
    sys.exit(4)

D = B - A

# scale for display/save as requested (multiply by 1e5)
scale = 1e5
A_scaled = A * scale
B_scaled = B * scale
D_scaled = D * scale

# eigen decomposition on scaled diff
try:
    w, v = np.linalg.eigh(D_scaled)
    idx = np.argsort(-w.real)
    w = w[idx]
    v = v[:, idx]
except Exception:
    w, v = np.linalg.eig(D_scaled)
    idx = np.argsort(-w.real)
    w = w[idx]
    v = v[:, idx]

# L2-normalize eigenvector columns
v_normed = l2_normalize_columns(v)

# overwrite scaled diff and eigs
np.save(out_npy, D_scaled)
# save CSV without scientific notation: fixed-point with 8 decimals
np.savetxt(out_csv, D_scaled, delimiter=",", fmt="%.8f")
np.save(eigvals_npy, w)
np.save(eigvecs_npy, v_normed)

# write readable eigen txt (values + normalized vectors)
with open(eig_txt, "w", encoding="utf-8") as f:
    f.write("Eigenvalues (descending by real part) for scaled diff (sft - sdft) * 1e5:\\n")
    for val in w:
        f.write(fmt_complex(val) + "\\n")
    f.write("\\nEigenvectors (L2-normalized columns corresponding to eigenvalues above):\\n")
    for k in range(v_normed.shape[1]):
        vec = v_normed[:, k]
        f.write(f"eig[{k+1}] = {fmt_complex(w[k])}\\n")
        f.write("vec[%d] = [" % (k+1))
        f.write(", ".join(fmt_complex(x) for x in vec))
        f.write("]\\n\\n")

# write combined summary txt containing scaled sdft, sft, diff, and eiginfo (overwrite)
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("Model: %s\\nEpoch: %s\\nMethod: %s minus %s\\n\\n" % ("${model}", "${epoch}", "${METHOD_SF}", "${METHOD_SD}"))
    f.write("sdft matrix path: %s\\n" % sd_path)
    f.write("sft matrix path: %s\\n" % sf_path)
    f.write("diff matrix path: %s\\n" % out_npy)
    f.write("\\n--- sdft matrix (A) scaled by 1e5 ---\\n")
    for row in A_scaled:
        f.write("  " + ", ".join(fmt_real(x) for x in row) + "\\n")
    f.write("\\n--- sft matrix (B) scaled by 1e5 ---\\n")
    for row in B_scaled:
        f.write("  " + ", ".join(fmt_real(x) for x in row) + "\\n")
    f.write("\\n--- diff matrix (B - A) scaled by 1e5 ---\\n")
    for row in D_scaled:
        f.write("  " + ", ".join(fmt_real(x) for x in row) + "\\n")
    f.write("\\n--- Eigenvalues (desc) of scaled diff ---\\n")
    for val in w:
        f.write("  " + fmt_complex(val) + "\\n")
    f.write("\\n--- Eigenvectors (L2-normalized) ---\\n")
    for k in range(v_normed.shape[1]):
        vec = v_normed[:, k]
        f.write("eig[%d] = %s\\n" % (k+1, fmt_complex(w[k])))
        f.write("  vec: [" + ", ".join(fmt_complex(x) for x in vec) + "]\\n\\n")

print("SAVED_DIFF_NPY:", out_npy)
print("SAVED_DIFF_CSV:", out_csv)
print("SAVED_DIFF_TXT:", out_txt)
print("SAVED_EIG_TXT:", eig_txt)
print("SAVED_SUMMARY:", summary_txt)
PYCODE

    echo "Done: outputs in $OUT_DIR"
    echo "-------------------------------------"
  done
done

echo "ALL DONE."
