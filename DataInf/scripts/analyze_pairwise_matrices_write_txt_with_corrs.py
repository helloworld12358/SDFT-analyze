#!/usr/bin/env python3
import os
import sys
import json
from typing import Optional, Tuple
import numpy as np

def load_matrix_from_dir(results_dir: str, model: str, epoch: str, method: str) -> Optional[np.ndarray]:
    base = os.path.join(results_dir, model, epoch, method)
    if not os.path.isdir(base):
        return None
    fname_npy = os.path.join(base, f"pairwise_matrix_{model}_{epoch}_{method}.npy")
    fname_csv = os.path.join(base, f"pairwise_matrix_{model}_{epoch}_{method}.csv")
    if os.path.isfile(fname_npy):
        try:
            return np.load(fname_npy)
        except Exception:
            pass
    if os.path.isfile(fname_csv):
        try:
            return np.loadtxt(fname_csv, delimiter=',')
        except Exception:
            pass
    pair_dir = os.path.join(base, "pairwise_results")
    if os.path.isdir(pair_dir):
        files = [f for f in os.listdir(pair_dir) if f.startswith("sim_") and f.endswith(".json")]
        if not files:
            return None
        pairs = {}
        labels = set()
        for fn in files:
            p = os.path.join(pair_dir, fn)
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                g1 = data.get("grad1")
                g2 = data.get("grad2")
                score = float(data.get("score", 0.0))
                if g1 is None or g2 is None:
                    continue
                labels.add(g1)
                labels.add(g2)
                pairs[(g1, g2)] = score
                pairs[(g2, g1)] = score
            except Exception:
                continue
        labels = sorted(labels)
        if not labels:
            return None
        n = len(labels)
        mat = np.zeros((n, n), dtype=float)
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                mat[i, j] = pairs.get((li, lj), 0.0)
        return mat
    return None

def is_symmetric(mat: np.ndarray, tol: float = 1e-8) -> bool:
    if mat.shape[0] != mat.shape[1]:
        return False
    return np.allclose(mat, mat.T, atol=tol, rtol=0)

def compute_eigvals_vecs(mat: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if mat is None or mat.size == 0:
        return np.array([]), np.array([[]])
    if is_symmetric(mat):
        w, v = np.linalg.eigh(mat)
        idx = np.argsort(-w.real)
        w = w[idx]
        v = v[:, idx]
        return w, v
    else:
        w, v = np.linalg.eig(mat)
        idx = np.argsort(-w.real)
        w = w[idx]
        v = v[:, idx]
        return w, v

def pretty_matrix_text(mat: Optional[np.ndarray]) -> str:
    if mat is None:
        return "MISSING\n"
    lines = []
    lines.append(f"shape: {mat.shape}")
    with np.printoptions(precision=6, suppress=True):
        for row in mat:
            lines.append("  " + "  ".join(f"{float(x):12.6f}" for x in row))
    return "\n".join(lines)

def complex_to_str(z, precision=6):
    if getattr(z, "imag", 0.0) == 0 or abs(getattr(z, "imag", 0.0)) < 1e-12:
        return f"{float(getattr(z, 'real', z)):.{precision}f}"
    else:
        return f"({z.real:.{precision}f}{z.imag:+.{precision}f}j)"

def vector_to_str(vec, precision=6):
    elems = [complex_to_str(v, precision) for v in np.asarray(vec).reshape(-1)]
    return "[" + ", ".join(elems) + "]"

def covariance_to_correlation(cov: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if cov is None:
        return None
    if cov.size == 0:
        return None
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        return None
    diag = np.diag(cov).copy()
    diag[diag < 0] = 0.0
    std = np.sqrt(diag)
    denom = np.outer(std, std)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = cov / denom
    corr[~np.isfinite(corr)] = 0.0
    for i in range(corr.shape[0]):
        corr[i, i] = 1.0 if diag[i] > 0 else 0.0
    corr = (corr + corr.T) / 2.0
    return corr

def write_single_txt_for_epoch(outdir: str,
                               model: str,
                               epoch: str,
                               sdft_corr: Optional[np.ndarray],
                               sft_corr: Optional[np.ndarray],
                               diff_corr: Optional[np.ndarray]) -> str:
    os.makedirs(outdir, exist_ok=True)
    base_name = "analysis_corr_safe.txt"
    path = os.path.join(outdir, base_name)
    if os.path.exists(path):
        idx = 1
        while True:
            path = os.path.join(outdir, f"analysis_corr_safe_{idx}.txt")
            if not os.path.exists(path):
                break
            idx += 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model}\nEpoch: {epoch}\n\n")
        f.write("=== sdft AS CORRELATION ===\n")
        f.write(pretty_matrix_text(sdft_corr) + "\n")
        if sdft_corr is not None:
            w, v = compute_eigvals_vecs(sdft_corr)
            f.write("sdft_corr eigenvalues and eigenvectors (eigenvector is listed as a column vector):\n")
            if w.size == 0:
                f.write("  (no eigenvalues)\n\n")
            else:
                for k in range(len(w)):
                    val = w[k]
                    vec = v[:, k]
                    f.write(f"  eig[{k+1}]: {complex_to_str(val)}\n")
                    f.write(f"    vec[{k+1}]: {vector_to_str(vec)}\n")
                f.write("\n")
        else:
            f.write("sdft_corr: MISSING\n\n")
        f.write("=== sft AS CORRELATION ===\n")
        f.write(pretty_matrix_text(sft_corr) + "\n")
        if sft_corr is not None:
            w, v = compute_eigvals_vecs(sft_corr)
            f.write("sft_corr eigenvalues and eigenvectors (eigenvector is listed as a column vector):\n")
            if w.size == 0:
                f.write("  (no eigenvalues)\n\n")
            else:
                for k in range(len(w)):
                    val = w[k]
                    vec = v[:, k]
                    f.write(f"  eig[{k+1}]: {complex_to_str(val)}\n")
                    f.write(f"    vec[{k+1}]: {vector_to_str(vec)}\n")
                f.write("\n")
        else:
            f.write("sft_corr: MISSING\n\n")
        f.write("=== diff_corr = sft_corr - sdft_corr ===\n")
        f.write(pretty_matrix_text(diff_corr) + "\n")
        if diff_corr is not None:
            w, v = compute_eigvals_vecs(diff_corr)
            f.write("diff_corr eigenvalues and eigenvectors (eigenvector is listed as a column vector):\n")
            if w.size == 0:
                f.write("  (no eigenvalues)\n\n")
            else:
                for k in range(len(w)):
                    val = w[k]
                    vec = v[:, k]
                    f.write(f"  eig[{k+1}]: {complex_to_str(val)}\n")
                    f.write(f"    vec[{k+1}]: {vector_to_str(vec)}\n")
                f.write("\n")
        else:
            f.write("diff_corr: MISSING or shape mismatch\n\n")
    return path

def analyze_one_model_epoch(results_root: str, model: str, epoch: str) -> dict:
    sdft = load_matrix_from_dir(results_root, model, epoch, "sdft")
    sft = load_matrix_from_dir(results_root, model, epoch, "sft")
    sdft_corr = covariance_to_correlation(sdft)
    sft_corr = covariance_to_correlation(sft)
    diff_corr = None
    if sdft_corr is not None and sft_corr is not None and sdft_corr.shape == sft_corr.shape:
        diff_corr = sft_corr - sdft_corr
    parent = os.path.join(results_root, model, epoch)
    safe_dir = os.path.join(parent, "analysis_safe")
    txt_path = write_single_txt_for_epoch(
        outdir=safe_dir,
        model=model,
        epoch=epoch,
        sdft_corr=sdft_corr,
        sft_corr=sft_corr,
        diff_corr=diff_corr
    )
    return {
        'model': model,
        'epoch': epoch,
        'txt_path': txt_path,
        'outdir': safe_dir
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.normpath(os.path.join(script_dir, "..", "results"))
    if not os.path.isdir(results_root):
        print(f"results_root not found: {results_root}", file=sys.stderr)
        sys.exit(1)
    models = sorted([d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))])
    for model in models:
        model_dir = os.path.join(results_root, model)
        epochs = sorted([d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))])
        for epoch in epochs:
            info = analyze_one_model_epoch(results_root, model, epoch)
            print(f"ANALYZED: {model} / {epoch} -> {info['txt_path']}")

if __name__ == "__main__":
    main()
