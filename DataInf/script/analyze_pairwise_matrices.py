#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_pairwise_matrices_write_txt_with_vecs.py

遍历 ../result 下的 <model>/<epoch> 子目录，
读取 sdft/sft 的 pairwise 矩阵（优先 .npy，再 .csv，再由 sim_*.json 重建），
计算每个矩阵的特征值与特征向量，保存并**在 analysis_log.txt 中以可读格式直接写出每个特征值对应的特征向量**（覆盖旧文件）。
同时保留原来的 npy 文件与可视化 png（若能绘制）。
将此文件放到：
    DataInf/script/

直接运行：
    python analyze_pairwise_matrices_write_txt_with_vecs.py
"""
import os
import sys
import json
from typing import Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_matrix_from_dir(result_dir: str, model: str, epoch: str, method: str) -> Optional[np.ndarray]:
    base = os.path.join(result_dir, model, epoch, method)
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
    # fallback: reconstruct from pairwise_result/*.json
    pair_dir = os.path.join(base, "pairwise_result")
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

def compute_eigvals_vecs(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if mat is None or mat.size == 0:
        return np.array([]), np.array([[]])
    if is_symmetric(mat):
        w, v = np.linalg.eigh(mat)  # ascending
        idx = np.argsort(-w.real)   # descending by real part
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
    # format complex or real number into readable string
    if getattr(z, "imag", 0.0) == 0 or abs(getattr(z, "imag", 0.0)) < 1e-12:
        return f"{float(getattr(z, 'real', z)):.{precision}f}"
    else:
        return f"({z.real:.{precision}f}{z.imag:+.{precision}f}j)"

def vector_to_str(vec, precision=6):
    elems = [complex_to_str(v, precision) for v in np.asarray(vec).reshape(-1)]
    return "[" + ", ".join(elems) + "]"

def plot_three_matrices(sdft_mat, sft_mat, diff_mat, out_png: str, model: str, epoch: str):
    mats = [sdft_mat, sft_mat, diff_mat]
    titles = ["sdft", "sft", "sft - sdft"]
    numeric_mats = [m for m in mats if m is not None]
    vmax = None
    vmin = None
    if numeric_mats:
        all_abs_max = max(np.max(np.abs(m)) for m in numeric_mats)
        if not np.isfinite(all_abs_max) or all_abs_max == 0:
            all_abs_max = 1.0
        vmax = all_abs_max
        vmin = -all_abs_max
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, mat, title in zip(axes, mats, titles):
        ax.set_title(title)
        if mat is None:
            ax.text(0.5, 0.5, "MISSING", ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            im = ax.imshow(mat, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(mat.shape[1]))
            ax.set_yticks(range(mat.shape[0]))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{model} | {epoch}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def analyze_one_model_epoch(result_root: str, model: str, epoch: str):
    sdft = load_matrix_from_dir(result_root, model, epoch, "sdft")
    sft  = load_matrix_from_dir(result_root, model, epoch, "sft")
    diff = None
    if sdft is not None and sft is not None and sdft.shape == sft.shape:
        diff = sft - sdft

    outdir = os.path.join(result_root, model, epoch, "analysis")
    os.makedirs(outdir, exist_ok=True)

    txt_path = os.path.join(outdir, "analysis_log.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model}\nEpoch: {epoch}\n\n")

        # sdft
        f.write("=== sdft ===\n")
        f.write(pretty_matrix_text(sdft) + "\n")
        if sdft is not None:
            mat_file = os.path.join(outdir, "sdft_matrix.npy")
            np.save(mat_file, sdft)
            w, v = compute_eigvals_vecs(sdft)
            eigvecs_file = os.path.join(outdir, "sdft_eigvecs.npy")
            np.save(eigvecs_file, v)
            f.write("sdft matrix saved: " + os.path.basename(mat_file) + "\n")
            f.write("sdft eigvecs saved: " + os.path.basename(eigvecs_file) + "\n")
            f.write("sdft eigenvalues and eigenvectors (eigenvector is listed as a column vector):\n")
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
            f.write("sdft: MISSING\n\n")

        # sft
        f.write("=== sft ===\n")
        f.write(pretty_matrix_text(sft) + "\n")
        if sft is not None:
            mat_file = os.path.join(outdir, "sft_matrix.npy")
            np.save(mat_file, sft)
            w, v = compute_eigvals_vecs(sft)
            eigvecs_file = os.path.join(outdir, "sft_eigvecs.npy")
            np.save(eigvecs_file, v)
            f.write("sft matrix saved: " + os.path.basename(mat_file) + "\n")
            f.write("sft eigvecs saved: " + os.path.basename(eigvecs_file) + "\n")
            f.write("sft eigenvalues and eigenvectors (eigenvector is listed as a column vector):\n")
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
            f.write("sft: MISSING\n\n")

        # diff
        f.write("=== diff (sft - sdft) ===\n")
        f.write(pretty_matrix_text(diff) + "\n")
        if diff is not None:
            mat_file = os.path.join(outdir, "diff_matrix.npy")
            np.save(mat_file, diff)
            w, v = compute_eigvals_vecs(diff)
            eigvecs_file = os.path.join(outdir, "diff_eigvecs.npy")
            np.save(eigvecs_file, v)
            f.write("diff matrix saved: " + os.path.basename(mat_file) + "\n")
            f.write("diff eigvecs saved: " + os.path.basename(eigvecs_file) + "\n")
            f.write("diff eigenvalues and eigenvectors (eigenvector is listed as a column vector):\n")
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
            f.write("diff: MISSING or shape mismatch\n\n")

    # try to generate PNG visualization (do not affect txt writing)
    png_path = os.path.join(outdir, "analysis_matrices.png")
    try:
        plot_three_matrices(sdft, sft, diff, png_path, model, epoch)
    except Exception:
        pass

    return {
        'model': model,
        'epoch': epoch,
        'txt_path': txt_path,
        'outdir': outdir
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_root = os.path.normpath(os.path.join(script_dir, "..", "result"))

    if not os.path.isdir(result_root):
        print(f"result_root not found: {result_root}", file=sys.stderr)
        sys.exit(1)

    models = sorted([d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d))])
    summary = []
    for model in models:
        model_dir = os.path.join(result_root, model)
        epochs = sorted([d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))])
        for epoch in epochs:
            info = analyze_one_model_epoch(result_root, model, epoch)
            summary.append(info)
            print(f"ANALYZED: {model} / {epoch} -> {info['txt_path']}")

    # top-level summary file (txt) - overwrite if exists
    top_summary = os.path.join(result_root, "analysis_summary.txt")
    with open(top_summary, "w", encoding="utf-8") as f:
        for s in summary:
            f.write(f"{s['model']}\t{s['epoch']}\t{s['txt_path']}\n")
    print("Done. summary saved to:", top_summary)

if __name__ == "__main__":
    main()
