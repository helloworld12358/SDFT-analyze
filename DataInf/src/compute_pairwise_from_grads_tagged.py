#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_pairwise_from_grads_tagged.py

基于预先保存的平均 LoRA 梯度向量 (.pt)，为给定（model, epoch, method）构造 pairwise 内积矩阵，
并在 results/<model>/<epoch>/<method>/ 下保存：
  - pairwise_matrix_<model>_<epoch>_<method>.npy
  - pairwise_matrix_<model>_<epoch>_<method>.csv
  - pairwise_matrix_<model>_<epoch>_<method>.txt    <-- **可读文本**（矩阵值 + 特征值 + 特征向量）
  - eigenvalues_....npy, eigenvectors_....npy

默认使用内积（不 normalize）。可通过 --normalize 开启余弦相似度（此处默认不启用）。
"""
import os
import sys
import argparse
import json
from glob import glob
import torch
import numpy as np

def _make_deterministic_eigvecs(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v
    v = v.copy()
    n, m = v.shape
    for k in range(m):
        vk = v[:, k]
        norm = np.linalg.norm(vk)
        if not np.isfinite(norm) or norm == 0:
            continue
        vk = vk / norm
        idx = int(np.argmax(np.abs(vk)))
        val = vk[idx]
        if abs(val) < 1e-16:
            v[:, k] = vk
            continue
        phase = val / abs(val)
        vk = vk / phase
        if np.isrealobj(vk) and vk[idx] < 0:
            vk = -vk
        v[:, k] = vk
    return v

def complex_to_str(z, prec=8):
    if abs(getattr(z, "imag", 0.0)) < 1e-12:
        return f"{float(getattr(z, 'real', z)):.{prec}g}"
    else:
        return f"({z.real:.{prec}g}{z.imag:+.{prec}g}j)"

def vector_to_str(vec, prec=8):
    return "[" + ", ".join(complex_to_str(x, prec) for x in vec.reshape(-1)) + "]"

def load_grad_vector(pt_path):
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(pt_path)
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().view(-1).numpy()
    # handle common saved formats
    if isinstance(obj, (list, tuple, np.ndarray)):
        return np.asarray(obj).reshape(-1)
    if isinstance(obj, dict):
        # try to flatten values that are tensors
        parts = []
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                parts.append(v.detach().cpu().view(-1).numpy())
        if parts:
            return np.concatenate(parts, axis=0)
        # otherwise try to coerce dict to array (unlikely)
    raise ValueError(f"Unsupported grad file format: {pt_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datainf_root", type=str, default=None,
                   help="DataInf 根目录（默认：脚本上级目录）")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--epoch", type=str, required=True, choices=["epoch_0","epoch_1","epoch_5"])
    p.add_argument("--method", type=str, required=True, choices=["sdft","sft"])
    p.add_argument("--dataset_names", type=str, default=None,
                   help="逗号分隔的数据集名称列表（默认使用 alpaca_eval,gsm8k,humaneval,multiarith,openfunction）")
    p.add_argument("--normalize", action="store_true", help="是否使用余弦相似度（默认关闭）")
    p.add_argument("--dtype64", action="store_true", help="使用 float64 计算（默认 True 行为）")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.datainf_root:
        DATAINF_ROOT = os.path.abspath(args.datainf_root)
    else:
        DATAINF_ROOT = os.path.normpath(os.path.join(script_dir, ".."))

    if args.dataset_names:
        DATASET_NAMES = [s.strip() for s in args.dataset_names.split(",") if s.strip()]
    else:
        DATASET_NAMES = ["alpaca_eval","gsm8k","humaneval","multiarith","openfunction"]

    MODEL = args.model
    EPOCH = args.epoch
    METHOD = args.method

    GRADS_BASE_DIR = os.path.join(DATAINF_ROOT, "output_grads", EPOCH, METHOD, MODEL)
    if args.verbose:
        print("DATAINF_ROOT:", DATAINF_ROOT)
        print("GRADS_BASE_DIR:", GRADS_BASE_DIR)

    # verify all grad files exist
    grad_paths = []
    for name in DATASET_NAMES:
        pth = os.path.join(GRADS_BASE_DIR, f"{name}.pt")
        if not os.path.isfile(pth):
            raise FileNotFoundError(f"Missing grad vector: {pth}")
        grad_paths.append(pth)

    # load vectors
    vecs = [load_grad_vector(p) for p in grad_paths]
    dims = [v.size for v in vecs]
    if len(set(dims)) != 1:
        raise ValueError(f"Inconsistent grad vector dims: {dims}")
    D = dims[0]
    n = len(vecs)

    dtype = np.float64 if args.dtype64 else np.float32
    V = np.stack([v.astype(dtype) for v in vecs], axis=1)  # shape (D, n)

    if args.normalize:
        norms = np.linalg.norm(V, axis=0)
        norms[norms==0] = 1.0
        Vn = V / norms[None, :]
        M = (Vn.T @ Vn).astype(dtype)
    else:
        M = (V.T @ V).astype(dtype)

    # results dir
    RESULTS_DIR = os.path.join(DATAINF_ROOT, "results", MODEL, EPOCH, METHOD)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    base_tag = f"pairwise_matrix_{MODEL}_{EPOCH}_{METHOD}"
    mat_npy = os.path.join(RESULTS_DIR, base_tag + ".npy")
    mat_csv = os.path.join(RESULTS_DIR, base_tag + ".csv")
    mat_txt = os.path.join(RESULTS_DIR, base_tag + ".txt")
    eigvals_npy = os.path.join(RESULTS_DIR, f"eigenvalues_{MODEL}_{EPOCH}_{METHOD}.npy")
    eigvecs_npy = os.path.join(RESULTS_DIR, f"eigenvectors_{MODEL}_{EPOCH}_{METHOD}.npy")

    np.save(mat_npy, M)
    np.savetxt(mat_csv, M, delimiter=",", fmt="%.18e")

    # eigen decomposition
    try:
        w, v = np.linalg.eigh(M)
        idx = np.argsort(-w.real)
        w = w[idx]
        v = v[:, idx]
    except Exception:
        w, v = np.linalg.eig(M)
        idx = np.argsort(-w.real)
        w = w[idx]
        v = v[:, idx]

    # deterministic normalization
    v = _make_deterministic_eigvecs(v)

    np.save(eigvals_npy, w)
    np.save(eigvecs_npy, v)

    # write readable txt (matrix + eigenvalues + eigenvectors)
    with open(mat_txt, "w", encoding="utf-8") as f:
        f.write(f"Model: {MODEL}\nEpoch: {EPOCH}\nMethod: {METHOD}\nDatasets: {DATASET_NAMES}\n\n")
        f.write("Pairwise matrix (rows/cols order = datasets order above):\n")
        with np.printoptions(precision=8, suppress=True):
            for row in M:
                f.write("  " + ", ".join(f"{float(x):.8e}" for x in row) + "\n")
        f.write("\nEigenvalues (descending by real part):\n")
        for val in w:
            if abs(getattr(val, "imag", 0.0)) < 1e-12:
                f.write(f"  {float(getattr(val,'real',val)):.18e}\n")
            else:
                f.write(f"  {val}\n")
        f.write("\nEigenvectors (each vector listed as column; same ordering as eigenvalues above):\n")
        for k in range(v.shape[1]):
            vec = v[:, k]
            f.write(f"eig[{k+1}] = {complex_to_str(w[k], prec=12)}\n")
            f.write("  vec: [" + ", ".join(complex_to_str(x, prec=12) for x in vec.reshape(-1)) + "]\n\n")

    # produce simple JSON metadata
    meta = {
        "model": MODEL,
        "epoch": EPOCH,
        "method": METHOD,
        "datasets": DATASET_NAMES,
        "matrix_npy": os.path.abspath(mat_npy),
        "matrix_csv": os.path.abspath(mat_csv),
        "matrix_txt": os.path.abspath(mat_txt),
        "eigvals_npy": os.path.abspath(eigvals_npy),
        "eigvecs_npy": os.path.abspath(eigvecs_npy),
        "normalize": bool(args.normalize),
        "dtype": str(dtype)
    }
    with open(os.path.join(RESULTS_DIR, f"summary_{MODEL}_{EPOCH}_{METHOD}.json"), "w", encoding="utf-8") as jf:
        json.dump(meta, jf, ensure_ascii=False, indent=2)

    if args.verbose:
        print("Saved matrix npy:", mat_npy)
        print("Saved matrix csv:", mat_csv)
        print("Saved summary json:", os.path.join(RESULTS_DIR, f"summary_{MODEL}_{EPOCH}_{METHOD}.json"))
    print(os.path.abspath(mat_txt))

if __name__ == "__main__":
    main()
