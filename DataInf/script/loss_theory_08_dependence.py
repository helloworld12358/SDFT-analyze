#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from loss_theory_utils import (
    detect_datainf_root,
    ensure_dir,
    find_sample_csvs,
    find_seq_probe_csvs,
    load_concat_csv,
    resolve_loss_theory_root,
    write_df_csv_json,
)


def acf_at_lag(x: np.ndarray, lag: int) -> float:
    n = len(x)
    if n <= lag + 1:
        return float("nan")
    a = x[:-lag]
    b = x[lag:]
    va = float(np.var(a))
    vb = float(np.var(b))
    if va <= 0 or vb <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def block_means_variance(x: np.ndarray, block_size: int) -> float:
    n = len(x)
    if n < block_size or block_size <= 0:
        return float("nan")
    n_blocks = n // block_size
    if n_blocks < 2:
        return float("nan")
    y = x[: n_blocks * block_size].reshape(n_blocks, block_size).mean(axis=1)
    return float(np.var(y, ddof=1))


def bootstrap_mean_ci_width(x: np.ndarray, repeats: int = 200) -> float:
    n = len(x)
    if n == 0:
        return float("nan")
    means = []
    for _ in range(repeats):
        idx = np.random.choice(n, size=n, replace=True)
        means.append(float(np.mean(x[idx])))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(hi - lo)


def block_bootstrap_mean_ci_width(x: np.ndarray, block_size: int, repeats: int = 200) -> float:
    n = len(x)
    if n == 0:
        return float("nan")
    block_size = max(1, int(block_size))
    n_blocks = int(np.ceil(n / block_size))
    blocks = [x[i * block_size : min((i + 1) * block_size, n)] for i in range(n_blocks)]
    means = []
    for _ in range(repeats):
        picked = [blocks[np.random.randint(0, n_blocks)] for _ in range(n_blocks)]
        arr = np.concatenate(picked, axis=0)[:n]
        means.append(float(np.mean(arr)))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(hi - lo)


def main() -> None:
    p = argparse.ArgumentParser(description="Dependence diagnostics with token probes and block bootstrap.")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default="")
    p.add_argument("--max_lag", type=int, default=10)
    p.add_argument("--bootstrap_repeats", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(int(args.seed))
    datainf_root = detect_datainf_root(args.datainf_root)
    loss_theory_root = resolve_loss_theory_root(datainf_root, args.output_root)
    out_dir = ensure_dir(os.path.join(loss_theory_root, "analysis", "dependence"))

    seq_df = load_concat_csv(find_seq_probe_csvs(loss_theory_root))
    sample_df = load_concat_csv(find_sample_csvs(loss_theory_root))
    if seq_df.empty or sample_df.empty:
        raise RuntimeError("sequence probe rows or sample rows are missing")

    seq_df["u_i_t"] = pd.to_numeric(seq_df["u_i_t"], errors="coerce")
    seq_df["token_pos"] = pd.to_numeric(seq_df["token_pos"], errors="coerce")
    seq_df["sample_idx"] = pd.to_numeric(seq_df["sample_idx"], errors="coerce")
    seq_df = seq_df.dropna(subset=["u_i_t", "token_pos", "sample_idx"]).copy()

    acf_rows: List[Dict[str, object]] = []
    block_var_rows: List[Dict[str, object]] = []
    lag_list = list(range(1, max(2, int(args.max_lag)) + 1))
    block_sizes = [2, 4, 8, 16, 32]

    by_sample = seq_df.sort_values(["combo_key", "sample_idx", "token_pos"]).groupby(["combo_key", "sample_idx"], sort=False)
    for (combo_key, sample_idx), sub in by_sample:
        arr = pd.to_numeric(sub["u_i_t"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if len(arr) < 6:
            continue
        for lag in lag_list:
            acf_rows.append(
                {
                    "combo_key": combo_key,
                    "sample_idx": int(sample_idx),
                    "lag": int(lag),
                    "acf": acf_at_lag(arr, lag),
                    "n_tokens": int(len(arr)),
                }
            )
        for b in block_sizes:
            block_var_rows.append(
                {
                    "combo_key": combo_key,
                    "sample_idx": int(sample_idx),
                    "block_size": int(b),
                    "block_mean_variance": block_means_variance(arr, b),
                    "n_tokens": int(len(arr)),
                }
            )

    acf_df = pd.DataFrame(acf_rows)
    bv_df = pd.DataFrame(block_var_rows)
    acf_csv = os.path.join(out_dir, "token_acf.csv")
    acf_json = os.path.join(out_dir, "token_acf.json")
    bv_csv = os.path.join(out_dir, "token_block_variance.csv")
    bv_json = os.path.join(out_dir, "token_block_variance.json")
    write_df_csv_json(acf_df, acf_csv, acf_json)
    write_df_csv_json(bv_df, bv_csv, bv_json)

    sample_df["Lbar_i"] = pd.to_numeric(sample_df["Lbar_i"], errors="coerce")
    sample_df["sample_idx"] = pd.to_numeric(sample_df["sample_idx"], errors="coerce")
    sample_df = sample_df.dropna(subset=["Lbar_i", "sample_idx"]).copy()
    sample_df["source_group"] = sample_df.get("source_id", "").astype(str)
    sample_df.loc[sample_df["source_group"].str.strip() == "", "source_group"] = "idx_block"

    dep_rows: List[Dict[str, object]] = []
    for combo_key, sub in sample_df.groupby("combo_key", sort=False):
        arr = pd.to_numeric(sub["Lbar_i"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if len(arr) < 30:
            continue
        naive_w = bootstrap_mean_ci_width(arr, repeats=int(args.bootstrap_repeats))

        sub_ord = sub.sort_values("sample_idx")
        if (sub_ord["source_group"] != "idx_block").any():
            block_size = int(max(2, np.median(sub_ord.groupby("source_group").size().to_numpy(dtype=np.float64))))
        else:
            block_size = int(max(2, round(np.sqrt(len(arr)))))

        ordered_w = block_bootstrap_mean_ci_width(
            pd.to_numeric(sub_ord["Lbar_i"], errors="coerce").dropna().to_numpy(dtype=np.float64),
            block_size=block_size,
            repeats=int(args.bootstrap_repeats),
        )
        shuffled = sub_ord.sample(frac=1.0, replace=False, random_state=int(args.seed))
        shuffled_w = block_bootstrap_mean_ci_width(
            pd.to_numeric(shuffled["Lbar_i"], errors="coerce").dropna().to_numpy(dtype=np.float64),
            block_size=block_size,
            repeats=int(args.bootstrap_repeats),
        )

        dep_rows.append(
            {
                "combo_key": combo_key,
                "n": int(len(arr)),
                "block_size_used": int(block_size),
                "naive_bootstrap_ci_width": float(naive_w),
                "ordered_block_bootstrap_ci_width": float(ordered_w),
                "shuffled_block_bootstrap_ci_width": float(shuffled_w),
                "ordered_vs_naive_ratio": float(ordered_w / naive_w) if naive_w > 0 else float("nan"),
                "ordered_vs_shuffled_ratio": float(ordered_w / shuffled_w) if shuffled_w > 0 else float("nan"),
            }
        )

    dep_df = pd.DataFrame(dep_rows)
    dep_csv = os.path.join(out_dir, "sample_block_bootstrap_compare.csv")
    dep_json = os.path.join(out_dir, "sample_block_bootstrap_compare.json")
    write_df_csv_json(dep_df, dep_csv, dep_json)

    print(os.path.abspath(acf_csv))
    print(os.path.abspath(bv_csv))
    print(os.path.abspath(dep_csv))


if __name__ == "__main__":
    main()

