#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loss_theory_utils import (
    detect_datainf_root,
    ensure_dir,
    find_sample_csvs,
    load_concat_csv,
    resolve_loss_theory_root,
    to_numeric_series,
    write_df_csv_json,
)


def sample_without_replacement_indices(n_total: int, n_pick: int) -> np.ndarray:
    return np.random.choice(n_total, size=n_pick, replace=False)


def empirical_bernstein_radius(var_hat: float, n: int, delta: float, scale_b: float) -> float:
    logt = np.log(3.0 / float(delta))
    return float(np.sqrt(2.0 * max(var_hat, 0.0) * logt / n) + 3.0 * scale_b * logt / n)


def build_n_grid(n_total: int) -> List[int]:
    cands = [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
    out = sorted({x for x in cands if x < n_total})
    if not out:
        out = [max(8, n_total // 4)]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Empirical Bernstein coverage check on normalized sample loss.")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default="")
    p.add_argument("--repeats", type=int, default=300)
    p.add_argument("--deltas", type=str, default="0.1,0.05,0.01")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(int(args.seed))
    datainf_root = detect_datainf_root(args.datainf_root)
    loss_theory_root = resolve_loss_theory_root(datainf_root, args.output_root)
    out_dir = ensure_dir(os.path.join(loss_theory_root, "analysis", "emp_bernstein"))

    sample_df = load_concat_csv(find_sample_csvs(loss_theory_root))
    vals = to_numeric_series(sample_df, "Lbar_i")
    if len(vals) < 100:
        raise RuntimeError("not enough sample rows for empirical Bernstein coverage")

    mu_full = float(np.mean(vals))
    n_total = int(len(vals))
    n_grid = build_n_grid(n_total)
    deltas = [float(x.strip()) for x in args.deltas.split(",") if x.strip()]
    scale_b = float(np.quantile(np.abs(vals - mu_full), 0.99))

    rows: List[Dict[str, object]] = []
    for n in n_grid:
        for delta in deltas:
            cover = 0
            abs_dev_list: List[float] = []
            rad_list: List[float] = []
            for _ in range(int(args.repeats)):
                idx = sample_without_replacement_indices(n_total, n)
                sub = vals[idx]
                mu_hat = float(np.mean(sub))
                var_hat = float(np.var(sub, ddof=1)) if len(sub) > 1 else 0.0
                rad = empirical_bernstein_radius(var_hat, n=n, delta=delta, scale_b=scale_b)
                abs_dev = abs(mu_hat - mu_full)
                if abs_dev <= rad:
                    cover += 1
                abs_dev_list.append(abs_dev)
                rad_list.append(rad)

            rows.append(
                {
                    "n": int(n),
                    "delta": float(delta),
                    "repeats": int(args.repeats),
                    "coverage_rate": float(cover / float(args.repeats)),
                    "mean_abs_dev": float(np.mean(abs_dev_list)),
                    "p95_abs_dev": float(np.quantile(abs_dev_list, 0.95)),
                    "mean_radius": float(np.mean(rad_list)),
                    "p95_radius": float(np.quantile(rad_list, 0.95)),
                    "mu_full": mu_full,
                    "scale_b_q99_abs_centered": scale_b,
                }
            )

    df = pd.DataFrame(rows).sort_values(["delta", "n"]).reset_index(drop=True)
    csv_path = os.path.join(out_dir, "emp_bernstein_coverage.csv")
    json_path = os.path.join(out_dir, "emp_bernstein_coverage.json")
    write_df_csv_json(df, csv_path, json_path)

    fig = plt.figure(figsize=(7, 5))
    for delta in sorted(df["delta"].unique().tolist()):
        dsub = df[df["delta"] == delta].sort_values("n")
        plt.plot(dsub["n"], dsub["coverage_rate"], marker="o", label=f"delta={delta}")
    plt.xscale("log")
    plt.ylim(0.0, 1.05)
    plt.xlabel("n (log scale)")
    plt.ylabel("coverage rate")
    plt.title("Empirical Bernstein coverage on Lbar_i")
    plt.legend(loc="best")
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "emp_bernstein_coverage.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(os.path.abspath(csv_path))
    print(os.path.abspath(json_path))
    print(os.path.abspath(plot_path))


if __name__ == "__main__":
    main()

