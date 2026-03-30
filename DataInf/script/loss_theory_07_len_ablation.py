#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from loss_theory_utils import (
    classify_tail,
    detect_datainf_root,
    ensure_dir,
    find_sample_csvs,
    fit_linear,
    load_concat_csv,
    log_mgf,
    positive_tail,
    resolve_loss_theory_root,
    survival_grid,
    to_numeric_series,
    write_df_csv_json,
)


def tail_label(values: np.ndarray) -> Dict[str, object]:
    centered = values - float(np.mean(values))
    tail = positive_tail(centered)
    xs, ss = survival_grid(tail, max_points=256)
    if len(xs) < 6:
        return {"tail_label": "insufficient_data", "semilog_r2": float("nan"), "loglog_r2": float("nan"), "weibull_r2": float("nan")}
    semilog = fit_linear(xs, np.log(np.clip(ss, 1e-12, 1.0)))
    loglog = fit_linear(np.log(np.clip(xs, 1e-12, None)), np.log(np.clip(ss, 1e-12, 1.0)))
    weib = fit_linear(
        np.log(np.clip(xs, 1e-12, None)),
        np.log(np.clip(-np.log(np.clip(ss, 1e-12, 1.0)), 1e-12, None)),
    )
    return {
        "tail_label": classify_tail(float(semilog["r2"]), float(loglog["r2"]), float(weib["r2"]), float(weib["slope"])),
        "semilog_r2": float(semilog["r2"]),
        "loglog_r2": float(loglog["r2"]),
        "weibull_r2": float(weib["r2"]),
    }


def local_mgf_mse(values: np.ndarray) -> float:
    x = values - float(np.mean(values))
    lam = np.linspace(0.01, 0.30, 30)
    y = np.asarray([log_mgf(x, float(t)) for t in lam], dtype=np.float64)
    a = float(np.sum((lam**2) * y) / np.sum(lam**4))
    pred = a * lam**2
    return float(np.mean((pred - y) ** 2))


def simple_bernstein_coverage(values: np.ndarray, repeats: int = 200, delta: float = 0.05) -> float:
    n_total = len(values)
    if n_total < 64:
        return float("nan")
    mu_full = float(np.mean(values))
    n = max(32, min(512, n_total // 4))
    scale_b = float(np.quantile(np.abs(values - mu_full), 0.99))
    cover = 0
    for _ in range(repeats):
        idx = np.random.choice(n_total, size=n, replace=False)
        sub = values[idx]
        mu_hat = float(np.mean(sub))
        var_hat = float(np.var(sub, ddof=1)) if len(sub) > 1 else 0.0
        rad = np.sqrt(2.0 * max(var_hat, 0.0) * np.log(3.0 / delta) / n) + 3.0 * scale_b * np.log(3.0 / delta) / n
        if abs(mu_hat - mu_full) <= rad:
            cover += 1
    return float(cover / repeats)


def main() -> None:
    p = argparse.ArgumentParser(description="Length-normalization ablation: Li vs Lbar_i.")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(int(args.seed))
    datainf_root = detect_datainf_root(args.datainf_root)
    loss_theory_root = resolve_loss_theory_root(datainf_root, args.output_root)
    out_dir = ensure_dir(os.path.join(loss_theory_root, "analysis", "len_ablation"))

    sample_df = load_concat_csv(find_sample_csvs(loss_theory_root))
    li = to_numeric_series(sample_df, "Li")
    lbar = to_numeric_series(sample_df, "Lbar_i")
    if len(li) < 100 or len(lbar) < 100:
        raise RuntimeError("insufficient rows for length ablation")

    rows: List[Dict[str, object]] = []
    for name, arr in [("Li_raw_total", li), ("Lbar_i_normalized", lbar)]:
        t = tail_label(arr)
        rows.append(
            {
                "target": name,
                "n": int(len(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "tail_label": t["tail_label"],
                "tail_semilog_r2": t["semilog_r2"],
                "tail_loglog_r2": t["loglog_r2"],
                "tail_weibull_r2": t["weibull_r2"],
                "mgf_quadratic_mse": float(local_mgf_mse(arr)),
                "emp_bernstein_coverage_delta_0p05": float(simple_bernstein_coverage(arr, repeats=200, delta=0.05)),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "len_ablation_summary.csv")
    json_path = os.path.join(out_dir, "len_ablation_summary.json")
    write_df_csv_json(df, csv_path, json_path)
    print(os.path.abspath(csv_path))
    print(os.path.abspath(json_path))


if __name__ == "__main__":
    main()

