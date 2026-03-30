#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import pandas as pd

from loss_theory_utils import (
    detect_datainf_root,
    ensure_dir,
    find_sample_csvs,
    load_concat_csv,
    resolve_loss_theory_root,
    write_df_csv_json,
)


def safe_quantile(series: pd.Series, q: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return float("nan")
    return float(s.quantile(q))


def summarize_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for g, sub in df.groupby(group_col):
        y = pd.to_numeric(sub["Lbar_i"], errors="coerce").dropna()
        if len(y) == 0:
            continue
        rows.append(
            {
                "group_col": group_col,
                "group_value": str(g),
                "n": int(len(y)),
                "mean": float(y.mean()),
                "var": float(y.var(ddof=1)) if len(y) > 1 else 0.0,
                "q90": float(y.quantile(0.90)),
                "q95": float(y.quantile(0.95)),
                "q99": float(y.quantile(0.99)),
                "emp_bernstein_cov_delta_0p05": estimate_group_coverage(y.to_numpy(dtype=float), delta=0.05, repeats=150),
            }
        )
    return pd.DataFrame(rows)


def estimate_group_coverage(values, delta: float = 0.05, repeats: int = 150) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 40:
        return float("nan")
    mu_full = float(np.mean(arr))
    n = max(20, min(256, len(arr) // 2))
    scale_b = float(np.quantile(np.abs(arr - mu_full), 0.99))
    ok = 0
    for _ in range(repeats):
        idx = np.random.choice(len(arr), size=n, replace=False)
        sub = arr[idx]
        mu_hat = float(np.mean(sub))
        var_hat = float(np.var(sub, ddof=1)) if len(sub) > 1 else 0.0
        rad = (2.0 * max(var_hat, 0.0) * np.log(3.0 / delta) / n) ** 0.5 + 3.0 * scale_b * np.log(3.0 / delta) / n
        if abs(mu_hat - mu_full) <= rad:
            ok += 1
    return float(ok / repeats)


def main() -> None:
    p = argparse.ArgumentParser(description="Conditional heteroscedasticity and conditional tail diagnostics.")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default="")
    p.add_argument("--length_bins", type=int, default=4)
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    loss_theory_root = resolve_loss_theory_root(datainf_root, args.output_root)
    out_dir = ensure_dir(os.path.join(loss_theory_root, "analysis", "conditional"))

    sample_df = load_concat_csv(find_sample_csvs(loss_theory_root))
    if sample_df.empty:
        raise RuntimeError("no sample rows found")

    sample_df["Lbar_i"] = pd.to_numeric(sample_df["Lbar_i"], errors="coerce")
    sample_df["Ti"] = pd.to_numeric(sample_df["Ti"], errors="coerce")
    sample_df = sample_df.dropna(subset=["Lbar_i", "Ti"]).copy()
    if sample_df.empty:
        raise RuntimeError("no valid rows after numeric cleaning")

    bin_count = max(2, int(args.length_bins))
    sample_df["length_bin"] = pd.qcut(sample_df["Ti"], q=bin_count, duplicates="drop")
    sample_df["domain_label"] = sample_df.get("domain_label", sample_df.get("task", "unknown")).astype(str)
    sample_df["contains_code"] = pd.to_numeric(sample_df.get("contains_code", 0), errors="coerce").fillna(0).astype(int)
    sample_df["contains_math"] = pd.to_numeric(sample_df.get("contains_math", 0), errors="coerce").fillna(0).astype(int)
    sample_df["difficulty_bin"] = (
        sample_df["contains_code"].astype(str) + "_code_" + sample_df["contains_math"].astype(str) + "_math"
    )

    pooled = pd.DataFrame(
        [
            {
                "group_col": "pooled",
                "group_value": "all",
                "n": int(len(sample_df)),
                "mean": float(sample_df["Lbar_i"].mean()),
                "var": float(sample_df["Lbar_i"].var(ddof=1)),
                "q90": safe_quantile(sample_df["Lbar_i"], 0.90),
                "q95": safe_quantile(sample_df["Lbar_i"], 0.95),
                "q99": safe_quantile(sample_df["Lbar_i"], 0.99),
                "emp_bernstein_cov_delta_0p05": estimate_group_coverage(sample_df["Lbar_i"].to_numpy(dtype=float), delta=0.05, repeats=200),
            }
        ]
    )
    by_len = summarize_group(sample_df, "length_bin")
    by_domain = summarize_group(sample_df, "domain_label")
    by_diff = summarize_group(sample_df, "difficulty_bin")

    summary_df = pd.concat([pooled, by_len, by_domain, by_diff], ignore_index=True)
    csv_path = os.path.join(out_dir, "conditional_summary.csv")
    json_path = os.path.join(out_dir, "conditional_summary.json")
    write_df_csv_json(summary_df, csv_path, json_path)

    # Simple pooled-vs-conditional gap indicator
    pooled_var = float(pooled.iloc[0]["var"])
    by_domain_mean_var = float(by_domain["var"].mean()) if not by_domain.empty else float("nan")
    by_len_mean_var = float(by_len["var"].mean()) if not by_len.empty else float("nan")
    overview = {
        "pooled_var": pooled_var,
        "mean_var_by_domain": by_domain_mean_var,
        "mean_var_by_length_bin": by_len_mean_var,
        "domain_variance_ratio": by_domain_mean_var / pooled_var if pooled_var > 0 else float("nan"),
        "length_variance_ratio": by_len_mean_var / pooled_var if pooled_var > 0 else float("nan"),
    }
    overview_json = os.path.join(out_dir, "conditional_overview.json")
    with open(overview_json, "w", encoding="utf-8") as f:
        import json

        json.dump(overview, f, ensure_ascii=False, indent=2)

    print(os.path.abspath(csv_path))
    print(os.path.abspath(json_path))
    print(os.path.abspath(overview_json))


if __name__ == "__main__":
    main()
