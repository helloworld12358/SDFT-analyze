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
    catoni_m_estimator,
    detect_datainf_root,
    ensure_dir,
    find_sample_csvs,
    load_concat_csv,
    median_of_means,
    resolve_loss_theory_root,
    to_numeric_series,
    trimmed_mean,
    write_df_csv_json,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Robust mean comparison under repeated subsampling.")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default="")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--repeats", type=int, default=400)
    p.add_argument("--trim_alpha", type=float, default=0.1)
    p.add_argument("--mom_blocks", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(int(args.seed))
    datainf_root = detect_datainf_root(args.datainf_root)
    loss_theory_root = resolve_loss_theory_root(datainf_root, args.output_root)
    out_dir = ensure_dir(os.path.join(loss_theory_root, "analysis", "robust_mean"))

    sample_df = load_concat_csv(find_sample_csvs(loss_theory_root))
    vals = to_numeric_series(sample_df, "Lbar_i")
    if len(vals) < max(50, int(args.n) + 1):
        raise RuntimeError("not enough sample rows for robust-mean diagnostics")
    mu_full = float(np.mean(vals))
    n = int(args.n)

    rows: List[Dict[str, object]] = []
    est_name_list = ["mean", "trimmed_mean", "median_of_means", "catoni"]
    deviations: Dict[str, List[float]] = {k: [] for k in est_name_list}
    inj_sensitivity: Dict[str, List[float]] = {k: [] for k in est_name_list}
    rm_sensitivity: Dict[str, List[float]] = {k: [] for k in est_name_list}

    top_ext = float(np.quantile(vals, 0.999))
    for rep in range(int(args.repeats)):
        idx = np.random.choice(len(vals), size=n, replace=False)
        sub = vals[idx]
        est = {
            "mean": float(np.mean(sub)),
            "trimmed_mean": float(trimmed_mean(sub, alpha=float(args.trim_alpha))),
            "median_of_means": float(median_of_means(sub, n_blocks=int(args.mom_blocks))),
            "catoni": float(catoni_m_estimator(sub)),
        }

        sub_inj = np.concatenate([sub[:-1], np.asarray([top_ext], dtype=np.float64)])
        sub_rm = np.sort(sub)[:-1] if len(sub) > 1 else sub
        est_inj = {
            "mean": float(np.mean(sub_inj)),
            "trimmed_mean": float(trimmed_mean(sub_inj, alpha=float(args.trim_alpha))),
            "median_of_means": float(median_of_means(sub_inj, n_blocks=int(args.mom_blocks))),
            "catoni": float(catoni_m_estimator(sub_inj)),
        }
        est_rm = {
            "mean": float(np.mean(sub_rm)),
            "trimmed_mean": float(trimmed_mean(sub_rm, alpha=float(args.trim_alpha))),
            "median_of_means": float(median_of_means(sub_rm, n_blocks=int(args.mom_blocks))),
            "catoni": float(catoni_m_estimator(sub_rm)),
        }

        for name in est_name_list:
            dev = abs(est[name] - mu_full)
            deviations[name].append(dev)
            inj_sensitivity[name].append(abs(est_inj[name] - est[name]))
            rm_sensitivity[name].append(abs(est_rm[name] - est[name]))
            rows.append(
                {
                    "repeat_id": rep,
                    "estimator": name,
                    "estimate": est[name],
                    "abs_dev_vs_mu_full": dev,
                    "inj_shift_abs": abs(est_inj[name] - est[name]),
                    "remove_shift_abs": abs(est_rm[name] - est[name]),
                }
            )

    rep_df = pd.DataFrame(rows)
    rep_csv = os.path.join(out_dir, "robust_mean_repeats.csv")
    rep_json = os.path.join(out_dir, "robust_mean_repeats.json")
    write_df_csv_json(rep_df, rep_csv, rep_json)

    summary_rows: List[Dict[str, object]] = []
    for name in est_name_list:
        d = np.asarray(deviations[name], dtype=np.float64)
        inj = np.asarray(inj_sensitivity[name], dtype=np.float64)
        rm = np.asarray(rm_sensitivity[name], dtype=np.float64)
        summary_rows.append(
            {
                "estimator": name,
                "mean_abs_dev": float(np.mean(d)),
                "p95_abs_dev": float(np.quantile(d, 0.95)),
                "max_abs_dev": float(np.max(d)),
                "mean_inj_shift": float(np.mean(inj)),
                "mean_remove_shift": float(np.mean(rm)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(out_dir, "robust_mean_summary.csv")
    sum_json = os.path.join(out_dir, "robust_mean_summary.json")
    write_df_csv_json(summary_df, sum_csv, sum_json)

    fig = plt.figure(figsize=(8, 5))
    box_data = [np.asarray(deviations[name], dtype=np.float64) for name in est_name_list]
    plt.boxplot(box_data, labels=est_name_list, showfliers=False)
    plt.ylabel("|estimate - mu_full|")
    plt.title("Estimator deviation comparison")
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "robust_mean_deviation_boxplot.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(os.path.abspath(rep_csv))
    print(os.path.abspath(sum_csv))
    print(os.path.abspath(plot_path))


if __name__ == "__main__":
    main()

