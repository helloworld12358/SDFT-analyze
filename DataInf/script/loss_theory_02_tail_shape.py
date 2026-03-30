#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loss_theory_utils import (
    classify_tail,
    detect_datainf_root,
    ensure_dir,
    find_sample_csvs,
    find_token_csvs,
    fit_linear,
    load_concat_csv,
    positive_tail,
    resolve_loss_theory_root,
    survival_grid,
    tail_slice_by_quantile,
    to_numeric_series,
    write_df_csv_json,
)


def analyze_one(values: np.ndarray, name: str, q_lo: float, q_hi: float, out_dir: str) -> Dict[str, object]:
    centered = values - float(np.mean(values))
    tail = positive_tail(centered)
    xg, sg = survival_grid(tail, max_points=512)

    sub = tail_slice_by_quantile(tail, q_lo=q_lo, q_hi=q_hi)
    xs, ss = survival_grid(sub, max_points=256)

    semilog = fit_linear(xs, np.log(np.clip(ss, 1e-12, 1.0))) if len(xs) > 0 else {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": 0}
    loglog = fit_linear(np.log(np.clip(xs, 1e-12, None)), np.log(np.clip(ss, 1e-12, 1.0))) if len(xs) > 0 else {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": 0}
    weib = fit_linear(
        np.log(np.clip(xs, 1e-12, None)),
        np.log(np.clip(-np.log(np.clip(ss, 1e-12, 1.0)), 1e-12, None)),
    ) if len(xs) > 0 else {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": 0}

    label = classify_tail(
        semilog_r2=float(semilog["r2"]),
        loglog_r2=float(loglog["r2"]),
        weibull_r2=float(weib["r2"]),
        weibull_slope=float(weib["slope"]),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(xg, np.log(np.clip(sg, 1e-12, 1.0)), marker=".", linestyle="-")
    axes[0].set_title(f"{name}: log S(x) vs x")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("log S(x)")

    axes[1].plot(np.log(np.clip(xg, 1e-12, None)), np.log(np.clip(sg, 1e-12, 1.0)), marker=".", linestyle="-")
    axes[1].set_title(f"{name}: log S(x) vs log x")
    axes[1].set_xlabel("log x")
    axes[1].set_ylabel("log S(x)")

    axes[2].plot(
        np.log(np.clip(xg, 1e-12, None)),
        np.log(np.clip(-np.log(np.clip(sg, 1e-12, 1.0)), 1e-12, None)),
        marker=".",
        linestyle="-",
    )
    axes[2].set_title(f"{name}: log(-log S(x)) vs log x")
    axes[2].set_xlabel("log x")
    axes[2].set_ylabel("log(-log S(x))")
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"tail_shape_{name}.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    return {
        "name": name,
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "tail_n": int(len(tail)),
        "fit_quantile_lo": float(q_lo),
        "fit_quantile_hi": float(q_hi),
        "semilog_slope": float(semilog["slope"]),
        "semilog_r2": float(semilog["r2"]),
        "loglog_slope": float(loglog["slope"]),
        "loglog_r2": float(loglog["r2"]),
        "weibull_slope": float(weib["slope"]),
        "weibull_r2": float(weib["r2"]),
        "empirical_label": label,
        "plot_path": os.path.abspath(plot_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Tail-shape diagnostics for normalized sample loss and token subsamples.")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default="")
    p.add_argument("--fit_q_lo", type=float, default=0.90)
    p.add_argument("--fit_q_hi", type=float, default=0.999)
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    loss_theory_root = resolve_loss_theory_root(datainf_root, args.output_root)
    out_dir = ensure_dir(os.path.join(loss_theory_root, "analysis", "tail_shape"))

    sample_df = load_concat_csv(find_sample_csvs(loss_theory_root))
    token_df = load_concat_csv(find_token_csvs(loss_theory_root))

    rows: List[Dict[str, object]] = []
    if not sample_df.empty:
        vals = to_numeric_series(sample_df, "Lbar_i")
        if len(vals) > 20:
            rows.append(analyze_one(vals, "sample_Lbar_i", args.fit_q_lo, args.fit_q_hi, out_dir))
    if not token_df.empty:
        vals = to_numeric_series(token_df, "u_i_t")
        if len(vals) > 20:
            rows.append(analyze_one(vals, "token_u_i_t", args.fit_q_lo, args.fit_q_hi, out_dir))

    out_df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "tail_shape_summary.csv")
    json_path = os.path.join(out_dir, "tail_shape_summary.json")
    write_df_csv_json(out_df, csv_path, json_path)
    print(os.path.abspath(csv_path))
    print(os.path.abspath(json_path))


if __name__ == "__main__":
    main()

