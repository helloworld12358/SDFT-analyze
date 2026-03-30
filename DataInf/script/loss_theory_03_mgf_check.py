#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loss_theory_utils import (
    detect_datainf_root,
    ensure_dir,
    find_sample_csvs,
    load_concat_csv,
    log_mgf,
    resolve_loss_theory_root,
    to_numeric_series,
    write_df_csv_json,
)


def fit_quadratic_through_origin(lam: np.ndarray, mgf_vals: np.ndarray) -> Dict[str, float]:
    denom = float(np.sum(lam**4))
    if denom <= 0:
        return {"a": float("nan"), "mse": float("nan")}
    a = float(np.sum((lam**2) * mgf_vals) / denom)
    pred = a * (lam**2)
    mse = float(np.mean((pred - mgf_vals) ** 2))
    return {"a": a, "mse": mse}


def subgamma_curve(lam: np.ndarray, v: float, c: float) -> np.ndarray:
    den = 2.0 * (1.0 - c * lam)
    return (v * lam**2) / np.clip(den, 1e-12, None)


def fit_subgamma_grid(lam: np.ndarray, mgf_vals: np.ndarray) -> Dict[str, float]:
    if len(lam) < 3:
        return {"v": float("nan"), "c": float("nan"), "mse": float("nan")}
    c_max = 0.95 / float(np.max(lam))
    best = {"v": float("nan"), "c": float("nan"), "mse": float("inf")}
    c_grid = np.linspace(0.0, max(1e-6, c_max), 100)
    for c in c_grid:
        basis = subgamma_curve(lam, v=1.0, c=c)
        denom = float(np.sum(basis**2))
        if denom <= 0:
            continue
        v = float(np.sum(basis * mgf_vals) / denom)
        pred = subgamma_curve(lam, v=v, c=c)
        mse = float(np.mean((pred - mgf_vals) ** 2))
        if mse < best["mse"]:
            best = {"v": v, "c": c, "mse": mse}
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="Empirical MGF local checks on centered normalized sample loss.")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default="")
    p.add_argument("--lambda_min", type=float, default=0.01)
    p.add_argument("--lambda_max", type=float, default=0.40)
    p.add_argument("--lambda_points", type=int, default=40)
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    loss_theory_root = resolve_loss_theory_root(datainf_root, args.output_root)
    out_dir = ensure_dir(os.path.join(loss_theory_root, "analysis", "mgf_check"))

    sample_df = load_concat_csv(find_sample_csvs(loss_theory_root))
    vals = to_numeric_series(sample_df, "Lbar_i")
    if len(vals) < 50:
        raise RuntimeError("not enough Lbar_i rows for MGF diagnostics")
    x = vals - float(np.mean(vals))

    lam_grid = np.linspace(float(args.lambda_min), float(args.lambda_max), int(args.lambda_points))
    mgf_vals: List[float] = []
    stable_flags: List[int] = []
    for lam in lam_grid:
        try:
            val = float(log_mgf(x, float(lam)))
            ok = int(np.isfinite(val))
        except Exception:
            val = float("nan")
            ok = 0
        mgf_vals.append(val)
        stable_flags.append(ok)

    arr_lam = np.asarray(lam_grid, dtype=np.float64)
    arr_mgf = np.asarray(mgf_vals, dtype=np.float64)
    stable = np.isfinite(arr_mgf)
    if np.any(stable):
        first_bad_idx = int(np.where(~stable)[0][0]) if np.any(~stable) else int(len(arr_lam))
        stable_max_lambda = float(arr_lam[max(0, first_bad_idx - 1)])
    else:
        stable_max_lambda = float("nan")

    fit_rows: List[Dict[str, float]] = []
    intervals: List[Tuple[str, float]] = [("full", 1.0), ("half", 0.5), ("quarter", 0.25)]
    for name, frac in intervals:
        k = max(3, int(len(arr_lam) * frac))
        lam_i = arr_lam[:k]
        mgf_i = arr_mgf[:k]
        mask = np.isfinite(mgf_i)
        lam_i = lam_i[mask]
        mgf_i = mgf_i[mask]
        if len(lam_i) < 3:
            continue
        qfit = fit_quadratic_through_origin(lam_i, mgf_i)
        sfit = fit_subgamma_grid(lam_i, mgf_i)
        fit_rows.append(
            {
                "interval": name,
                "n_points": int(len(lam_i)),
                "quadratic_a": float(qfit["a"]),
                "quadratic_mse": float(qfit["mse"]),
                "subgamma_v": float(sfit["v"]),
                "subgamma_c": float(sfit["c"]),
                "subgamma_mse": float(sfit["mse"]),
            }
        )

    fit_df = pd.DataFrame(fit_rows)
    fit_csv = os.path.join(out_dir, "mgf_fit_summary.csv")
    fit_json = os.path.join(out_dir, "mgf_fit_summary.json")
    write_df_csv_json(fit_df, fit_csv, fit_json)

    curve_df = pd.DataFrame({"lambda": arr_lam, "Lambda_hat": arr_mgf, "is_stable": stable_flags})
    curve_csv = os.path.join(out_dir, "mgf_curve.csv")
    curve_json = os.path.join(out_dir, "mgf_curve.json")
    write_df_csv_json(curve_df, curve_csv, curve_json)

    fig = plt.figure(figsize=(7, 5))
    plt.plot(arr_lam, arr_mgf, marker="o", markersize=3, label="empirical Lambda_hat")
    if not fit_df.empty:
        best_idx = int(np.nanargmin(pd.to_numeric(fit_df["subgamma_mse"], errors="coerce").to_numpy()))
        best = fit_df.iloc[best_idx]
        v = float(best["subgamma_v"])
        c = float(best["subgamma_c"])
        pred = subgamma_curve(arr_lam, v=v, c=c)
        plt.plot(arr_lam, pred, linestyle="--", label=f"sub-gamma fit (v={v:.4g}, c={c:.4g})")
    plt.xlabel("lambda")
    plt.ylabel("Lambda_hat(lambda)")
    plt.title("Empirical log-MGF on centered Lbar_i")
    plt.legend(loc="best")
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "mgf_curve.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    summary = {
        "n_samples": int(len(vals)),
        "lambda_min": float(args.lambda_min),
        "lambda_max": float(args.lambda_max),
        "lambda_points": int(args.lambda_points),
        "stable_max_lambda": stable_max_lambda,
        "plot_path": os.path.abspath(plot_path),
        "fit_csv": os.path.abspath(fit_csv),
        "curve_csv": os.path.abspath(curve_csv),
    }
    summary_json = os.path.join(out_dir, "mgf_overview.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(os.path.abspath(fit_csv))
    print(os.path.abspath(curve_csv))
    print(os.path.abspath(summary_json))


if __name__ == "__main__":
    main()

