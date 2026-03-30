#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import glob
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAINF_ROOT_DEFAULT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(DATAINF_ROOT_DEFAULT, "src")

import sys

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gram_scheme_a_utils import detect_datainf_root, ensure_dir, resolve_result_root  # type: ignore  # noqa: E402


def resolve_loss_theory_root(datainf_root: str, output_root: str = "") -> str:
    if output_root.strip():
        return os.path.abspath(output_root.strip())
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    return os.path.join(result_root, "loss_theory")


def find_sample_csvs(loss_theory_root: str) -> List[str]:
    pattern = os.path.join(loss_theory_root, "by_combo", "*", "*", "*", "*", "sample_stats.csv")
    return sorted(glob.glob(pattern))


def find_token_csvs(loss_theory_root: str) -> List[str]:
    pattern = os.path.join(loss_theory_root, "by_combo", "*", "*", "*", "*", "token_subsample_stats.csv")
    return sorted(glob.glob(pattern))


def find_seq_probe_csvs(loss_theory_root: str) -> List[str]:
    pattern = os.path.join(loss_theory_root, "by_combo", "*", "*", "*", "*", "token_sequence_probe_stats.csv")
    return sorted(glob.glob(pattern))


def load_concat_csv(paths: Sequence[str]) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if not df.empty:
                dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, axis=0, ignore_index=True)


def to_numeric_series(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.asarray([], dtype=np.float64)
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return vals


def fit_linear(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if len(x) < 3 or len(y) < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "n": int(len(x))}
    coef = np.polyfit(x, y, deg=1)
    slope = float(coef[0])
    intercept = float(coef[1])
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": slope, "intercept": intercept, "r2": r2, "n": int(len(x))}


def positive_tail(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    y = y[np.isfinite(y)]
    y = y[y > 0]
    return np.sort(y)


def survival_grid(x_pos_sorted: np.ndarray, max_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x_pos_sorted)
    if n == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
    x = x_pos_sorted[idx]
    s = (n - idx) / float(n)
    return x, s


def tail_slice_by_quantile(x_pos_sorted: np.ndarray, q_lo: float, q_hi: float) -> np.ndarray:
    if len(x_pos_sorted) == 0:
        return np.asarray([], dtype=np.float64)
    lo = np.quantile(x_pos_sorted, q_lo)
    hi = np.quantile(x_pos_sorted, q_hi)
    return x_pos_sorted[(x_pos_sorted >= lo) & (x_pos_sorted <= hi)]


def classify_tail(semilog_r2: float, loglog_r2: float, weibull_r2: float, weibull_slope: float) -> str:
    if np.isfinite(loglog_r2) and loglog_r2 >= 0.97:
        return "power_law_like_heavy_tail"
    if np.isfinite(weibull_r2) and weibull_r2 >= 0.97:
        if np.isfinite(weibull_slope):
            if weibull_slope >= 1.8:
                return "sub_gaussian_like"
            if weibull_slope >= 0.9:
                return "sub_exponential_or_sub_gamma_like"
            return "sub_weibull_like"
    if np.isfinite(semilog_r2) and semilog_r2 >= 0.97:
        return "sub_exponential_or_sub_gamma_like"
    return "mixed_or_unclear"


def log_mgf(x: np.ndarray, lam: float) -> float:
    z = lam * x
    m = float(np.max(z))
    return m + math.log(float(np.mean(np.exp(z - m))))


def trimmed_mean(x: np.ndarray, alpha: float = 0.1) -> float:
    y = np.sort(np.asarray(x, dtype=np.float64))
    n = len(y)
    if n == 0:
        return float("nan")
    k = int(max(0, min(n // 2 - 1, math.floor(alpha * n))))
    if k <= 0:
        return float(np.mean(y))
    return float(np.mean(y[k : n - k]))


def median_of_means(x: np.ndarray, n_blocks: int = 8) -> float:
    y = np.asarray(x, dtype=np.float64)
    n = len(y)
    if n == 0:
        return float("nan")
    k = int(max(2, min(n, n_blocks)))
    perm = np.random.permutation(n)
    blocks = np.array_split(y[perm], k)
    means = np.asarray([np.mean(b) for b in blocks if len(b) > 0], dtype=np.float64)
    return float(np.median(means))


def catoni_m_estimator(x: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> float:
    y = np.asarray(x, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = len(y)
    if n == 0:
        return float("nan")
    m = float(np.mean(y))
    s = float(np.std(y)) + 1e-12
    alpha = 1.0 / (2.0 * s)
    for _ in range(max_iter):
        u = alpha * (y - m)
        psi = np.tanh(u)
        grad = -alpha * np.mean(1.0 - np.tanh(u) ** 2)
        step = np.mean(psi) / (grad - 1e-12)
        m_new = m - step
        if abs(m_new - m) <= tol:
            return float(m_new)
        m = float(m_new)
    return float(m)


def write_df_csv_json(df: pd.DataFrame, csv_path: str, json_path: str) -> Tuple[str, str]:
    ensure_dir(os.path.dirname(os.path.abspath(csv_path)))
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    return csv_path, json_path

