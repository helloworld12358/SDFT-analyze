#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from loss_theory_utils import (
    classify_tail,
    detect_datainf_root,
    ensure_dir,
    find_sample_csvs,
    find_seq_probe_csvs,
    find_token_csvs,
    fit_linear,
    load_concat_csv,
    positive_tail,
    resolve_loss_theory_root,
    survival_grid,
    tail_slice_by_quantile,
)

from gram_scheme_a_utils import DEFAULT_TASKS, DEFAULT_TRAIN_DATASETS  # type: ignore


DEFAULT_METHODS = ["sft", "sdft"]
DEFAULT_EPOCHS = ["epoch_1", "epoch_5"]

DEFAULT_METRICS = [
    "sample_lbar_mean",
    "sample_lbar_std",
    "sample_lbar_q90",
    "sample_lbar_q95",
    "sample_lbar_q99",
    "sample_tail_semilog_r2",
    "sample_tail_loglog_r2",
    "sample_tail_weibull_r2",
    "token_u_mean",
    "token_u_std",
    "token_u_q95",
    "token_u_q99",
    "seq_acf_lag1_mean",
]


def parse_csv_arg(raw: str, default: Sequence[str]) -> List[str]:
    s = (raw or "").strip()
    if not s:
        return list(default)
    vals = [x.strip() for x in s.split(",") if x.strip()]
    return vals if vals else list(default)


def to_float_array(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return vals


def quantile_or_nan(x: np.ndarray, q: float) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.quantile(x, q))


def _safe_tail_metrics(x: np.ndarray) -> Dict[str, float]:
    out = {
        "sample_tail_semilog_slope": float("nan"),
        "sample_tail_semilog_r2": float("nan"),
        "sample_tail_loglog_slope": float("nan"),
        "sample_tail_loglog_r2": float("nan"),
        "sample_tail_weibull_slope": float("nan"),
        "sample_tail_weibull_r2": float("nan"),
        "sample_tail_label_code": float("nan"),
    }
    if len(x) < 12:
        return out

    centered = x - float(np.mean(x))
    x_pos = positive_tail(centered)
    if len(x_pos) < 12:
        return out

    x_tail = tail_slice_by_quantile(x_pos, q_lo=0.90, q_hi=0.999)
    if len(x_tail) < 12:
        x_tail = x_pos

    gx, gs = survival_grid(np.sort(x_tail), max_points=256)
    valid = (gx > 0) & (gs > 0) & (gs < 1)
    gx = gx[valid]
    gs = gs[valid]
    if len(gx) < 8:
        return out

    semilog = fit_linear(gx, np.log(gs))
    loglog = fit_linear(np.log(gx), np.log(gs))
    w_valid = -np.log(gs) > 0
    wx = np.log(gx[w_valid])
    wy = np.log(-np.log(gs[w_valid]))
    weibull = fit_linear(wx, wy) if len(wx) >= 8 else {"slope": float("nan"), "r2": float("nan")}

    out["sample_tail_semilog_slope"] = float(semilog.get("slope", float("nan")))
    out["sample_tail_semilog_r2"] = float(semilog.get("r2", float("nan")))
    out["sample_tail_loglog_slope"] = float(loglog.get("slope", float("nan")))
    out["sample_tail_loglog_r2"] = float(loglog.get("r2", float("nan")))
    out["sample_tail_weibull_slope"] = float(weibull.get("slope", float("nan")))
    out["sample_tail_weibull_r2"] = float(weibull.get("r2", float("nan")))

    label = classify_tail(
        out["sample_tail_semilog_r2"],
        out["sample_tail_loglog_r2"],
        out["sample_tail_weibull_r2"],
        out["sample_tail_weibull_slope"],
    )
    label_code = {
        "power_law_like_heavy_tail": 1.0,
        "sub_weibull_like": 2.0,
        "sub_exponential_or_sub_gamma_like": 3.0,
        "sub_gaussian_like": 4.0,
        "mixed_or_unclear": 5.0,
    }.get(label, float("nan"))
    out["sample_tail_label_code"] = float(label_code)
    return out


def lag1_acf(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return float("nan")
    x = arr[:-1]
    y = arr[1:]
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 0 or sy <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def combo_key(row: pd.Series) -> Tuple[str, str, str, str]:
    return (
        str(row.get("train_dataset", "")),
        str(row.get("method", "")),
        str(row.get("epoch", "")),
        str(row.get("task", "")),
    )


def numeric_or_none(v: float) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def fmt_cell(v: float | None) -> str:
    if v is None:
        return "NA"
    if not math.isfinite(v):
        return "NA"
    return f"{v:.6f}"


def build_markdown_table(
    train_order: Sequence[str],
    task_order: Sequence[str],
    value_map: Dict[Tuple[str, str], float | None],
) -> List[str]:
    lines: List[str] = []
    lines.append("|train_dataset|" + "|".join(task_order) + "|")
    lines.append("|---|" + "|".join(["---:"] * len(task_order)) + "|")
    for tr in train_order:
        vals = [fmt_cell(value_map.get((tr, t), None)) for t in task_order]
        lines.append("|" + tr + "|" + "|".join(vals) + "|")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description="Build 7x5 combo-level theory tables in metric->epoch->method tree form.")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default="")
    p.add_argument("--train_datasets", type=str, default=",".join(DEFAULT_TRAIN_DATASETS))
    p.add_argument("--tasks", type=str, default="alpaca_eval,gsm8k,humaneval,multiarith,openfunction")
    p.add_argument("--methods", type=str, default="sft,sdft")
    p.add_argument("--epochs", type=str, default="epoch_1,epoch_5")
    p.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS))
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    root = resolve_loss_theory_root(datainf_root, args.output_root)
    out_dir = ensure_dir(os.path.join(root, "analysis", "combo_matrix"))

    train_order = parse_csv_arg(args.train_datasets, DEFAULT_TRAIN_DATASETS)
    task_order = parse_csv_arg(args.tasks, DEFAULT_TASKS)
    method_order = parse_csv_arg(args.methods, DEFAULT_METHODS)
    epoch_order = parse_csv_arg(args.epochs, DEFAULT_EPOCHS)
    metrics = parse_csv_arg(args.metrics, DEFAULT_METRICS)

    sample_paths = find_sample_csvs(root)
    token_paths = find_token_csvs(root)
    seq_paths = find_seq_probe_csvs(root)

    sample_df = load_concat_csv(sample_paths)
    token_df = load_concat_csv(token_paths)
    seq_df = load_concat_csv(seq_paths)

    if not sample_df.empty:
        sample_df["train_dataset"] = sample_df["train_dataset"].astype(str)
        sample_df["method"] = sample_df["method"].astype(str)
        sample_df["epoch"] = sample_df["epoch"].astype(str)
        sample_df["task"] = sample_df["task"].astype(str)
    if not token_df.empty:
        token_df["train_dataset"] = token_df["train_dataset"].astype(str)
        token_df["method"] = token_df["method"].astype(str)
        token_df["epoch"] = token_df["epoch"].astype(str)
        token_df["task"] = token_df["task"].astype(str)
    if not seq_df.empty:
        seq_df["train_dataset"] = seq_df["train_dataset"].astype(str)
        seq_df["method"] = seq_df["method"].astype(str)
        seq_df["epoch"] = seq_df["epoch"].astype(str)
        seq_df["task"] = seq_df["task"].astype(str)

    metric_by_combo: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}

    # sample-level metrics
    if not sample_df.empty and "Lbar_i" in sample_df.columns:
        gcols = ["train_dataset", "method", "epoch", "task"]
        for keys, sub in sample_df.groupby(gcols):
            vals = to_float_array(sub["Lbar_i"])
            if len(vals) == 0:
                continue
            m = metric_by_combo.setdefault((str(keys[0]), str(keys[1]), str(keys[2]), str(keys[3])), {})
            m["sample_lbar_mean"] = float(np.mean(vals))
            m["sample_lbar_std"] = float(np.std(vals))
            m["sample_lbar_q90"] = quantile_or_nan(vals, 0.90)
            m["sample_lbar_q95"] = quantile_or_nan(vals, 0.95)
            m["sample_lbar_q99"] = quantile_or_nan(vals, 0.99)
            m.update(_safe_tail_metrics(vals))

    # token-level metrics
    if not token_df.empty and "u_i_t" in token_df.columns:
        gcols = ["train_dataset", "method", "epoch", "task"]
        for keys, sub in token_df.groupby(gcols):
            vals = to_float_array(sub["u_i_t"])
            if len(vals) == 0:
                continue
            m = metric_by_combo.setdefault((str(keys[0]), str(keys[1]), str(keys[2]), str(keys[3])), {})
            m["token_u_mean"] = float(np.mean(vals))
            m["token_u_std"] = float(np.std(vals))
            m["token_u_q95"] = quantile_or_nan(vals, 0.95)
            m["token_u_q99"] = quantile_or_nan(vals, 0.99)

    # sequence dependence proxy
    if not seq_df.empty and {"sample_idx", "token_pos", "u_i_t"}.issubset(set(seq_df.columns)):
        seq_df = seq_df.copy()
        seq_df["sample_idx"] = pd.to_numeric(seq_df["sample_idx"], errors="coerce")
        seq_df["token_pos"] = pd.to_numeric(seq_df["token_pos"], errors="coerce")
        seq_df["u_i_t"] = pd.to_numeric(seq_df["u_i_t"], errors="coerce")
        seq_df = seq_df.dropna(subset=["sample_idx", "token_pos", "u_i_t"])
        gcols = ["train_dataset", "method", "epoch", "task"]
        for keys, sub in seq_df.groupby(gcols):
            acfs: List[float] = []
            for _, one in sub.groupby("sample_idx"):
                one = one.sort_values("token_pos")
                arr = one["u_i_t"].to_numpy(dtype=np.float64)
                a = lag1_acf(arr)
                if math.isfinite(a):
                    acfs.append(float(a))
            if not acfs:
                continue
            m = metric_by_combo.setdefault((str(keys[0]), str(keys[1]), str(keys[2]), str(keys[3])), {})
            m["seq_acf_lag1_mean"] = float(np.mean(acfs))

    # full grid rows (7x5 for each epoch/method)
    rows: List[Dict[str, object]] = []
    expected_keys: List[Tuple[str, str, str, str]] = []
    for tr in train_order:
        for me in method_order:
            for ep in epoch_order:
                for ta in task_order:
                    key = (tr, me, ep, ta)
                    expected_keys.append(key)
                    row: Dict[str, object] = {
                        "train_dataset": tr,
                        "method": me,
                        "epoch": ep,
                        "task": ta,
                    }
                    combo_metrics = metric_by_combo.get(key, {})
                    for metric in sorted(set(DEFAULT_METRICS + metrics)):
                        row[metric] = numeric_or_none(combo_metrics.get(metric, float("nan")))
                    rows.append(row)

    long_df = pd.DataFrame(rows)
    long_csv = os.path.join(out_dir, "combo_metric_long.csv")
    long_json = os.path.join(out_dir, "combo_metric_long.json")
    long_df.to_csv(long_csv, index=False)
    long_df.to_json(long_json, orient="records", force_ascii=False, indent=2)

    # tree tables
    table_tree: Dict[str, Dict[str, Dict[str, object]]] = {}
    txt_lines: List[str] = []
    txt_lines.append("# Loss Theory Combo Matrix Tables")
    txt_lines.append("")
    txt_lines.append(f"- rows(train_dataset): {', '.join(train_order)}")
    txt_lines.append(f"- cols(task): {', '.join(task_order)}")
    txt_lines.append(f"- epochs: {', '.join(epoch_order)}")
    txt_lines.append(f"- methods: {', '.join(method_order)}")
    txt_lines.append("")
    txt_lines.append("Tree order: metric -> epoch -> method")
    txt_lines.append("")

    for metric in metrics:
        table_tree[metric] = {}
        txt_lines.append(f"## {metric}")
        txt_lines.append("")
        for ep in epoch_order:
            table_tree[metric][ep] = {}
            txt_lines.append(f"### {ep}")
            txt_lines.append("")
            for me in method_order:
                sub = long_df[(long_df["epoch"] == ep) & (long_df["method"] == me)]
                value_map: Dict[Tuple[str, str], float | None] = {}
                for _, r in sub.iterrows():
                    k = (str(r["train_dataset"]), str(r["task"]))
                    vv = r.get(metric, None)
                    value_map[k] = numeric_or_none(float(vv)) if vv is not None and not pd.isna(vv) else None

                rows_matrix: List[List[float | None]] = []
                for tr in train_order:
                    one_row: List[float | None] = []
                    for ta in task_order:
                        one_row.append(value_map.get((tr, ta), None))
                    rows_matrix.append(one_row)

                table_tree[metric][ep][me] = {
                    "rows": list(train_order),
                    "cols": list(task_order),
                    "values": rows_matrix,
                }

                txt_lines.append(f"#### {me}")
                txt_lines.append("")
                txt_lines.extend(build_markdown_table(train_order, task_order, value_map))
                txt_lines.append("")

    table_json = os.path.join(out_dir, "combo_metric_tree_tables.json")
    table_txt = os.path.join(out_dir, "combo_metric_tree_tables.txt")
    payload = {
        "output_root": os.path.abspath(root),
        "analysis_dir": os.path.abspath(out_dir),
        "tree_order": ["metric", "epoch", "method"],
        "train_order": train_order,
        "task_order": task_order,
        "method_order": method_order,
        "epoch_order": epoch_order,
        "metrics": metrics,
        "tables": table_tree,
    }
    with open(table_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(table_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    # unavailable / missing diagnostics
    missing: List[Dict[str, str]] = []
    for tr, me, ep, ta in expected_keys:
        if (tr, me, ep, ta) not in metric_by_combo:
            missing.append(
                {
                    "train_dataset": tr,
                    "method": me,
                    "epoch": ep,
                    "task": ta,
                    "reason": "no sample_stats/token_stats/seq_stats found for this combo",
                }
            )
    unavailable = {
        "sample_csv_count": len(sample_paths),
        "token_csv_count": len(token_paths),
        "seq_csv_count": len(seq_paths),
        "expected_combo_count": len(expected_keys),
        "observed_combo_count": len(metric_by_combo),
        "missing_combo_count": len(missing),
        "missing_combos": missing,
    }
    unavailable_json = os.path.join(out_dir, "unavailable_combo_metric_tables.json")
    with open(unavailable_json, "w", encoding="utf-8") as f:
        json.dump(unavailable, f, ensure_ascii=False, indent=2)

    print(os.path.abspath(long_csv))
    print(os.path.abspath(long_json))
    print(os.path.abspath(table_json))
    print(os.path.abspath(table_txt))
    print(os.path.abspath(unavailable_json))


if __name__ == "__main__":
    main()

