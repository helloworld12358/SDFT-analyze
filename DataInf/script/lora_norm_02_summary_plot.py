#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize LoRA norm rows into:
- txt tables
- csv/json wide tables
- line plots (epoch_0 -> epoch_1 -> epoch_5)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAINF_ROOT_DEFAULT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(DATAINF_ROOT_DEFAULT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gram_scheme_a_utils import (  # type: ignore  # noqa: E402
    DEFAULT_EPOCHS,
    DEFAULT_TRAIN_DATASETS,
    detect_datainf_root,
    resolve_result_root,
)


def split_csv_arg(s: str, default: Sequence[str]) -> List[str]:
    if not s.strip():
        return list(default)
    out = [x.strip() for x in s.split(",") if x.strip()]
    return out if out else list(default)


def resolve_output_root(datainf_root: str, output_root: str) -> str:
    if output_root.strip():
        return os.path.abspath(output_root.strip())
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    return os.path.join(result_root, "lora_norm")


def to_float(v: object) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)  # type: ignore[arg-type]
        if math.isfinite(f):
            return f
        return None
    except Exception:
        return None


def fmt(v: Optional[float], digits: int = 6) -> str:
    if v is None:
        return "NA"
    return f"{v:.{digits}f}"


def get_metric_value(df: pd.DataFrame, train_dataset: str, method: str, epoch: str, metric: str) -> Optional[float]:
    sub = df[
        (df["train_dataset"] == train_dataset)
        & (df["method"] == method)
        & (df["epoch"] == epoch)
    ]
    if sub.empty:
        return None
    v = sub.iloc[0].get(metric, None)
    return to_float(v)


def table_lines_for_metric(df: pd.DataFrame, train_datasets: Sequence[str], epochs: Sequence[str], metric: str, title: str) -> List[str]:
    methods = ["sft", "sdft"]
    cols: List[Tuple[str, str]] = []
    for method in methods:
        for epoch in epochs:
            cols.append((method, epoch))

    lines: List[str] = []
    lines.append(f"## {title} ({metric})")
    lines.append("")
    header = ["train_dataset"] + [f"{m}_{e}" for m, e in cols]
    lines.append("|" + "|".join(header) + "|")
    lines.append("|" + "|".join(["---"] + ["---:"] * len(cols)) + "|")
    for train_dataset in train_datasets:
        row = [train_dataset]
        for method, epoch in cols:
            v = get_metric_value(df, train_dataset, method, epoch, metric)
            row.append(fmt(v))
        lines.append("|" + "|".join(row) + "|")
    lines.append("")
    return lines


def build_wide_rows(df: pd.DataFrame, train_datasets: Sequence[str], epochs: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for train_dataset in train_datasets:
        row: Dict[str, object] = {"train_dataset": train_dataset}
        for metric in ["ba_norm_global", "a_norm_global", "b_norm_global"]:
            for method in ["sft", "sdft"]:
                for epoch in epochs:
                    k = f"{metric}__{method}__{epoch}"
                    row[k] = get_metric_value(df, train_dataset, method, epoch, metric)
        rows.append(row)
    return rows


def plot_metric(
    df: pd.DataFrame,
    train_datasets: Sequence[str],
    epochs: Sequence[str],
    metric: str,
    ylabel: str,
    out_png: str,
) -> None:
    x = list(range(len(epochs)))
    n = len(train_datasets)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for i, train_dataset in enumerate(train_datasets):
        ax = axes_flat[i]
        y_sft = [get_metric_value(df, train_dataset, "sft", ep, metric) for ep in epochs]
        y_sdft = [get_metric_value(df, train_dataset, "sdft", ep, metric) for ep in epochs]
        y_sft_plot = [float("nan") if v is None else v for v in y_sft]
        y_sdft_plot = [float("nan") if v is None else v for v in y_sdft]

        ax.plot(x, y_sft_plot, marker="o", linewidth=1.8, label="SFT")
        ax.plot(x, y_sdft_plot, marker="o", linewidth=1.8, label="SDFT")
        ax.set_title(train_dataset)
        ax.set_xticks(x)
        ax.set_xticklabels(list(epochs))
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linestyle="--")
        if i == 0:
            ax.legend(loc="best")

    for j in range(len(train_datasets), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"LoRA Norm Curves | {metric}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Build lora norm summary tables and plots.")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default="")
    p.add_argument("--train_datasets", type=str, default=",".join(DEFAULT_TRAIN_DATASETS))
    p.add_argument("--epochs", type=str, default="epoch_0,epoch_1,epoch_5")
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    output_root = resolve_output_root(datainf_root, args.output_root)
    os.makedirs(output_root, exist_ok=True)

    train_datasets = split_csv_arg(args.train_datasets, DEFAULT_TRAIN_DATASETS)
    epochs = split_csv_arg(args.epochs, DEFAULT_EPOCHS)

    rows_csv = os.path.join(output_root, "lora_norm_rows.csv")
    rows_json = os.path.join(output_root, "lora_norm_rows.json")
    if os.path.isfile(rows_csv):
        df = pd.read_csv(rows_csv)
    elif os.path.isfile(rows_json):
        with open(rows_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        df = pd.DataFrame(obj if isinstance(obj, list) else [])
    else:
        raise FileNotFoundError(f"missing lora_norm_rows.csv/json under {output_root}")

    if df.empty:
        raise RuntimeError("lora_norm rows are empty")

    for col in ["train_dataset", "method", "epoch"]:
        if col not in df.columns:
            raise RuntimeError(f"missing required column: {col}")
        df[col] = df[col].astype(str)

    wide_rows = build_wide_rows(df, train_datasets, epochs)
    wide_csv = os.path.join(output_root, "lora_norm_wide.csv")
    wide_json = os.path.join(output_root, "lora_norm_wide.json")
    pd.DataFrame(wide_rows).to_csv(wide_csv, index=False)
    with open(wide_json, "w", encoding="utf-8") as f:
        json.dump(wide_rows, f, ensure_ascii=False, indent=2)

    txt_lines: List[str] = []
    txt_lines.append("# LoRA Norm Summary")
    txt_lines.append("")
    txt_lines.append("Definitions:")
    txt_lines.append("- BA_norm_global = sqrt(sum_modules ||(alpha/r) * B @ A||_F^2)")
    txt_lines.append("- A_norm_global  = sqrt(sum_modules ||A||_F^2)")
    txt_lines.append("- B_norm_global  = sqrt(sum_modules ||B||_F^2)")
    txt_lines.append("- epoch_0 is fixed at 0 (base model only, no adapter)")
    txt_lines.append("")
    txt_lines.extend(table_lines_for_metric(df, train_datasets, epochs, "ba_norm_global", "Table 1: BA Norm"))
    txt_lines.extend(table_lines_for_metric(df, train_datasets, epochs, "a_norm_global", "Table 2: A Norm"))
    txt_lines.extend(table_lines_for_metric(df, train_datasets, epochs, "b_norm_global", "Table 3: B Norm"))
    summary_txt = os.path.join(output_root, "lora_norm_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    plots_dir = os.path.join(output_root, "plots")
    ba_png = os.path.join(plots_dir, "lora_norm_ba_curves.png")
    a_png = os.path.join(plots_dir, "lora_norm_a_curves.png")
    b_png = os.path.join(plots_dir, "lora_norm_b_curves.png")
    plot_metric(df, train_datasets, epochs, "ba_norm_global", "global Fro norm", ba_png)
    plot_metric(df, train_datasets, epochs, "a_norm_global", "global Fro norm", a_png)
    plot_metric(df, train_datasets, epochs, "b_norm_global", "global Fro norm", b_png)

    out_meta = {
        "output_root": os.path.abspath(output_root),
        "rows_csv": os.path.abspath(rows_csv) if os.path.isfile(rows_csv) else None,
        "rows_json": os.path.abspath(rows_json) if os.path.isfile(rows_json) else None,
        "wide_csv": os.path.abspath(wide_csv),
        "wide_json": os.path.abspath(wide_json),
        "summary_txt": os.path.abspath(summary_txt),
        "plots": [os.path.abspath(ba_png), os.path.abspath(a_png), os.path.abspath(b_png)],
    }
    out_json = os.path.join(output_root, "lora_norm_summary_meta.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_meta, f, ensure_ascii=False, indent=2)

    print(os.path.abspath(wide_csv))
    print(os.path.abspath(wide_json))
    print(os.path.abspath(summary_txt))
    print(os.path.abspath(ba_png))
    print(os.path.abspath(a_png))
    print(os.path.abspath(b_png))
    print(os.path.abspath(out_json))


if __name__ == "__main__":
    main()

