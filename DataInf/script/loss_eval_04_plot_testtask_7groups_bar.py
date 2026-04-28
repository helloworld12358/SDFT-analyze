#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 5 figures for loss comparison:
- One figure per test task.
- Each figure contains 7 small subplots (train datasets).
- Each subplot has 6 thin bars:
  epoch_0(SFT), epoch_0(SDFT), epoch_1(SFT), epoch_1(SDFT), epoch_5(SFT), epoch_5(SDFT)

Notes:
- If epoch_5 is missing for a (train_dataset, method, task), fallback to the
  largest available epoch_N where N > 1 (e.g., epoch_2).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRAIN_ORDER_DEFAULT = [
    "gsm8k",
    "openfunction",
    "magicoder",
    "alpaca",
    "dolly",
    "lima",
    "openhermes",
]
TASKS_DEFAULT = ["alpaca_eval", "gsm8k", "humaneval", "multiarith", "openfunction"]
METHODS_INTERLEAVED = ["sft", "sdft"]
EPOCHS_TARGET = ["epoch_0", "epoch_1", "epoch_5"]


def split_csv_arg(s: str, default: Sequence[str]) -> List[str]:
    if not s.strip():
        return list(default)
    out = [x.strip() for x in s.split(",") if x.strip()]
    return out if out else list(default)


def detect_result_roots(datainf_root: Path) -> List[Path]:
    roots = []
    for name in ["results", "result"]:
        p = datainf_root / name
        if p.is_dir():
            roots.append(p)
    return roots


def try_load_from_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    required = {"train_dataset", "method", "epoch"}
    if not required.issubset(set(df.columns)):
        return None
    return df


def try_load_from_json(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    rows = None
    if isinstance(obj, dict) and isinstance(obj.get("wide_rows"), list):
        rows = obj.get("wide_rows")
    elif isinstance(obj, list):
        rows = obj
    if rows is None:
        return None
    df = pd.DataFrame(rows)
    required = {"train_dataset", "method", "epoch"}
    if not required.issubset(set(df.columns)):
        return None
    return df


def locate_wide_table(datainf_root: Path, matrix_csv: str) -> Tuple[pd.DataFrame, Path]:
    if matrix_csv.strip():
        p = Path(matrix_csv).expanduser().resolve()
        df = try_load_from_csv(p)
        if df is None:
            raise FileNotFoundError(f"Invalid or unreadable matrix_csv: {p}")
        return df, p

    candidates: List[Path] = []
    for rr in detect_result_roots(datainf_root):
        candidates.extend(
            [
                rr / "loss_eval" / "loss_tables_7x3x5_sft__sdft.csv",
                rr / "loss_eval_test2" / "loss_tables_7x3x5_sft__sdft.csv",
                rr / "loss_eval" / "loss_tables_7x3x5_sft__sdft.json",
                rr / "loss_eval_test2" / "loss_tables_7x3x5_sft__sdft.json",
            ]
        )

    for p in candidates:
        if p.suffix.lower() == ".csv":
            df = try_load_from_csv(p)
        else:
            df = try_load_from_json(p)
        if df is None:
            continue
        value_cols = [c for c in df.columns if c not in ("train_dataset", "method", "epoch")]
        if not value_cols:
            continue
        # Accept the table if not all numeric cells are NaN.
        num = df[value_cols].apply(pd.to_numeric, errors="coerce")
        if not bool(num.notna().any().any()):
            continue
        return df, p

    raise FileNotFoundError(
        "Cannot find a valid loss wide table with non-NaN values. "
        "Tried loss_eval/loss_eval_test2 *loss_tables_7x3x5_sft__sdft*."
    )


def epoch_num(epoch: str) -> Optional[int]:
    m = re.match(r"^epoch_(\d+)$", str(epoch).strip())
    if not m:
        return None
    return int(m.group(1))


def build_lookup(df: pd.DataFrame, tasks: Sequence[str]) -> Dict[Tuple[str, str, str, str], Optional[float]]:
    lookup: Dict[Tuple[str, str, str, str], Optional[float]] = {}
    for _, row in df.iterrows():
        train = str(row.get("train_dataset", "")).strip().lower()
        method = str(row.get("method", "")).strip().lower()
        epoch = str(row.get("epoch", "")).strip().lower()
        if not train or not method or not epoch:
            continue
        for task in tasks:
            v = row.get(task)
            vv = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
            lookup[(train, method, epoch, task)] = None if pd.isna(vv) else float(vv)
    return lookup


def available_epochs(lookup: Dict[Tuple[str, str, str, str], Optional[float]], train: str, method: str, task: str) -> List[str]:
    out = []
    for (tr, me, ep, ta), val in lookup.items():
        if tr == train and me == method and ta == task and val is not None:
            out.append(ep)
    return sorted(set(out), key=lambda x: (epoch_num(x) is None, epoch_num(x) if epoch_num(x) is not None else 10**9))


def pick_value(
    lookup: Dict[Tuple[str, str, str, str], Optional[float]],
    train: str,
    method: str,
    target_epoch: str,
    task: str,
) -> Tuple[Optional[float], str]:
    k = (train, method, target_epoch, task)
    v = lookup.get(k)
    if v is not None:
        return v, target_epoch

    # Fallback only for target epoch_5: use largest available epoch_N where N > 1.
    if target_epoch == "epoch_5":
        eps = available_epochs(lookup, train, method, task)
        eps_num = [(e, epoch_num(e)) for e in eps]
        eps_num = [(e, n) for e, n in eps_num if n is not None and n > 1]
        if eps_num:
            e_best = max(eps_num, key=lambda x: x[1])[0]
            v2 = lookup.get((train, method, e_best, task))
            if v2 is not None:
                return v2, e_best
    return None, target_epoch


def make_task_figure(
    lookup: Dict[Tuple[str, str, str, str], Optional[float]],
    task: str,
    train_datasets: Sequence[str],
    out_dir: Path,
    fmt: str,
) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    axes = axes.flatten()

    x = np.arange(6)
    x_labels = ["E0-S", "E0-D", "E1-S", "E1-D", "E5-S", "E5-D"]
    colors = ["#4C78A8", "#F58518", "#4C78A8", "#F58518", "#4C78A8", "#F58518"]

    fallback_notes: List[str] = []

    for i, train in enumerate(train_datasets):
        ax = axes[i]
        train_l = train.lower()
        vals: List[float] = []
        labels_used: List[str] = []
        for ep in EPOCHS_TARGET:
            for method in METHODS_INTERLEAVED:
                v, ep_used = pick_value(lookup, train_l, method, ep, task)
                vals.append(np.nan if v is None else float(v))
                labels_used.append(ep_used)
                if ep == "epoch_5" and ep_used != "epoch_5":
                    fallback_notes.append(f"{train}:{method}:{task} uses {ep_used} as E5")

        ax.bar(x, vals, color=colors, width=0.45, edgecolor="black", linewidth=0.4)
        ax.set_title(train, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Loss", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

        ymax = np.nanmax(vals) if np.isfinite(np.nanmax(vals)) else 1.0
        for xi, yi in zip(x, vals):
            if np.isnan(yi):
                ax.text(xi, ymax * 0.04, "NA", ha="center", va="bottom", fontsize=7, color="red", rotation=90)
            else:
                ax.text(xi, yi + max(0.01, ymax * 0.01), f"{yi:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    # Hide the last empty subplot (8th slot).
    if len(train_datasets) < len(axes):
        for j in range(len(train_datasets), len(axes)):
            axes[j].axis("off")

    fig.suptitle(f"Loss Bars by Train Dataset | Test Task: {task}", fontsize=14)

    if fallback_notes:
        note = " ; ".join(sorted(set(fallback_notes))[:5])
        fig.text(
            0.01,
            0.01,
            f"Note: E5 fallback detected (showing first few): {note}",
            fontsize=8,
            color="#555555",
        )

    out_path = out_dir / f"loss_bar_7groups_{task}.{fmt}"
    fig.savefig(out_path, dpi=200 if fmt.lower() == "png" else None, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 5 figures: each test task with 7 train-dataset bar panels.")
    parser.add_argument("--datainf_root", type=str, default="", help="Default auto: <repo>/DataInf")
    parser.add_argument("--matrix_csv", type=str, default="", help="Optional explicit path to loss_tables_7x3x5_sft__sdft.csv")
    parser.add_argument("--output_dir", type=str, default="", help="Default: <result_root>/loss_eval_task_panels")
    parser.add_argument("--train_datasets", type=str, default="gsm8k,openfunction,magicoder,alpaca,dolly,lima,openhermes")
    parser.add_argument("--tasks", type=str, default="alpaca_eval,gsm8k,humaneval,multiarith,openfunction")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    datainf_root = Path(args.datainf_root).resolve() if args.datainf_root.strip() else (script_dir / "..").resolve()
    train_datasets = split_csv_arg(args.train_datasets, TRAIN_ORDER_DEFAULT)
    tasks = split_csv_arg(args.tasks, TASKS_DEFAULT)

    result_roots = detect_result_roots(datainf_root)
    if not result_roots:
        raise FileNotFoundError(f"No result/results directory under {datainf_root}")
    result_root = result_roots[0]

    output_dir = Path(args.output_dir).resolve() if args.output_dir.strip() else (result_root / "loss_eval_task_panels")
    output_dir.mkdir(parents=True, exist_ok=True)

    df, src = locate_wide_table(datainf_root, args.matrix_csv)
    lookup = build_lookup(df, tasks)

    print(f"[source] {src}")
    for task in tasks:
        p = make_task_figure(lookup, task, train_datasets, output_dir, args.format)
        print(str(p.resolve()))


if __name__ == "__main__":
    main()

