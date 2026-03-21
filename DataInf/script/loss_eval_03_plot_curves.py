#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot SFT vs SDFT loss curves.

Output style:
- 7 figures (one per train dataset)
- each figure has 5 subplots (one per test task)
- x-axis: epoch_0, epoch_1, epoch_5
- two lines: SFT and SDFT
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

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
    write_unavailable_note,
)


TASKS_5 = ["alpaca_eval", "gsm8k", "humaneval", "multiarith", "openfunction"]


def split_csv_arg(s: str, default: Sequence[str]) -> List[str]:
    if not s.strip():
        return list(default)
    out = [x.strip() for x in s.split(",") if x.strip()]
    return out if out else list(default)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rows_list(path: str) -> List[Dict[str, object]]:
    if not os.path.isfile(path):
        return []
    obj = load_json(path)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


def to_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def collect_rows(output_root: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for p in sorted(glob.glob(os.path.join(output_root, "loss_rows_all_*.json"))):
        rows.extend(load_rows_list(p))
    if rows:
        return rows
    for p in sorted(glob.glob(os.path.join(output_root, "by_train_dataset", "*", "loss_rows_*.json"))):
        rows.extend(load_rows_list(p))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Plot SFT vs SDFT loss curves")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default="", help="Default: <result_root>/loss_eval")
    p.add_argument("--train_datasets", type=str, default="")
    p.add_argument("--epochs", type=str, default="epoch_0,epoch_1,epoch_5")
    p.add_argument("--tasks", type=str, default="alpaca_eval,gsm8k,humaneval,multiarith,openfunction")
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    output_root = args.output_root.strip() or os.path.join(result_root, "loss_eval")
    os.makedirs(output_root, exist_ok=True)

    train_datasets = split_csv_arg(args.train_datasets, DEFAULT_TRAIN_DATASETS)
    epochs = split_csv_arg(args.epochs, DEFAULT_EPOCHS)
    tasks = split_csv_arg(args.tasks, TASKS_5)

    rows = collect_rows(output_root)
    if not rows:
        note = write_unavailable_note(
            os.path.join(output_root, "unavailable_loss_plots.json"),
            reason="no loss rows found under output_root",
            context={"output_root": os.path.abspath(output_root)},
        )
        print(os.path.abspath(note))
        return

    plots_dir = os.path.join(output_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for train_dataset in train_datasets:
        fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.8), squeeze=False)
        axes_flat = axes[0]
        fig.suptitle(f"Loss Curves | train_dataset={train_dataset}", fontsize=12)

        for i, task in enumerate(tasks):
            ax = axes_flat[i]
            x = list(range(len(epochs)))
            y_sft: List[float] = []
            y_sdft: List[float] = []
            for ep in epochs:
                sft_loss = None
                sdft_loss = None
                for r in rows:
                    if str(r.get("train_dataset", "")) != train_dataset:
                        continue
                    if str(r.get("epoch", "")) != ep:
                        continue
                    if str(r.get("test_task", "")) != task:
                        continue
                    if str(r.get("status", "")) != "ok":
                        continue
                    method = str(r.get("method", ""))
                    if method == "sft":
                        sft_loss = to_float(r.get("loss_mean_token"))
                    elif method == "sdft":
                        sdft_loss = to_float(r.get("loss_mean_token"))
                y_sft.append(float("nan") if sft_loss is None else sft_loss)
                y_sdft.append(float("nan") if sdft_loss is None else sdft_loss)

            ax.plot(x, y_sft, marker="o", linewidth=1.8, label="SFT")
            ax.plot(x, y_sdft, marker="o", linewidth=1.8, label="SDFT")
            ax.set_title(task)
            ax.set_xticks(x)
            ax.set_xticklabels(epochs, rotation=20)
            ax.set_ylabel("mean token CE loss")
            ax.grid(True, alpha=0.25, linestyle="--")
            if i == 0:
                ax.legend(loc="best")

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        out_png = os.path.join(plots_dir, f"loss_curves_{train_dataset}.png")
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        print(os.path.abspath(out_png))


if __name__ == "__main__":
    main()

