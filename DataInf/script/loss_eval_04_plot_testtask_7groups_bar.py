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


def detect_loss_eval_roots(datainf_root: Path) -> List[Path]:
    roots: List[Path] = []
    for rr in detect_result_roots(datainf_root):
        p = rr / "loss_eval"
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


def has_non_nan_value(df: pd.DataFrame) -> bool:
    value_cols = [c for c in df.columns if c not in ("train_dataset", "method", "epoch")]
    if not value_cols:
        return False
    num = df[value_cols].apply(pd.to_numeric, errors="coerce")
    return bool(num.notna().any().any())


def load_rows_list_json(path: Path) -> List[Dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        rows = obj.get("rows")
        if isinstance(rows, list):
            return [x for x in rows if isinstance(x, dict)]
    return []


def build_wide_table_from_loss_eval_rows(datainf_root: Path, loss_eval_root: str = "") -> Optional[Tuple[pd.DataFrame, Path]]:
    roots: List[Path] = []
    if loss_eval_root.strip():
        p = Path(loss_eval_root).expanduser().resolve()
        if p.is_dir():
            roots.append(p)
    else:
        roots.extend(detect_loss_eval_roots(datainf_root))

    for root in roots:
        files: List[Path] = []
        files.extend(Path(x) for x in sorted((root).glob("loss_rows_all_*.json")))
        files.extend(Path(x) for x in sorted((root / "by_train_dataset").glob("*/loss_rows_*.json")))
        if not files:
            continue

        rows: List[Dict[str, object]] = []
        seen = set()
        for fp in files:
            for r in load_rows_list_json(fp):
                train = str(r.get("train_dataset", "")).strip().lower()
                method = str(r.get("method", "")).strip().lower()
                epoch = str(r.get("epoch", "")).strip().lower()
                task = str(r.get("test_task", "")).strip().lower()
                if not train or not method or not epoch or not task:
                    continue
                if method not in ("sft", "sdft"):
                    continue
                status = str(r.get("status", "")).strip().lower()
                if status and status != "ok":
                    continue
                val = pd.to_numeric(pd.Series([r.get("loss_mean_token")]), errors="coerce").iloc[0]
                if pd.isna(val):
                    continue
                key = (train, method, epoch, task)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "train_dataset": train,
                        "method": method,
                        "epoch": epoch,
                        "task": task,
                        "loss_mean": float(val),
                    }
                )

        if not rows:
            continue

        agg = pd.DataFrame(rows)
        wide = agg.pivot_table(
            index=["train_dataset", "method", "epoch"],
            columns="task",
            values="loss_mean",
            aggfunc="mean",
        ).reset_index()
        wide.columns.name = None
        if has_non_nan_value(wide):
            return wide, root
    return None


def parse_combo_from_path(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    # Expected pattern:
    # .../loss_theory/by_combo/<train_dataset>/<method>/<epoch>/<task>/sample_stats.csv
    parts = [p.lower() for p in path.parts]
    try:
        i = parts.index("by_combo")
        train = parts[i + 1]
        method = parts[i + 2]
        epoch = parts[i + 3]
        task = parts[i + 4]
        return train, method, epoch, task
    except Exception:
        return None, None, None, None


def build_wide_table_from_sample_stats(datainf_root: Path) -> Optional[pd.DataFrame]:
    files: List[Path] = []
    for rr in detect_result_roots(datainf_root):
        files.extend((rr / "loss_theory" / "by_combo").glob("**/sample_stats.csv"))
    files = sorted(set(files))
    if not files:
        return None

    rows: List[Dict[str, object]] = []
    for p in files:
        try:
            d = pd.read_csv(p, usecols=lambda c: c in {"train_dataset", "method", "epoch", "task", "Lbar_i"})
        except Exception:
            continue
        if d.empty:
            continue

        train_col = str(d["train_dataset"].iloc[0]).strip().lower() if "train_dataset" in d.columns else ""
        method_col = str(d["method"].iloc[0]).strip().lower() if "method" in d.columns else ""
        epoch_col = str(d["epoch"].iloc[0]).strip().lower() if "epoch" in d.columns else ""
        task_col = str(d["task"].iloc[0]).strip().lower() if "task" in d.columns else ""

        train_p, method_p, epoch_p, task_p = parse_combo_from_path(p)
        train = train_col or (train_p or "")
        method = method_col or (method_p or "")
        epoch = epoch_col or (epoch_p or "")
        task = task_col or (task_p or "")

        if not train or not method or not epoch or not task:
            continue
        if method not in ("sft", "sdft"):
            continue

        if "Lbar_i" in d.columns:
            vals = pd.to_numeric(d["Lbar_i"], errors="coerce").dropna()
        else:
            vals = pd.Series(dtype=float)
        if vals.empty:
            continue

        rows.append(
            {
                "train_dataset": train,
                "method": method,
                "epoch": epoch,
                "task": task,
                "loss_mean": float(vals.mean()),
            }
        )

    if not rows:
        return None

    agg = pd.DataFrame(rows)
    agg = (
        agg.groupby(["train_dataset", "method", "epoch", "task"], as_index=False)["loss_mean"]
        .mean()
    )
    wide = agg.pivot_table(
        index=["train_dataset", "method", "epoch"],
        columns="task",
        values="loss_mean",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    return wide


def locate_wide_table(datainf_root: Path, matrix_csv: str, source_mode: str, loss_eval_root: str) -> Tuple[pd.DataFrame, Path]:
    # 1) force from loss_eval rows
    if source_mode == "loss_eval_rows":
        got = build_wide_table_from_loss_eval_rows(datainf_root, loss_eval_root)
        if got is not None:
            return got
        raise FileNotFoundError(
            f"source_mode=loss_eval_rows but cannot build from loss_rows*.json under {loss_eval_root or '<auto loss_eval>'}"
        )

    # 2) explicit matrix csv/json
    if matrix_csv.strip():
        p = Path(matrix_csv).expanduser().resolve()
        df = try_load_from_csv(p)
        if df is not None and has_non_nan_value(df):
            return df, p
        print(f"[warn] matrix_csv unusable (missing/invalid/all-NaN), fallback to auto source: {p}")

    # 3) try loss_eval rows before legacy tables (for better alignment with user's requested source)
    got = build_wide_table_from_loss_eval_rows(datainf_root, loss_eval_root)
    if got is not None:
        return got

    # 4) fallback to legacy wide tables
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
        if not has_non_nan_value(df):
            continue
        return df, p

    # 5) final fallback from loss_theory sample_stats
    df_sample = build_wide_table_from_sample_stats(datainf_root)
    if df_sample is not None and has_non_nan_value(df_sample):
        return df_sample, (datainf_root / "results" / "loss_theory" / "by_combo" / "**/sample_stats.csv")

    raise FileNotFoundError(
        "Cannot find valid non-NaN loss table. Tried: explicit matrix_csv, "
        "loss_eval/loss_eval_test2 sft__sdft tables, and loss_theory/by_combo sample_stats.csv."
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
    y_min: Optional[float],
    add_delta_panel: bool,
    sort_by_delta: bool,
) -> Path:
    # Prepare per-train values first for optional sorting and shared y-limits.
    recs: List[Dict[str, object]] = []
    fallback_notes: List[str] = []
    all_main_vals: List[float] = []
    all_delta_vals: List[float] = []

    for train in train_datasets:
        train_l = train.lower()
        sft_vals: List[float] = []
        sdft_vals: List[float] = []
        for ep in EPOCHS_TARGET:
            v_sft, ep_used_sft = pick_value(lookup, train_l, "sft", ep, task)
            v_sdft, ep_used_sdft = pick_value(lookup, train_l, "sdft", ep, task)
            sft_vals.append(np.nan if v_sft is None else float(v_sft))
            sdft_vals.append(np.nan if v_sdft is None else float(v_sdft))
            if ep == "epoch_5":
                if ep_used_sft != "epoch_5":
                    fallback_notes.append(f"{train}:sft:{task} uses {ep_used_sft} as E5")
                if ep_used_sdft != "epoch_5":
                    fallback_notes.append(f"{train}:sdft:{task} uses {ep_used_sdft} as E5")

        main_vals = [
            sft_vals[0],
            sdft_vals[0],
            sft_vals[1],
            sdft_vals[1],
            sft_vals[2],
            sdft_vals[2],
        ]
        deltas = [
            (sdft_vals[0] - sft_vals[0]) if np.isfinite(sdft_vals[0]) and np.isfinite(sft_vals[0]) else np.nan,
            (sdft_vals[1] - sft_vals[1]) if np.isfinite(sdft_vals[1]) and np.isfinite(sft_vals[1]) else np.nan,
            (sdft_vals[2] - sft_vals[2]) if np.isfinite(sdft_vals[2]) and np.isfinite(sft_vals[2]) else np.nan,
        ]
        finite_delta = [d for d in deltas if np.isfinite(d)]
        delta_score = float(np.mean(finite_delta)) if finite_delta else np.inf

        recs.append(
            {
                "train": train,
                "main_vals": main_vals,
                "deltas": deltas,
                "delta_score": delta_score,
            }
        )
        all_main_vals.extend([v for v in main_vals if np.isfinite(v)])
        all_delta_vals.extend([d for d in deltas if np.isfinite(d)])

    if sort_by_delta:
        recs.sort(key=lambda r: float(r["delta_score"]))

    fig = plt.figure(figsize=(21, 10), constrained_layout=True)
    outer = fig.add_gridspec(2, 4, wspace=0.28, hspace=0.30)

    x_main = np.arange(6)
    x_main_labels = [
        "Epoch 0\nSFT",
        "Epoch 0\nSDFT",
        "Epoch 1\nSFT",
        "Epoch 1\nSDFT",
        "Epoch 5\nSFT",
        "Epoch 5\nSDFT",
    ]
    x_delta = np.arange(3)
    x_delta_labels = ["Epoch 0", "Epoch 1", "Epoch 5"]
    color_sft = "#4C78A8"
    color_sdft = "#F58518"
    color_delta = "#6B6B6B"

    top_axes: List[plt.Axes] = []
    bot_axes: List[plt.Axes] = []

    for idx in range(8):
        r = idx // 4
        c = idx % 4
        slot = outer[r, c]
        if idx >= len(recs):
            ax_empty = fig.add_subplot(slot)
            ax_empty.axis("off")
            continue

        rec = recs[idx]
        train = str(rec["train"])
        main_vals = np.array(rec["main_vals"], dtype=float)
        deltas = np.array(rec["deltas"], dtype=float)

        if add_delta_panel:
            inner = slot.subgridspec(2, 1, height_ratios=[3.8, 1.4], hspace=0.02)
            ax_top = fig.add_subplot(inner[0])
            ax_bot = fig.add_subplot(inner[1])
        else:
            ax_top = fig.add_subplot(slot)
            ax_bot = None

        bar_colors = [color_sft, color_sdft, color_sft, color_sdft, color_sft, color_sdft]
        ax_top.bar(x_main, main_vals, color=bar_colors, width=0.36, edgecolor="black", linewidth=0.35)
        ax_top.set_title(train, fontsize=11)
        ax_top.grid(axis="y", linestyle="--", alpha=0.25)
        ax_top.tick_params(axis="y", labelsize=8)

        # Top-axis x labels: keep full wording (no short abbreviations).
        ax_top.set_xticks(x_main)
        ax_top.set_xticklabels(x_main_labels, fontsize=7, rotation=0)
        ax_top.set_ylabel("Loss", fontsize=9)

        ymax_local = np.nanmax(main_vals) if np.isfinite(np.nanmax(main_vals)) else 1.0
        for xi, yi in zip(x_main, main_vals):
            if np.isnan(yi):
                ax_top.text(xi, ymax_local * 0.03, "NA", ha="center", va="bottom", fontsize=7, color="red", rotation=90)
            else:
                ax_top.text(xi, yi + max(0.01, ymax_local * 0.012), f"{yi:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

        # Epoch-wise delta annotations above each pair in top panel.
        for g in range(3):
            v1 = main_vals[2 * g]
            v2 = main_vals[2 * g + 1]
            d = deltas[g]
            if np.isfinite(v1) and np.isfinite(v2) and np.isfinite(d):
                y_pair = max(v1, v2)
                ax_top.text(
                    2 * g + 0.5,
                    y_pair + max(0.02, ymax_local * 0.04),
                    f"\u0394={d:+.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#333333",
                )

        if add_delta_panel and ax_bot is not None:
            ax_bot.axhline(0.0, color="#333333", linewidth=0.7)
            ax_bot.bar(x_delta, deltas, color=color_delta, width=0.52, edgecolor="black", linewidth=0.35)
            ax_bot.set_xticks(x_delta)
            ax_bot.set_xticklabels(x_delta_labels, fontsize=7)
            ax_bot.tick_params(axis="y", labelsize=7)
            ax_bot.grid(axis="y", linestyle="--", alpha=0.2)
            ax_bot.set_ylabel("\u0394", fontsize=8, rotation=0, labelpad=8)
            for xi, di in zip(x_delta, deltas):
                if np.isfinite(di):
                    ax_bot.text(
                        xi,
                        di + (0.01 if di >= 0 else -0.01),
                        f"{di:+.3f}",
                        ha="center",
                        va="bottom" if di >= 0 else "top",
                        fontsize=7,
                    )

            bot_axes.append(ax_bot)

        top_axes.append(ax_top)

    # Shared y-limits (top panels).
    if all_main_vals:
        global_max = max(all_main_vals)
        y0 = 0.0 if y_min is None else float(y_min)
        y1 = max(global_max * 1.16, y0 + 0.2)
        for ax in top_axes:
            ax.set_ylim(y0, y1)

    # Shared y-limits (delta panels).
    if add_delta_panel and bot_axes:
        if all_delta_vals:
            d_abs = max(abs(min(all_delta_vals)), abs(max(all_delta_vals)))
            d_abs = max(d_abs * 1.25, 0.02)
            for ax in bot_axes:
                ax.set_ylim(-d_abs, d_abs)

    # Global legend with full method names.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_sft, ec="black", lw=0.35),
        plt.Rectangle((0, 0), 1, 1, color=color_sdft, ec="black", lw=0.35),
        plt.Rectangle((0, 0), 1, 1, color=color_delta, ec="black", lw=0.35),
    ]
    legend_labels = [
        "SFT (Supervised Fine-Tuning)",
        "SDFT (Self-Distillation Fine-Tuning)",
        "\u0394 = SDFT - SFT",
    ]
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3, fontsize=10, frameon=False)

    title_suffix = " (sorted by avg \u0394 ascending)" if sort_by_delta else ""
    fig.suptitle(
        f"Loss Comparison by Train Dataset | Test Task: {task}{title_suffix}",
        fontsize=14,
        y=0.995,
    )

    if fallback_notes:
        note = " ; ".join(sorted(set(fallback_notes))[:5])
        fig.text(
            0.01,
            0.006,
            f"Note: Epoch-5 fallback detected (showing first few): {note}",
            fontsize=8,
            color="#555555",
        )

    out_path = out_dir / f"loss_bar_7groups_{task}.{fmt}"
    fig.savefig(out_path, dpi=220 if fmt.lower() == "png" else None, bbox_inches="tight")
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
    parser.add_argument("--y_min", type=float, default=None, help="Optional fixed y-axis lower bound (e.g., 1.0).")
    parser.add_argument(
        "--add_delta_panel",
        type=int,
        default=1,
        help="1: add delta (SDFT-SFT) mini-panel below each train subplot; 0: disable.",
    )
    parser.add_argument(
        "--sort_train_by_delta",
        type=int,
        default=0,
        help="1: sort 7 train datasets by average delta ascending (more negative first).",
    )
    parser.add_argument(
        "--source_mode",
        type=str,
        default="auto",
        choices=["auto", "loss_eval_rows"],
        help="auto: matrix_csv->loss_eval_rows->legacy tables->sample_stats; "
        "loss_eval_rows: force only loss_eval rows source.",
    )
    parser.add_argument(
        "--loss_eval_root",
        type=str,
        default="",
        help="Optional explicit loss_eval root directory, e.g. /.../DataInf/results/loss_eval",
    )
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

    df, src = locate_wide_table(datainf_root, args.matrix_csv, args.source_mode, args.loss_eval_root)
    lookup = build_lookup(df, tasks)

    print(f"[source] {src}")
    for task in tasks:
        p = make_task_figure(
            lookup,
            task,
            train_datasets,
            output_dir,
            args.format,
            args.y_min,
            add_delta_panel=bool(int(args.add_delta_panel)),
            sort_by_delta=bool(int(args.sort_train_by_delta)),
        )
        print(str(p.resolve()))


if __name__ == "__main__":
    main()
