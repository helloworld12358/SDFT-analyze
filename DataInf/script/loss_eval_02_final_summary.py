#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge per-dataset loss rows into a final summary:
- one TXT containing 7 tables, each table is 3x5 (epochs x test tasks)
- CSV/JSON summaries for downstream plotting/analysis
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

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
    write_rows_csv,
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


def save_json(path: str, obj: object) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


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


def fmt_loss(v: Optional[float]) -> str:
    if v is None:
        return "NA"
    return f"{v:.6f}"


def collect_rows(output_root: str, methods: Sequence[str]) -> Tuple[List[Dict[str, object]], List[str]]:
    files: List[str] = []
    for p in sorted(glob.glob(os.path.join(output_root, "loss_rows_all_*.json"))):
        files.append(p)
    for p in sorted(glob.glob(os.path.join(output_root, "by_train_dataset", "*", "loss_rows_*.json"))):
        if p not in files:
            files.append(p)

    rows: List[Dict[str, object]] = []
    seen = set()
    for p in files:
        for r in load_rows_list(p):
            m = str(r.get("method", ""))
            if m in methods:
                key = (
                    str(r.get("train_dataset", "")),
                    str(r.get("method", "")),
                    str(r.get("epoch", "")),
                    str(r.get("test_task", "")),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(r)
    return rows, files


def render_table(
    train_dataset: str,
    method: str,
    rows: Sequence[Dict[str, object]],
    epochs: Sequence[str],
    tasks: Sequence[str],
) -> str:
    lookup: Dict[Tuple[str, str], Optional[float]] = {}
    for r in rows:
        ep = str(r.get("epoch", ""))
        task = str(r.get("test_task", ""))
        status = str(r.get("status", ""))
        loss = to_float(r.get("loss_mean_token"))
        lookup[(ep, task)] = loss if status == "ok" else None

    first_col = "epoch\\task"
    widths: Dict[str, int] = {first_col: max(len(first_col), max(len(e) for e in epochs))}
    for t in tasks:
        widths[t] = max(len(t), 10)

    lines: List[str] = []
    lines.append(f"[train_dataset={train_dataset}] method={method}")
    lines.append("")
    head = first_col.ljust(widths[first_col]) + " | " + " | ".join(t.ljust(widths[t]) for t in tasks)
    sep = "-" * widths[first_col] + "-+-" + "-+-".join("-" * widths[t] for t in tasks)
    lines.append(head)
    lines.append(sep)
    for ep in epochs:
        vals = [fmt_loss(lookup.get((ep, t))) for t in tasks]
        line = ep.ljust(widths[first_col]) + " | " + " | ".join(vals[i].ljust(widths[tasks[i]]) for i in range(len(tasks)))
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Final summary for loss_eval tables")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default="", help="Default: <result_root>/loss_eval")
    p.add_argument("--methods", type=str, default="sft,sdft", help="comma-separated: sft,sdft")
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
    if len(tasks) != 5:
        raise ValueError(f"--tasks must contain exactly 5 tasks, got {len(tasks)}")

    methods = [m for m in split_csv_arg(args.methods, ["sft", "sdft"]) if m in ("sft", "sdft")]
    if not methods:
        raise ValueError(f"--methods invalid: {args.methods}")

    rows, source_files = collect_rows(output_root, methods)
    by_dataset: Dict[str, List[Dict[str, object]]] = {d: [] for d in train_datasets}
    for r in rows:
        d = str(r.get("train_dataset", ""))
        if d in by_dataset:
            by_dataset[d].append(r)

    wide_rows: List[Dict[str, object]] = []
    for d in train_datasets:
        lookup: Dict[Tuple[str, str, str], Optional[float]] = {}
        for r in by_dataset.get(d, []):
            method = str(r.get("method", ""))
            ep = str(r.get("epoch", ""))
            task = str(r.get("test_task", ""))
            status = str(r.get("status", ""))
            lookup[(method, ep, task)] = to_float(r.get("loss_mean_token")) if status == "ok" else None
        for method in methods:
            for ep in epochs:
                row = {
                    "train_dataset": d,
                    "method": method,
                    "epoch": ep,
                }
                for t in tasks:
                    row[t] = lookup.get((method, ep, t))
                wide_rows.append(row)

    method_tag = "__".join(methods)
    final_txt = os.path.join(output_root, f"loss_tables_7x3x5_{method_tag}.txt")
    final_csv = os.path.join(output_root, f"loss_tables_7x3x5_{method_tag}.csv")
    final_json = os.path.join(output_root, f"loss_tables_7x3x5_{method_tag}.json")

    with open(final_txt, "w", encoding="utf-8") as f:
        f.write("Loss Summary (token-level CE loss; lower is better)\n")
        f.write(f"methods={','.join(methods)}\n")
        f.write(f"epochs={','.join(epochs)}\n")
        f.write(f"tasks={','.join(tasks)}\n")
        f.write("layout=14 tables (for each train_dataset: SFT table then SDFT table)\n")
        f.write("\n")
        for d in train_datasets:
            rows_d = by_dataset.get(d, [])
            if "sft" in methods:
                f.write(render_table(d, "sft", [r for r in rows_d if str(r.get("method", "")) == "sft"], epochs, tasks))
                f.write("\n")
            if "sdft" in methods:
                f.write(render_table(d, "sdft", [r for r in rows_d if str(r.get("method", "")) == "sdft"], epochs, tasks))
            f.write("\n")

    write_rows_csv(final_csv, wide_rows)
    payload = {
        "methods": methods,
        "epochs": epochs,
        "tasks": tasks,
        "train_datasets": train_datasets,
        "n_input_rows": len(rows),
        "source_files": [os.path.abspath(p) for p in source_files],
        "outputs": {
            "txt": os.path.abspath(final_txt),
            "csv": os.path.abspath(final_csv),
        },
        "wide_rows": wide_rows,
    }
    save_json(final_json, payload)

    print(os.path.abspath(final_txt))
    print(os.path.abspath(final_csv))
    print(os.path.abspath(final_json))


if __name__ == "__main__":
    main()
