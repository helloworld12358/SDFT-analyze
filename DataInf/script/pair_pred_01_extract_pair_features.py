#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pair-level predictor - Step 01

Extract pair-wise features for (train_dataset=A, epoch=e, test_task=B):
- Cent
- Load
- Self
- DeltaCent / DeltaLoad / DeltaSelf
- true_perf_diff (SDFT - SFT)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAINF_ROOT_DEFAULT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(DATAINF_ROOT_DEFAULT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gram_scheme_a_utils import (  # type: ignore  # noqa: E402
    DEFAULT_EPOCHS,
    DEFAULT_TASKS,
    DEFAULT_TRAIN_DATASETS,
    detect_datainf_root,
    load_existing_ownh_from_analysis,
    resolve_existing_result_roots,
    resolve_result_root,
    write_rows_csv,
    write_rows_txt,
    write_unavailable_note,
)


TASK_TO_METRIC_KEYS: Dict[str, List[str]] = {
    "alpaca_eval": ["metric_alpaca_eval", "metric_accuracy_generic"],
    "gsm8k": ["metric_gsm8k", "metric_accuracy_generic"],
    "humaneval": ["metric_humaneval", "metric_humaneval_passk", "metric_accuracy_generic"],
    "multiarith": ["metric_multiarith", "metric_accuracy_generic"],
    "openfunction": ["metric_openfunction", "metric_accuracy_generic"],
}


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_list(path: str) -> List[Dict[str, object]]:
    if not os.path.isfile(path):
        return []
    obj = _load_json(path)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


def _to_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        s = str(v).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _sign_of(v: Optional[float]) -> Optional[int]:
    if v is None:
        return None
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def _is_5x5(mat: object) -> bool:
    return isinstance(mat, np.ndarray) and mat.shape == (5, 5)


def _cent(C: np.ndarray, b: int) -> float:
    row = np.asarray(C[b, :], dtype=np.float64)
    others = np.delete(row, b)
    return float(np.mean(others))


def _load(C: np.ndarray, b: int) -> float:
    vals, vecs = np.linalg.eigh(np.asarray(C, dtype=np.float64))
    idx = int(np.argmax(np.real(vals)))
    u1 = np.real(vecs[:, idx])
    return float(u1[b] ** 2)


def _self(T: np.ndarray, b: int) -> float:
    return float(np.asarray(T, dtype=np.float64)[b, b])


def _perf_index(perf_rows: Sequence[Dict[str, object]]) -> Dict[Tuple[str, str, str], Dict[str, object]]:
    idx: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for r in perf_rows:
        td = str(r.get("train_dataset", ""))
        ep = str(r.get("epoch", ""))
        md = str(r.get("method", ""))
        if td and ep and md:
            idx[(td, ep, md)] = r
    return idx


def _task_metric(row: Optional[Dict[str, object]], task: str) -> Tuple[Optional[float], Optional[str]]:
    if row is None:
        return None, None
    for k in TASK_TO_METRIC_KEYS.get(task, []):
        v = _to_float(row.get(k))
        if v is not None:
            return v, k
    return None, None


def _pick_list(arg_value: str, default_list: Sequence[str]) -> List[str]:
    if not arg_value.strip():
        return list(default_list)
    out = [x.strip() for x in arg_value.split(",") if x.strip()]
    return out if out else list(default_list)


def main() -> None:
    p = argparse.ArgumentParser(description="Pair-level predictor step01 feature extractor")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default=None, help="Default: <result_root>/pair_pred")
    p.add_argument("--extra_result_roots", type=str, default="")
    p.add_argument("--train_dataset", type=str, default="")
    p.add_argument("--all_train_datasets", action="store_true")
    p.add_argument("--epoch", type=str, default="")
    p.add_argument("--all_epochs", action="store_true")
    p.add_argument("--task_names", type=str, default=",".join(DEFAULT_TASKS))
    p.add_argument("--inventory_json", type=str, default="", help="Optional inventory_mapping.json path from step00")
    p.add_argument("--perf_json", type=str, default="", help="Optional perf_standardized.json path from step00")
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    output_root = args.output_root or os.path.join(result_root, "pair_pred")
    os.makedirs(output_root, exist_ok=True)

    if args.all_train_datasets:
        train_datasets = list(DEFAULT_TRAIN_DATASETS)
    else:
        train_datasets = _pick_list(args.train_dataset, DEFAULT_TRAIN_DATASETS)
    if args.all_epochs:
        epochs = list(DEFAULT_EPOCHS)
    else:
        epochs = _pick_list(args.epoch, DEFAULT_EPOCHS)
    task_names = _pick_list(args.task_names, DEFAULT_TASKS)

    perf_json = args.perf_json or os.path.join(output_root, "perf", "perf_standardized.json")
    perf_rows = _load_json_list(perf_json)
    perf_idx = _perf_index(perf_rows)

    extra_roots: List[str] = [x.strip() for x in args.extra_result_roots.split(",") if x.strip()]
    if args.inventory_json and os.path.isfile(args.inventory_json):
        inv = _load_json(args.inventory_json)
        roots = inv.get("result_roots_scanned", [])
        if isinstance(roots, list):
            for r in roots:
                rs = str(r).strip()
                if rs and rs not in extra_roots:
                    extra_roots.append(rs)
    result_roots = resolve_existing_result_roots(datainf_root, explicit_roots=extra_roots)

    rows: List[Dict[str, object]] = []
    unavailable_dir = os.path.join(output_root, "unavailable")
    os.makedirs(unavailable_dir, exist_ok=True)
    unavailable_rows: List[Dict[str, object]] = []

    if len(task_names) != 5:
        reason = write_unavailable_note(
            os.path.join(unavailable_dir, "unavailable_task_order.json"),
            reason=f"task_names must be 5 tasks, got {len(task_names)}",
            context={"task_names": task_names},
        )
        print(os.path.abspath(reason))
        return

    task_to_idx = {t: i for i, t in enumerate(task_names)}

    for ds in train_datasets:
        for ep in epochs:
            existing = load_existing_ownh_from_analysis(result_roots, ds, ep)
            T_sft = existing.get("sft", {}).get("T")
            C_sft = existing.get("sft", {}).get("C")
            T_sdft = existing.get("sdft", {}).get("T")
            C_sdft = existing.get("sdft", {}).get("C")

            if not (_is_5x5(T_sft) and _is_5x5(C_sft) and _is_5x5(T_sdft) and _is_5x5(C_sdft)):
                reason = write_unavailable_note(
                    os.path.join(unavailable_dir, f"unavailable_{ds}_{ep}_matrix.json"),
                    reason="missing or non-5x5 matrix for sft/sdft",
                    context={
                        "train_dataset": ds,
                        "epoch": ep,
                        "result_roots_scanned": result_roots,
                        "sft_analysis_log_path": existing.get("sft", {}).get("analysis_log_path"),
                        "sdft_analysis_log_path": existing.get("sdft", {}).get("analysis_log_path"),
                        "sft_analysis_corr_path": existing.get("sft", {}).get("analysis_corr_path"),
                        "sdft_analysis_corr_path": existing.get("sdft", {}).get("analysis_corr_path"),
                    },
                )
                unavailable_rows.append({"train_dataset": ds, "epoch": ep, "reason_file": os.path.abspath(reason)})
                continue

            sft_perf = perf_idx.get((ds, ep, "sft"))
            sdft_perf = perf_idx.get((ds, ep, "sdft"))
            if sft_perf is None or sdft_perf is None:
                reason = write_unavailable_note(
                    os.path.join(unavailable_dir, f"unavailable_{ds}_{ep}_perf.json"),
                    reason="missing standardized performance row for sft/sdft",
                    context={
                        "train_dataset": ds,
                        "epoch": ep,
                        "perf_json": os.path.abspath(perf_json),
                    },
                )
                unavailable_rows.append({"train_dataset": ds, "epoch": ep, "reason_file": os.path.abspath(reason)})
                continue

            for task in task_names:
                b = task_to_idx[task]
                cent_sft = _cent(C_sft, b)
                cent_sdft = _cent(C_sdft, b)
                load_sft = _load(C_sft, b)
                load_sdft = _load(C_sdft, b)
                self_sft = _self(T_sft, b)
                self_sdft = _self(T_sdft, b)

                true_sft, true_sft_key = _task_metric(sft_perf, task)
                true_sdft, true_sdft_key = _task_metric(sdft_perf, task)
                true_diff = None
                if true_sft is not None and true_sdft is not None:
                    true_diff = float(true_sdft - true_sft)  # SDFT - SFT

                rows.append(
                    {
                        "train_dataset": ds,
                        "epoch": ep,
                        "test_task": task,
                        "task_index": b,
                        "Cent_sft": cent_sft,
                        "Cent_sdft": cent_sdft,
                        "DeltaCent": cent_sdft - cent_sft,
                        "Load_sft": load_sft,
                        "Load_sdft": load_sdft,
                        "DeltaLoad": load_sdft - load_sft,
                        "Self_sft": self_sft,
                        "Self_sdft": self_sdft,
                        "DeltaSelf": self_sdft - self_sft,
                        "true_perf_sft": true_sft,
                        "true_perf_sdft": true_sdft,
                        "true_perf_diff": true_diff,
                        "true_sign": _sign_of(true_diff),
                        "true_metric_key_sft": true_sft_key,
                        "true_metric_key_sdft": true_sdft_key,
                        "analysis_log_sft": existing.get("sft", {}).get("analysis_log_path"),
                        "analysis_log_sdft": existing.get("sdft", {}).get("analysis_log_path"),
                        "analysis_corr_sft": existing.get("sft", {}).get("analysis_corr_path"),
                        "analysis_corr_sdft": existing.get("sdft", {}).get("analysis_corr_path"),
                    }
                )

    all_csv = os.path.join(output_root, "pair_features_all.csv")
    all_json = os.path.join(output_root, "pair_features_all.json")
    all_txt = os.path.join(output_root, "pair_features_all.txt")
    write_rows_csv(all_csv, rows)
    write_rows_txt(all_txt, rows, max_cols=20)
    with open(all_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    for ep in epochs:
        ep_rows = [r for r in rows if str(r.get("epoch")) == ep]
        ep_csv = os.path.join(output_root, f"pair_features_{ep}.csv")
        ep_json = os.path.join(output_root, f"pair_features_{ep}.json")
        write_rows_csv(ep_csv, ep_rows)
        with open(ep_json, "w", encoding="utf-8") as f:
            json.dump(ep_rows, f, ensure_ascii=False, indent=2)

    unavailable_summary_json = os.path.join(output_root, "unavailable_pair_features.json")
    with open(unavailable_summary_json, "w", encoding="utf-8") as f:
        json.dump(unavailable_rows, f, ensure_ascii=False, indent=2)

    print(os.path.abspath(all_csv))
    print(os.path.abspath(all_json))
    print(os.path.abspath(unavailable_summary_json))


if __name__ == "__main__":
    main()
