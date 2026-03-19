#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pair-level predictor - Step 02

Predict:
pred_score = alpha * DeltaCent + beta * DeltaLoad + gamma * DeltaSelf

Supports:
- manual weights
- simple grid search
- leave-one-train-dataset-out (LODO) grid fitting
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAINF_ROOT_DEFAULT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(DATAINF_ROOT_DEFAULT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gram_scheme_a_utils import (  # type: ignore  # noqa: E402
    detect_datainf_root,
    resolve_result_root,
    write_rows_csv,
    write_rows_txt,
    write_unavailable_note,
)


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
        x = float(s)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def _rankdata_average_ties(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> Tuple[Optional[float], Optional[str]]:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return None, "n<2"
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None, "constant_series"
    val = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(val):
        return None, "non_finite"
    return val, None


def _spearman(x: Sequence[float], y: Sequence[float]) -> Tuple[Optional[float], Optional[str]]:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return None, "n<2"
    ra = _rankdata_average_ties(a.tolist())
    rb = _rankdata_average_ties(b.tolist())
    return _pearson(ra.tolist(), rb.tolist())


def _sign(v: Optional[float]) -> Optional[int]:
    if v is None:
        return None
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def _valid_feature_row(r: Dict[str, object]) -> bool:
    return (
        _to_float(r.get("DeltaCent")) is not None
        and _to_float(r.get("DeltaLoad")) is not None
        and _to_float(r.get("DeltaSelf")) is not None
        and _to_float(r.get("true_perf_diff")) is not None
    )


def _predict_score(r: Dict[str, object], alpha: float, beta: float, gamma: float) -> Optional[float]:
    dcent = _to_float(r.get("DeltaCent"))
    dload = _to_float(r.get("DeltaLoad"))
    dself = _to_float(r.get("DeltaSelf"))
    if dcent is None or dload is None or dself is None:
        return None
    return float(alpha * dcent + beta * dload + gamma * dself)


def _sign_accuracy(rows: Sequence[Dict[str, object]]) -> Optional[float]:
    valid = []
    for r in rows:
        ps = _to_float(r.get("pred_score"))
        ys = _to_float(r.get("true_perf_diff"))
        if ps is None or ys is None:
            continue
        valid.append((ps, ys))
    if not valid:
        return None
    hit = 0
    for ps, ys in valid:
        if _sign(ps) == _sign(ys):
            hit += 1
    return float(hit / len(valid))


def _corr_metrics(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    x: List[float] = []
    y: List[float] = []
    for r in rows:
        ps = _to_float(r.get("pred_score"))
        ys = _to_float(r.get("true_perf_diff"))
        if ps is None or ys is None:
            continue
        x.append(ps)
        y.append(ys)
    p, p_reason = _pearson(x, y)
    s, s_reason = _spearman(x, y)
    return {
        "n_for_corr": len(x),
        "pearson": p,
        "pearson_unavailable_reason": p_reason,
        "spearman": s,
        "spearman_unavailable_reason": s_reason,
    }


def _topk_metrics(rows: Sequence[Dict[str, object]], ks: Sequence[int]) -> Dict[str, object]:
    valid = []
    for r in rows:
        ps = _to_float(r.get("pred_score"))
        ys = _to_float(r.get("true_perf_diff"))
        if ps is None or ys is None:
            continue
        valid.append((ps, ys))
    out: Dict[str, object] = {"n_for_topk": len(valid)}
    if not valid:
        return out
    pred_order = np.argsort([-v[0] for v in valid]).tolist()
    true_order = np.argsort([-v[1] for v in valid]).tolist()
    for k in ks:
        kk = min(k, len(valid))
        if kk <= 0:
            continue
        pred_top = pred_order[:kk]
        true_top = true_order[:kk]
        overlap = len(set(pred_top) & set(true_top)) / float(kk)
        pred_pos_precision = sum(1 for i in pred_top if valid[i][1] > 0) / float(kk)
        out[f"top{kk}_overlap"] = overlap
        out[f"top{kk}_pred_pos_precision"] = pred_pos_precision
    return out


def _evaluate_rows(
    rows: Sequence[Dict[str, object]],
    alpha: float,
    beta: float,
    gamma: float,
    mode_label: str,
    fold_label: str = "",
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in rows:
        rr = dict(r)
        ps = _predict_score(r, alpha, beta, gamma)
        ys = _to_float(r.get("true_perf_diff"))
        rr["pred_mode"] = mode_label
        rr["pred_alpha"] = alpha
        rr["pred_beta"] = beta
        rr["pred_gamma"] = gamma
        rr["pred_fold"] = fold_label
        rr["pred_score"] = ps
        rr["pred_sign"] = _sign(ps)
        rr["pred_correct_sign"] = None if (ps is None or ys is None) else int(_sign(ps) == _sign(ys))
        out.append(rr)
    return out


def _objective(pred_rows: Sequence[Dict[str, object]]) -> Tuple[float, float]:
    acc = _sign_accuracy(pred_rows)
    corr = _corr_metrics(pred_rows).get("pearson")
    acc_v = -1.0 if acc is None else float(acc)
    corr_v = -2.0 if corr is None else float(corr)
    return acc_v, corr_v


def _grid_fit(train_rows: Sequence[Dict[str, object]], grid_values: Sequence[float]) -> Tuple[float, float, float, Dict[str, object]]:
    best = None
    best_stats: Dict[str, object] = {}
    for a, b, g in product(grid_values, grid_values, grid_values):
        if abs(a) + abs(b) + abs(g) == 0:
            continue
        pred = _evaluate_rows(train_rows, a, b, g, mode_label="grid_fit")
        score = _objective(pred)
        # tie-break by smaller L1 norm to keep simpler weights
        l1 = abs(a) + abs(b) + abs(g)
        key = (score[0], score[1], -l1)
        if best is None or key > best[0]:
            best = (key, (a, b, g))
            best_stats = {
                "train_sign_accuracy": _sign_accuracy(pred),
                **_corr_metrics(pred),
            }
    if best is None:
        return 0.0, 0.0, 0.0, {"train_sign_accuracy": None, "n_for_corr": 0}
    a, b, g = best[1]
    return float(a), float(b), float(g), best_stats


def _parse_grid_values(s: str) -> List[float]:
    vals = []
    for x in s.split(","):
        t = x.strip()
        if not t:
            continue
        vals.append(float(t))
    return vals if vals else [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


def main() -> None:
    p = argparse.ArgumentParser(description="Pair-level predictor step02 scorer")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default=None, help="Default: <result_root>/pair_pred")
    # Backward-compatible alias: --epoch maps to feature_epoch if --feature_epoch is not given.
    p.add_argument("--epoch", type=str, default="epoch_1")
    p.add_argument("--feature_epoch", type=str, default="")
    p.add_argument("--label_epoch", type=str, default="", help="If empty, use same as feature_epoch")
    p.add_argument("--features_json", type=str, default="", help="Default: <output_root>/pair_features_all.json")
    p.add_argument("--mode", type=str, default="manual", choices=["manual", "grid", "lodo"])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--grid_values", type=str, default="-2,-1,-0.5,0,0.5,1,2")
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    output_root = args.output_root or os.path.join(result_root, "pair_pred")
    os.makedirs(output_root, exist_ok=True)

    feature_epoch = args.feature_epoch.strip() or str(args.epoch).strip() or "epoch_1"
    label_epoch = args.label_epoch.strip() or feature_epoch
    epoch_tag = f"feat_{feature_epoch}__label_{label_epoch}"

    features_json = args.features_json or os.path.join(output_root, "pair_features_all.json")
    all_rows = _load_json_list(features_json)
    rows = []
    for r in all_rows:
        fe = str(r.get("feature_epoch", r.get("epoch", "")))
        le = str(r.get("label_epoch", r.get("true_perf_epoch", fe)))
        if fe == feature_epoch and le == label_epoch:
            rows.append(r)

    unavailable_dir = os.path.join(output_root, "unavailable")
    os.makedirs(unavailable_dir, exist_ok=True)

    if not rows:
        reason = write_unavailable_note(
            os.path.join(unavailable_dir, f"unavailable_predictor_{epoch_tag}.json"),
            reason="no pair feature rows for target epoch",
            context={
                "features_json": os.path.abspath(features_json),
                "feature_epoch": feature_epoch,
                "label_epoch": label_epoch,
            },
        )
        print(os.path.abspath(reason))
        return

    valid_rows = [r for r in rows if _valid_feature_row(r)]
    if len(valid_rows) < 2:
        reason = write_unavailable_note(
            os.path.join(unavailable_dir, f"unavailable_predictor_{epoch_tag}.json"),
            reason="insufficient valid rows for predictor",
            context={
                "epoch_rows": len(rows),
                "valid_rows": len(valid_rows),
                "feature_epoch": feature_epoch,
                "label_epoch": label_epoch,
            },
        )
        print(os.path.abspath(reason))
        return

    pred_rows: List[Dict[str, object]] = []
    fit_rows: List[Dict[str, object]] = []
    fit_meta: Dict[str, object] = {}

    if args.mode == "manual":
        pred_rows = _evaluate_rows(rows, args.alpha, args.beta, args.gamma, mode_label="manual")
        fit_meta = {"mode": "manual", "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma}
    elif args.mode == "grid":
        grid_values = _parse_grid_values(args.grid_values)
        a, b, g, train_stats = _grid_fit(valid_rows, grid_values)
        pred_rows = _evaluate_rows(rows, a, b, g, mode_label="grid")
        fit_meta = {
            "mode": "grid",
            "grid_values": grid_values,
            "best_alpha": a,
            "best_beta": b,
            "best_gamma": g,
            "train_stats": train_stats,
        }
    else:  # lodo
        grid_values = _parse_grid_values(args.grid_values)
        by_ds: Dict[str, List[Dict[str, object]]] = {}
        for r in rows:
            by_ds.setdefault(str(r.get("train_dataset")), []).append(r)
        for holdout_ds, hold_rows in sorted(by_ds.items()):
            train = [r for r in rows if str(r.get("train_dataset")) != holdout_ds and _valid_feature_row(r)]
            if len(train) < 2:
                # fallback manual weights if insufficient training rows
                a, b, g = args.alpha, args.beta, args.gamma
                train_stats = {"fallback": True}
            else:
                a, b, g, train_stats = _grid_fit(train, grid_values)
            fit_rows.append(
                {
                    "feature_epoch": feature_epoch,
                    "label_epoch": label_epoch,
                    "holdout_train_dataset": holdout_ds,
                    "alpha": a,
                    "beta": b,
                    "gamma": g,
                    "train_rows": len(train),
                    "train_sign_accuracy": train_stats.get("train_sign_accuracy"),
                    "train_pearson": train_stats.get("pearson"),
                    "train_spearman": train_stats.get("spearman"),
                }
            )
            pred_rows.extend(_evaluate_rows(hold_rows, a, b, g, mode_label="lodo", fold_label=holdout_ds))
        fit_meta = {"mode": "lodo", "grid_values": grid_values, "fold_count": len(fit_rows)}

    # Metrics on rows with both pred and true
    sign_acc = _sign_accuracy(pred_rows)
    corr = _corr_metrics(pred_rows)
    topk = _topk_metrics(pred_rows, ks=[3, 5, 10])

    by_ds_acc: List[Dict[str, object]] = []
    ds_groups: Dict[str, List[Dict[str, object]]] = {}
    for r in pred_rows:
        ds_groups.setdefault(str(r.get("train_dataset")), []).append(r)
    for ds, rr in sorted(ds_groups.items()):
        by_ds_acc.append(
            {
                "feature_epoch": feature_epoch,
                "label_epoch": label_epoch,
                "train_dataset": ds,
                "n_rows": len(rr),
                "sign_accuracy": _sign_accuracy(rr),
                "pearson": _corr_metrics(rr).get("pearson"),
                "spearman": _corr_metrics(rr).get("spearman"),
            }
        )

    score_csv = os.path.join(output_root, f"pair_pred_scores_{epoch_tag}.csv")
    score_json = os.path.join(output_root, f"pair_pred_scores_{epoch_tag}.json")
    score_txt = os.path.join(output_root, f"pair_pred_scores_{epoch_tag}.txt")
    write_rows_csv(score_csv, pred_rows)
    write_rows_txt(score_txt, pred_rows, max_cols=22)
    with open(score_json, "w", encoding="utf-8") as f:
        json.dump(pred_rows, f, ensure_ascii=False, indent=2)

    fit_csv = os.path.join(output_root, f"pair_pred_fit_{epoch_tag}.csv")
    if fit_rows:
        write_rows_csv(fit_csv, fit_rows)

    byds_csv = os.path.join(output_root, f"pair_pred_by_dataset_{epoch_tag}.csv")
    write_rows_csv(byds_csv, by_ds_acc)

    summary = {
        "epoch": feature_epoch,  # backward-compatible
        "feature_epoch": feature_epoch,
        "label_epoch": label_epoch,
        "mode": args.mode,
        "fit_meta": fit_meta,
        "n_rows_total": len(rows),
        "n_rows_valid_feature_and_label": len(valid_rows),
        "n_rows_scored": len(pred_rows),
        "sign_accuracy": sign_acc,
        **corr,
        **topk,
        "outputs": {
            "score_csv": os.path.abspath(score_csv),
            "score_json": os.path.abspath(score_json),
            "score_txt": os.path.abspath(score_txt),
            "fit_csv": os.path.abspath(fit_csv) if fit_rows else None,
            "by_dataset_csv": os.path.abspath(byds_csv),
        },
    }
    summary_json = os.path.join(output_root, f"pair_pred_summary_{epoch_tag}.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(os.path.abspath(score_csv))
    print(os.path.abspath(summary_json))


if __name__ == "__main__":
    main()
