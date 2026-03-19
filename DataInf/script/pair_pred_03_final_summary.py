#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pair-level predictor - Step 03

Final aggregation:
- feature table quality (corr/sign-acc vs true diff)
- predictor summary
- unavailable list
Outputs: csv/json/md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
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
    detect_datainf_root,
    resolve_result_root,
    write_rows_csv,
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


def _sign(v: Optional[float]) -> Optional[int]:
    if v is None:
        return None
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


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


def _feature_eval(rows: Sequence[Dict[str, object]], feature_col: str) -> Dict[str, object]:
    x: List[float] = []
    y: List[float] = []
    sign_hits = 0
    sign_n = 0
    for r in rows:
        fv = _to_float(r.get(feature_col))
        tv = _to_float(r.get("true_perf_diff"))
        if fv is None or tv is None:
            continue
        x.append(fv)
        y.append(tv)
        if _sign(fv) == _sign(tv):
            sign_hits += 1
        sign_n += 1
    p, p_reason = _pearson(x, y)
    s, s_reason = _spearman(x, y)
    return {
        "feature": feature_col,
        "n_for_eval": len(x),
        "sign_accuracy": (sign_hits / sign_n) if sign_n > 0 else None,
        "pearson": p,
        "pearson_unavailable_reason": p_reason,
        "spearman": s,
        "spearman_unavailable_reason": s_reason,
    }


def _summary_md(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Pair-level 预测实验最终汇总")
    lines.append("")
    lines.append(f"- 输出时间: {payload.get('timestamp')}")
    lines.append(f"- 输出目录: `{payload.get('output_root')}`")
    lines.append(f"- feature_epoch: `{payload.get('feature_epoch')}`")
    lines.append(f"- label_epoch: `{payload.get('label_epoch')}`")
    lines.append("")
    lines.append("## 文件概览")
    outs = payload.get("outputs", {})
    if isinstance(outs, dict):
        for k, v in outs.items():
            lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Feature 与真实增益相关性")
    feat = payload.get("feature_eval_rows", [])
    if isinstance(feat, list) and feat:
        lines.append("")
        lines.append("|feature|n_for_eval|sign_accuracy|pearson|spearman|")
        lines.append("|---|---:|---:|---:|---:|")
        for r in feat:
            if not isinstance(r, dict):
                continue
            lines.append(
                f"|{r.get('feature')}|{r.get('n_for_eval')}|{r.get('sign_accuracy')}|{r.get('pearson')}|{r.get('spearman')}|"
            )
    else:
        lines.append("- 无可评估的 feature 行。")
    lines.append("")
    lines.append("## Predictor 结果")
    pred = payload.get("predictor_summary", {})
    if isinstance(pred, dict) and pred:
        for k in [
            "feature_epoch",
            "label_epoch",
            "mode",
            "n_rows_total",
            "n_rows_valid_feature_and_label",
            "n_rows_scored",
            "sign_accuracy",
            "pearson",
            "spearman",
        ]:
            lines.append(f"- `{k}`: {pred.get(k)}")
    else:
        lines.append("- 未找到 predictor summary。")
    lines.append("")
    lines.append("## Unavailable 列表")
    una = payload.get("unavailable_files", [])
    if isinstance(una, list) and una:
        for p in una:
            lines.append(f"- `{p}`")
    else:
        lines.append("- 无 unavailable 文件。")
    lines.append("")
    return "\n".join(lines) + "\n"


def _filter_feature_rows(rows: Sequence[Dict[str, object]], feature_epoch: str, label_epoch: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in rows:
        fe = str(r.get("feature_epoch", r.get("epoch", "")))
        le = str(r.get("label_epoch", r.get("true_perf_epoch", fe)))
        if fe == feature_epoch and le == label_epoch:
            out.append(r)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Pair-level predictor step03 final summary")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default=None, help="Default: <result_root>/pair_pred")
    # Backward-compatible alias: --epoch maps to feature_epoch if --feature_epoch absent.
    p.add_argument("--epoch", type=str, default="epoch_1")
    p.add_argument("--feature_epoch", type=str, default="")
    p.add_argument("--label_epoch", type=str, default="", help="If empty, use same as feature_epoch")
    p.add_argument("--features_json", type=str, default="")
    p.add_argument("--predictor_summary_json", type=str, default="")
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    output_root = args.output_root or os.path.join(result_root, "pair_pred")
    os.makedirs(output_root, exist_ok=True)

    feature_epoch = args.feature_epoch.strip() or str(args.epoch).strip() or "epoch_1"
    label_epoch = args.label_epoch.strip() or feature_epoch
    epoch_tag = f"feat_{feature_epoch}__label_{label_epoch}"

    features_json = args.features_json or os.path.join(output_root, "pair_features_all.json")
    feature_rows_all = _load_json_list(features_json)
    feature_rows = _filter_feature_rows(feature_rows_all, feature_epoch, label_epoch)

    feat_eval_rows: List[Dict[str, object]] = []
    for feature_col in ["DeltaCent", "DeltaLoad", "DeltaSelf"]:
        feat_eval_rows.append(_feature_eval(feature_rows, feature_col))
    feat_eval_csv = os.path.join(output_root, f"pair_pred_feature_eval_{epoch_tag}.csv")
    write_rows_csv(feat_eval_csv, feat_eval_rows)

    pred_summary_json = args.predictor_summary_json or os.path.join(output_root, f"pair_pred_summary_{epoch_tag}.json")
    predictor_summary = _load_json(pred_summary_json) if os.path.isfile(pred_summary_json) else {}

    unavailable_files = sorted(glob.glob(os.path.join(output_root, "unavailable", "unavailable_*.json")))

    final_payload: Dict[str, object] = {
        "timestamp": np.datetime64("now").astype(str),
        "output_root": os.path.abspath(output_root),
        "epoch": feature_epoch,  # backward-compatible
        "feature_epoch": feature_epoch,
        "label_epoch": label_epoch,
        "feature_eval_rows": feat_eval_rows,
        "predictor_summary": predictor_summary,
        "unavailable_files": [os.path.abspath(x) for x in unavailable_files],
        "outputs": {
            "feature_eval_csv": os.path.abspath(feat_eval_csv),
            "pair_features_json": os.path.abspath(features_json),
            "predictor_summary_json": os.path.abspath(pred_summary_json) if os.path.isfile(pred_summary_json) else None,
        },
    }

    summary_json = os.path.join(output_root, f"pair_pred_final_summary_{epoch_tag}.json")
    summary_md = os.path.join(output_root, f"pair_pred_final_summary_{epoch_tag}.md")
    summary_csv = os.path.join(output_root, f"pair_pred_final_summary_{epoch_tag}.csv")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(_summary_md(final_payload))

    summary_rows = [
        {
            "epoch": feature_epoch,
            "feature_epoch": feature_epoch,
            "label_epoch": label_epoch,
            "predictor_mode": predictor_summary.get("mode") if isinstance(predictor_summary, dict) else None,
            "predictor_sign_accuracy": predictor_summary.get("sign_accuracy") if isinstance(predictor_summary, dict) else None,
            "predictor_pearson": predictor_summary.get("pearson") if isinstance(predictor_summary, dict) else None,
            "predictor_spearman": predictor_summary.get("spearman") if isinstance(predictor_summary, dict) else None,
            "feature_DeltaCent_pearson": feat_eval_rows[0].get("pearson") if feat_eval_rows else None,
            "feature_DeltaLoad_pearson": feat_eval_rows[1].get("pearson") if len(feat_eval_rows) > 1 else None,
            "feature_DeltaSelf_pearson": feat_eval_rows[2].get("pearson") if len(feat_eval_rows) > 2 else None,
            "feature_DeltaCent_sign_accuracy": feat_eval_rows[0].get("sign_accuracy") if feat_eval_rows else None,
            "feature_DeltaLoad_sign_accuracy": feat_eval_rows[1].get("sign_accuracy") if len(feat_eval_rows) > 1 else None,
            "feature_DeltaSelf_sign_accuracy": feat_eval_rows[2].get("sign_accuracy") if len(feat_eval_rows) > 2 else None,
            "unavailable_files_count": len(unavailable_files),
        }
    ]
    write_rows_csv(summary_csv, summary_rows)

    print(os.path.abspath(summary_json))
    print(os.path.abspath(summary_md))
    print(os.path.abspath(summary_csv))


if __name__ == "__main__":
    main()
