#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scheme A - Step 08
汇总 Scheme A 指标 + 现有性能日志，输出 per-method 与 sft-minus-sdft 总表。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAINF_ROOT_DEFAULT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(DATAINF_ROOT_DEFAULT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gram_scheme_a_utils import (  # type: ignore  # noqa: E402
    DEFAULT_EPOCHS,
    DEFAULT_METHODS,
    DEFAULT_TRAIN_DATASETS,
    detect_datainf_root,
    discover_performance_rows,
    resolve_result_root,
    resolve_sdft_root,
    write_rows_csv,
    write_rows_txt,
)


def _load_json_list(path: str) -> List[Dict[str, object]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


def _to_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _key(train_dataset: str, epoch: str, method: str, h_mode: str) -> Tuple[str, str, str, str]:
    return train_dataset, epoch, method, h_mode


def _normalize_score_h_mode(h_mode: str) -> str:
    # score bridge historically emits oracle_sft/oracle_sdft
    # while per-method summary rows use cross_oracle_* naming.
    if h_mode == "oracle_sft":
        return "cross_oracle_sft"
    if h_mode == "oracle_sdft":
        return "cross_oracle_sdft"
    return h_mode


def _load_raw_rewrite_rows(raw_rewrite_root: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for p in glob.glob(os.path.join(raw_rewrite_root, "**", "raw_rewrite_summary_*.json"), recursive=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            out.append(
                {
                    "train_dataset": obj.get("train_dataset"),
                    "epoch": obj.get("epoch"),
                    "method": obj.get("feature_method"),
                    "A_leak": obj.get("A_leak"),
                    "style_trace": obj.get("style_covariance", {}).get("trace"),
                    "style_top_eig": obj.get("style_covariance", {}).get("top_eig_real"),
                    "content_trace": obj.get("content_covariance", {}).get("trace"),
                    "content_top_eig": obj.get("content_covariance", {}).get("top_eig_real"),
                    "raw_rewrite_summary_json": os.path.abspath(p),
                }
            )
        except Exception:
            continue
    return out


def _safe_corr_outputs(df, out_prefix: str) -> List[str]:
    paths: List[str] = []
    try:
        num = df.select_dtypes(include=["number"])
        if num.shape[1] >= 2:
            pearson = num.corr(method="pearson")
            spearman = num.corr(method="spearman")
            p_path = f"{out_prefix}_pearson.csv"
            s_path = f"{out_prefix}_spearman.csv"
            pearson.to_csv(p_path)
            spearman.to_csv(s_path)
            paths.extend([os.path.abspath(p_path), os.path.abspath(s_path)])
    except Exception:
        pass
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description="Scheme A final summary")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--schemea_root", type=str, default=None, help="默认: <result_root>/schemeA")
    p.add_argument("--output_root", type=str, default=None, help="默认: <result_root>/schemeA/final_summary")
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    schemea_root = args.schemea_root or os.path.join(result_root, "schemeA")
    output_root = args.output_root or os.path.join(result_root, "schemeA", "final_summary")
    os.makedirs(output_root, exist_ok=True)

    own_rows = _load_json_list(os.path.join(schemea_root, "ownH", "ownH_summary.json"))
    cross_rows = _load_json_list(os.path.join(schemea_root, "crossH", "crossH_summary.json"))
    mixed_rows = _load_json_list(os.path.join(schemea_root, "mixedH", "mixedH_summary.json"))
    score_rows = _load_json_list(os.path.join(schemea_root, "score_hsic", "score_cka_hsic_summary.json"))
    rr_rows = _load_raw_rewrite_rows(os.path.join(schemea_root, "raw_rewrite"))

    table: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}

    for r in own_rows:
        if r.get("status") != "ok" or r.get("mode") != "ownH":
            continue
        train_dataset = str(r.get("train_dataset"))
        epoch = str(r.get("epoch"))
        method = str(r.get("method"))
        k = _key(train_dataset, epoch, method, "own")
        table[k] = {
            "train_dataset": train_dataset,
            "epoch": epoch,
            "method": method,
            "h_mode": "own",
            "lambda1_C": r.get("lambda1_C"),
            "lambda1_minus_lambda2_C": r.get("lambda1_minus_lambda2_C"),
            "mean_offdiag_C": r.get("mean_offdiag_C"),
            "fro_offdiag_C": r.get("fro_offdiag_C"),
            "trace_C": r.get("trace_C"),
            "summary_json": r.get("summary_json"),
        }

    for r in cross_rows:
        if r.get("status") != "ok":
            continue
        train_dataset = str(r.get("train_dataset"))
        epoch = str(r.get("epoch"))
        method = str(r.get("target_method"))
        h_mode = f"cross_oracle_{r.get('oracle_method')}"
        k = _key(train_dataset, epoch, method, h_mode)
        table[k] = {
            "train_dataset": train_dataset,
            "epoch": epoch,
            "method": method,
            "h_mode": h_mode,
            "lambda1_C": r.get("lambda1_C"),
            "lambda1_minus_lambda2_C": r.get("lambda1_minus_lambda2_C"),
            "mean_offdiag_C": r.get("mean_offdiag_C"),
            "fro_offdiag_C": r.get("fro_offdiag_C"),
            "trace_C": r.get("trace_C"),
            "summary_json": r.get("summary_json"),
        }

    for r in mixed_rows:
        if r.get("status") != "ok":
            continue
        train_dataset = str(r.get("train_dataset"))
        epoch = str(r.get("epoch"))
        method = str(r.get("target_method"))
        k = _key(train_dataset, epoch, method, "mixed")
        table[k] = {
            "train_dataset": train_dataset,
            "epoch": epoch,
            "method": method,
            "h_mode": "mixed",
            "lambda1_C": r.get("lambda1_C"),
            "lambda1_minus_lambda2_C": r.get("lambda1_minus_lambda2_C"),
            "mean_offdiag_C": r.get("mean_offdiag_C"),
            "fro_offdiag_C": r.get("fro_offdiag_C"),
            "trace_C": r.get("trace_C"),
            "summary_json": r.get("summary_json"),
        }

    for r in rr_rows:
        train_dataset = str(r.get("train_dataset"))
        epoch = str(r.get("epoch"))
        method = str(r.get("method"))
        for h_mode in ("own", "mixed", "cross_oracle_sft", "cross_oracle_sdft"):
            k = _key(train_dataset, epoch, method, h_mode)
            row = table.setdefault(k, {"train_dataset": train_dataset, "epoch": epoch, "method": method, "h_mode": h_mode})
            row["A_leak"] = r.get("A_leak")
            row["style_trace"] = r.get("style_trace")
            row["style_top_eig"] = r.get("style_top_eig")
            row["content_trace"] = r.get("content_trace")
            row["content_top_eig"] = r.get("content_top_eig")
            row["raw_rewrite_summary_json"] = r.get("raw_rewrite_summary_json")

    score_map: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for r in score_rows:
        if r.get("status") != "ok":
            continue
        score_h_mode = _normalize_score_h_mode(str(r.get("h_mode")))
        score_map[(str(r.get("train_dataset")), str(r.get("epoch")), score_h_mode)] = {
            "score_linear_cka": r.get("score_linear_cka"),
            "score_linear_hsic": r.get("score_linear_hsic"),
            "score_gaussian_hsic": r.get("score_gaussian_hsic"),
        }

    for (train_dataset, epoch, method, h_mode), row in table.items():
        s = score_map.get((train_dataset, epoch, h_mode))
        if s:
            row.update(s)

    sdft_root = resolve_sdft_root(datainf_root)
    perf_rows = discover_performance_rows(
        sdft_root=sdft_root,
        models=DEFAULT_TRAIN_DATASETS,
        methods=DEFAULT_METHODS,
        epochs=DEFAULT_EPOCHS,
    )
    perf_map = {(str(r["train_dataset"]), str(r["epoch"]), str(r["method"])): r for r in perf_rows}

    for (train_dataset, epoch, method, _h_mode), row in table.items():
        pr = perf_map.get((train_dataset, epoch, method))
        if not pr:
            continue
        row["perf_log_path"] = pr.get("perf_log_path")
        row["perf_log_exists"] = pr.get("perf_log_exists")
        for k, v in pr.items():
            if k.startswith("metric_"):
                row[k] = v

    per_method_rows = sorted(table.values(), key=lambda x: (str(x.get("train_dataset")), str(x.get("epoch")), str(x.get("h_mode")), str(x.get("method"))))

    grouped: Dict[Tuple[str, str, str], Dict[str, Dict[str, object]]] = {}
    for r in per_method_rows:
        gk = (str(r.get("train_dataset")), str(r.get("epoch")), str(r.get("h_mode")))
        grouped.setdefault(gk, {})[str(r.get("method"))] = r

    diff_rows: List[Dict[str, object]] = []
    for (train_dataset, epoch, h_mode), m in sorted(grouped.items()):
        sft = m.get("sft")
        sdft = m.get("sdft")
        if not sft or not sdft:
            continue
        row: Dict[str, object] = {
            "train_dataset": train_dataset,
            "epoch": epoch,
            "h_mode": h_mode,
            "delta_method": "sft_minus_sdft",
        }
        keys = set(sft.keys()) | set(sdft.keys())
        for k in keys:
            if k in ("train_dataset", "epoch", "method", "h_mode"):
                continue
            a = _to_float(sft.get(k))
            b = _to_float(sdft.get(k))
            if a is not None and b is not None:
                row[f"delta_{k}"] = a - b
        diff_rows.append(row)

    per_csv = os.path.join(output_root, "schemeA_per_method_summary.csv")
    per_json = os.path.join(output_root, "schemeA_per_method_summary.json")
    per_txt = os.path.join(output_root, "schemeA_per_method_summary.txt")
    diff_csv = os.path.join(output_root, "schemeA_sft_minus_sdft_summary.csv")
    diff_json = os.path.join(output_root, "schemeA_sft_minus_sdft_summary.json")
    diff_txt = os.path.join(output_root, "schemeA_sft_minus_sdft_summary.txt")

    write_rows_csv(per_csv, per_method_rows)
    write_rows_txt(per_txt, per_method_rows, max_cols=18)
    with open(per_json, "w", encoding="utf-8") as f:
        json.dump(per_method_rows, f, ensure_ascii=False, indent=2)

    write_rows_csv(diff_csv, diff_rows)
    write_rows_txt(diff_txt, diff_rows, max_cols=18)
    with open(diff_json, "w", encoding="utf-8") as f:
        json.dump(diff_rows, f, ensure_ascii=False, indent=2)

    corr_files: List[str] = []
    try:
        import pandas as pd  # type: ignore

        corr_files += _safe_corr_outputs(pd.DataFrame(per_method_rows), os.path.join(output_root, "schemeA_per_method_corr"))
        corr_files += _safe_corr_outputs(pd.DataFrame(diff_rows), os.path.join(output_root, "schemeA_diff_corr"))
    except Exception:
        pass

    run_summary = {
        "schemea_root": os.path.abspath(schemea_root),
        "output_root": os.path.abspath(output_root),
        "sources": {
            "own": os.path.join(schemea_root, "ownH", "ownH_summary.json"),
            "cross": os.path.join(schemea_root, "crossH", "crossH_summary.json"),
            "mixed": os.path.join(schemea_root, "mixedH", "mixedH_summary.json"),
            "score": os.path.join(schemea_root, "score_hsic", "score_cka_hsic_summary.json"),
            "raw_rewrite_glob": os.path.join(schemea_root, "raw_rewrite", "**", "raw_rewrite_summary_*.json"),
        },
        "rows_count": {"per_method": len(per_method_rows), "diff": len(diff_rows)},
        "outputs": {
            "per_method_csv": os.path.abspath(per_csv),
            "diff_csv": os.path.abspath(diff_csv),
            "correlation_files": corr_files,
        },
    }
    run_summary_path = os.path.join(output_root, "schemeA_final_run_summary.json")
    with open(run_summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print(os.path.abspath(per_csv))
    print(os.path.abspath(diff_csv))
    print(os.path.abspath(run_summary_path))
    for cp in corr_files:
        print(cp)


if __name__ == "__main__":
    main()
