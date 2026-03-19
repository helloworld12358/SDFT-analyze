#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pair-level predictor - Step 00

Inventory + standardized loader:
1) scan matrix sources (analysis_log / analysis_corr_safe)
2) standardize performance table (for true label extraction)
3) write inventory_mapping.{json,md}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAINF_ROOT_DEFAULT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(DATAINF_ROOT_DEFAULT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gram_scheme_a_utils import (  # type: ignore  # noqa: E402
    DEFAULT_EPOCHS,
    DEFAULT_METHODS,
    DEFAULT_TASKS,
    DEFAULT_TRAIN_DATASETS,
    detect_datainf_root,
    discover_performance_rows,
    find_analysis_corr_file,
    load_existing_ownh_from_analysis,
    resolve_existing_result_roots,
    resolve_result_root,
    resolve_sdft_root,
    write_rows_csv,
    write_rows_txt,
)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


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


def _load_csv_rows(path: str) -> List[Dict[str, object]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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


def _is_square_shape_5(mat: object) -> bool:
    return isinstance(mat, np.ndarray) and mat.shape == (5, 5)


def _pick_performance_rows_from_schemea(schemea_root: str) -> List[Dict[str, object]]:
    # Prefer finalized summary if present.
    per_json = os.path.join(schemea_root, "final_summary", "schemeA_per_method_summary.json")
    per_csv = os.path.join(schemea_root, "final_summary", "schemeA_per_method_summary.csv")
    rows = _load_json_list(per_json)
    if not rows:
        rows = _load_csv_rows(per_csv)
    if not rows:
        return []

    keep_keys = {"train_dataset", "epoch", "method"}
    perf_keys = set()
    for r in rows:
        for k in r.keys():
            if str(k).startswith("metric_") or k in ("perf_log_exists", "perf_log_path"):
                perf_keys.add(k)

    table: Dict[tuple, Dict[str, object]] = {}
    for r in rows:
        td = str(r.get("train_dataset", ""))
        ep = str(r.get("epoch", ""))
        md = str(r.get("method", ""))
        if not td or not ep or md not in DEFAULT_METHODS:
            continue
        k = (td, ep, md)
        row = table.setdefault(k, {"train_dataset": td, "epoch": ep, "method": md, "source": "schemeA_per_method_summary"})
        for pk in perf_keys:
            v = r.get(pk)
            if pk.startswith("metric_"):
                fv = _to_float(v)
                if fv is not None:
                    row[pk] = fv
            else:
                if v is not None and str(v).strip() != "":
                    row[pk] = v

    return list(table.values())


def _pick_performance_rows(datainf_root: str, schemea_root: str) -> List[Dict[str, object]]:
    rows = _pick_performance_rows_from_schemea(schemea_root)
    if rows:
        return rows
    sdft_root = resolve_sdft_root(datainf_root)
    raw = discover_performance_rows(
        sdft_root=sdft_root,
        models=DEFAULT_TRAIN_DATASETS,
        methods=DEFAULT_METHODS,
        epochs=DEFAULT_EPOCHS,
    )
    out: List[Dict[str, object]] = []
    for r in raw:
        row: Dict[str, object] = {
            "train_dataset": r.get("train_dataset"),
            "epoch": r.get("epoch"),
            "method": r.get("method"),
            "source": "discover_performance_rows",
            "perf_log_exists": r.get("perf_log_exists"),
            "perf_log_path": r.get("perf_log_path"),
        }
        for k, v in r.items():
            if str(k).startswith("metric_"):
                row[k] = v
        out.append(row)
    return out


def _shape_status(existing: Dict[str, Dict[str, object]]) -> Dict[str, bool]:
    return {
        "sft_T_5x5": _is_square_shape_5(existing.get("sft", {}).get("T")),
        "sft_C_5x5": _is_square_shape_5(existing.get("sft", {}).get("C")),
        "sdft_T_5x5": _is_square_shape_5(existing.get("sdft", {}).get("T")),
        "sdft_C_5x5": _is_square_shape_5(existing.get("sdft", {}).get("C")),
    }


def _scan_matrix_sources(result_roots: Sequence[str], train_datasets: Sequence[str], epochs: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for ds in train_datasets:
        for ep in epochs:
            existing = load_existing_ownh_from_analysis(result_roots, ds, ep)
            shape = _shape_status(existing)
            found_root = existing.get("sft", {}).get("source_root") or existing.get("sdft", {}).get("source_root")
            base = os.path.join(str(found_root), ds, ep) if found_root else ""
            analysis_log = os.path.join(base, "analysis", "analysis_log.txt") if base else ""
            analysis_corr = find_analysis_corr_file(os.path.join(base, "analysis_safe")) if base else None
            status = "ok" if all(shape.values()) else "partial_or_missing"
            rows.append(
                {
                    "train_dataset": ds,
                    "epoch": ep,
                    "status": status,
                    "source_root": found_root,
                    "analysis_log_path": analysis_log,
                    "analysis_log_exists": bool(analysis_log and os.path.isfile(analysis_log)),
                    "analysis_corr_path": analysis_corr,
                    "analysis_corr_exists": bool(analysis_corr and os.path.isfile(analysis_corr)),
                    **shape,
                }
            )
    return rows


def _build_inventory_md(mapping: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Pair-level 预测实验 Inventory Mapping")
    lines.append("")
    lines.append(f"- 生成时间: {mapping['timestamp']}")
    lines.append(f"- DataInf 根目录: `{mapping['datainf_root']}`")
    lines.append(f"- 结果根目录(自动解析): `{mapping['result_root']}`")
    lines.append(f"- pair_pred 输出目录: `{mapping['pair_pred_root']}`")
    lines.append("")
    lines.append("## 任务与矩阵约定")
    lines.append(f"- 任务顺序沿用项目默认: `{','.join(DEFAULT_TASKS)}`")
    lines.append("- 本实验按上述顺序解释 5x5 矩阵的行列。")
    lines.append("- 若云端原始日志实际顺序不一致，请在运行后检查 `matrix_source_scan` 中的状态并人工确认。")
    lines.append("")
    lines.append("## 矩阵来源扫描（analysis_log / analysis_corr_safe）")
    lines.append(f"- 扫描根目录: `{', '.join(mapping['result_roots_scanned'])}`")
    rows = mapping.get("matrix_source_scan", [])
    ok_count = sum(1 for r in rows if isinstance(r, dict) and r.get("status") == "ok")
    lines.append(f"- 总组合: {len(rows)}")
    lines.append(f"- 完整组合(四个矩阵都 5x5): {ok_count}")
    lines.append("")
    lines.append("## 真实性能标签来源")
    perf = mapping.get("performance_loader", {})
    lines.append(f"- 来源策略: `{perf.get('strategy')}`")
    lines.append(f"- 标准化行数: `{perf.get('rows_count')}`")
    lines.append(f"- 标准化文件(JSON): `{perf.get('perf_json')}`")
    lines.append(f"- 标准化文件(CSV): `{perf.get('perf_csv')}`")
    lines.append("")
    lines.append("## 复用的现有逻辑")
    lines.append("- `gram_scheme_a_utils.load_existing_ownh_from_analysis`（读取 analysis 文本矩阵）")
    lines.append("- `gram_scheme_a_utils.parse_method_matrices_from_analysis_txt`（鲁棒文本矩阵 parser）")
    lines.append("- `gram_scheme_a_utils.discover_performance_rows`（性能日志聚合兜底）")
    lines.append("- `gram_scheme_a_utils.resolve_result_root / resolve_existing_result_roots`（result/results 自动兼容）")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Pair-level predictor step00 inventory + loader")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default=None, help="Default: <result_root>/pair_pred")
    p.add_argument("--extra_result_roots", type=str, default="")
    p.add_argument("--train_dataset", type=str, default="")
    p.add_argument("--all_train_datasets", action="store_true")
    p.add_argument("--epoch", type=str, default="")
    p.add_argument("--all_epochs", action="store_true")
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    pair_pred_root = args.output_root or os.path.join(result_root, "pair_pred")
    _ensure_dir(pair_pred_root)

    if args.all_train_datasets or not args.train_dataset:
        train_datasets = list(DEFAULT_TRAIN_DATASETS)
    else:
        train_datasets = [x.strip() for x in args.train_dataset.split(",") if x.strip()]
    if args.all_epochs or not args.epoch:
        epochs = list(DEFAULT_EPOCHS)
    else:
        epochs = [x.strip() for x in args.epoch.split(",") if x.strip()]

    extra_roots = [x.strip() for x in args.extra_result_roots.split(",") if x.strip()]
    result_roots = resolve_existing_result_roots(datainf_root, explicit_roots=extra_roots)

    matrix_scan_rows = _scan_matrix_sources(result_roots, train_datasets, epochs)
    matrix_scan_csv = os.path.join(pair_pred_root, "matrix_source_scan.csv")
    matrix_scan_txt = os.path.join(pair_pred_root, "matrix_source_scan.txt")
    write_rows_csv(matrix_scan_csv, matrix_scan_rows)
    write_rows_txt(matrix_scan_txt, matrix_scan_rows, max_cols=18)

    schemea_root = os.path.join(result_root, "schemeA")
    perf_rows = _pick_performance_rows(datainf_root, schemea_root)
    perf_dir = _ensure_dir(os.path.join(pair_pred_root, "perf"))
    perf_csv = os.path.join(perf_dir, "perf_standardized.csv")
    perf_json = os.path.join(perf_dir, "perf_standardized.json")
    perf_txt = os.path.join(perf_dir, "perf_standardized.txt")
    write_rows_csv(perf_csv, perf_rows)
    write_rows_txt(perf_txt, perf_rows, max_cols=22)
    with open(perf_json, "w", encoding="utf-8") as f:
        json.dump(perf_rows, f, ensure_ascii=False, indent=2)

    mapping: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "datainf_root": os.path.abspath(datainf_root),
        "result_root": os.path.abspath(result_root),
        "pair_pred_root": os.path.abspath(pair_pred_root),
        "result_roots_scanned": [os.path.abspath(x) for x in result_roots],
        "enumeration": {
            "train_datasets": train_datasets,
            "epochs": epochs,
            "methods": list(DEFAULT_METHODS),
            "task_order_assumed": list(DEFAULT_TASKS),
        },
        "matrix_source_scan": matrix_scan_rows,
        "performance_loader": {
            "strategy": "schemeA_per_method_summary_first_then_log_discovery",
            "rows_count": len(perf_rows),
            "perf_csv": os.path.abspath(perf_csv),
            "perf_json": os.path.abspath(perf_json),
        },
        "notes": [
            "Matrix parser is reused from gram_scheme_a_utils.parse_method_matrices_from_analysis_txt.",
            "Task order assumption follows DEFAULT_TASKS unless your original logs use different ordering.",
            "If cloud logs differ from this assumption, please verify and adjust before final analysis.",
        ],
    }

    mapping_json = os.path.join(pair_pred_root, "inventory_mapping.json")
    mapping_md = os.path.join(pair_pred_root, "inventory_mapping.md")
    with open(mapping_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    with open(mapping_md, "w", encoding="utf-8") as f:
        f.write(_build_inventory_md(mapping))

    print(os.path.abspath(mapping_json))
    print(os.path.abspath(mapping_md))
    print(os.path.abspath(perf_csv))
    print(os.path.abspath(matrix_scan_csv))


if __name__ == "__main__":
    main()
