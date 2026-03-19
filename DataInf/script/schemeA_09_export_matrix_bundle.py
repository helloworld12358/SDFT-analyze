#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scheme A - Step 09
Export all available T/C matrices into one JSON + one TXT report.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAINF_ROOT_DEFAULT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(DATAINF_ROOT_DEFAULT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gram_scheme_a_utils import (  # type: ignore  # noqa: E402
    detect_datainf_root,
    resolve_result_root,
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


def _to_abs(path_like: Optional[str], base_dir: str) -> Optional[str]:
    if not path_like:
        return None
    p = str(path_like)
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_dir, p))


def _pick_first_existing(candidates: List[Optional[str]]) -> Optional[str]:
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def _resolve_summary_path(summary_path_like: Optional[str], fallback_candidates: List[str]) -> Optional[str]:
    cands: List[Optional[str]] = []
    if summary_path_like:
        p = str(summary_path_like)
        cands.append(p if os.path.isabs(p) else os.path.abspath(p))
    cands.extend(fallback_candidates)
    return _pick_first_existing(cands)


def _safe_load_matrix(path_like: Optional[str]) -> Optional[np.ndarray]:
    if not path_like:
        return None
    p = str(path_like)
    if not os.path.isfile(p):
        return None
    try:
        if p.endswith(".npy"):
            return np.load(p)
        if p.endswith(".csv"):
            return np.loadtxt(p, delimiter=",")
        return None
    except Exception:
        return None


def _shape_ok(arr: np.ndarray, expect_shape: Optional[Tuple[int, int]]) -> bool:
    if expect_shape is None:
        return True
    return tuple(arr.shape) == tuple(expect_shape)


def _to_builtin_matrix(arr: np.ndarray) -> List[List[float]]:
    return np.asarray(arr, dtype=np.float64).tolist()


def _matrix_lines(name: str, mat: List[List[float]]) -> List[str]:
    out = [f"{name}:"]
    for row in mat:
        vals = ", ".join(f"{float(v):.12g}" for v in row)
        out.append(f"  [{vals}]")
    return out


def _extract_from_summary_json(summary_json: str) -> Tuple[Optional[str], Optional[str], Dict[str, object]]:
    if not os.path.isfile(summary_json):
        return None, None, {}
    try:
        obj = _load_json(summary_json)
    except Exception:
        return None, None, {}
    if not isinstance(obj, dict):
        return None, None, {}
    paths = obj.get("paths", {})
    if not isinstance(paths, dict):
        paths = {}
    t_path = paths.get("T_npy") or paths.get("T_csv")
    c_path = paths.get("C_npy") or paths.get("C_csv")
    meta = obj.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    return (
        str(t_path) if t_path else None,
        str(c_path) if c_path else None,
        meta,
    )


def _resolve_matrix_path(path_like: Optional[str], summary_dir: str) -> Optional[str]:
    if not path_like:
        return None
    p = str(path_like)
    if os.path.isabs(p):
        if os.path.isfile(p):
            return p
        # allow relocated result roots: fall back to same filename under summary dir
        b = os.path.basename(p)
        local = os.path.join(summary_dir, b)
        if os.path.isfile(local):
            return local
        return None
    local = os.path.abspath(os.path.join(summary_dir, p))
    if os.path.isfile(local):
        return local
    return None


def collect_rows(
    schemea_root: str,
    include_raw_rewrite: bool,
    expect_shape: Optional[Tuple[int, int]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    own_rows = _load_json_list(os.path.join(schemea_root, "ownH", "ownH_summary.json"))
    for r in own_rows:
        if str(r.get("status")) != "ok":
            continue
        ds = str(r.get("train_dataset"))
        ep = str(r.get("epoch"))
        method = str(r.get("method"))
        run_dir = os.path.join(schemea_root, "ownH", ds, ep, method)
        expected_summary = os.path.join(run_dir, f"summary_{ds}_{ep}_{method}_ownH.json")
        summary_json = _resolve_summary_path(r.get("summary_json"), [expected_summary])
        t_path = _to_abs(r.get("T_npy"), schemea_root)
        c_path = _to_abs(r.get("C_npy"), schemea_root)
        if summary_json and os.path.isfile(summary_json):
            t_from_s, c_from_s, _meta = _extract_from_summary_json(summary_json)
            t_path = _resolve_matrix_path(t_path or t_from_s, os.path.dirname(summary_json))
            c_path = _resolve_matrix_path(c_path or c_from_s, os.path.dirname(summary_json))
        else:
            t_path = _resolve_matrix_path(t_path, run_dir)
            c_path = _resolve_matrix_path(c_path, run_dir)
        t_mat = _safe_load_matrix(t_path)
        c_mat = _safe_load_matrix(c_path)
        if t_mat is None or c_mat is None:
            continue
        if not (_shape_ok(t_mat, expect_shape) and _shape_ok(c_mat, expect_shape)):
            continue
        rows.append(
            {
                "mode": "ownH",
                "h_mode": "own",
                "train_dataset": r.get("train_dataset"),
                "epoch": r.get("epoch"),
                "method": r.get("method"),
                "oracle_method": None,
                "shape_T": list(t_mat.shape),
                "shape_C": list(c_mat.shape),
                "T": _to_builtin_matrix(t_mat),
                "C": _to_builtin_matrix(c_mat),
            }
        )

    cross_rows = _load_json_list(os.path.join(schemea_root, "crossH", "crossH_summary.json"))
    for r in cross_rows:
        if str(r.get("status")) != "ok":
            continue
        ds = str(r.get("train_dataset"))
        ep = str(r.get("epoch"))
        target = str(r.get("target_method"))
        oracle = str(r.get("oracle_method"))
        run_dir = os.path.join(schemea_root, "crossH", ds, ep, f"target_{target}__oracle_{oracle}")
        expected_summary = os.path.join(run_dir, f"summary_{ds}_{ep}_{target}_under_{oracle}_crossH.json")
        summary_json = _resolve_summary_path(r.get("summary_json"), [expected_summary])
        if not summary_json or not os.path.isfile(summary_json):
            continue
        t_path, c_path, _meta = _extract_from_summary_json(summary_json)
        t_path = _resolve_matrix_path(t_path, os.path.dirname(summary_json))
        c_path = _resolve_matrix_path(c_path, os.path.dirname(summary_json))
        t_mat = _safe_load_matrix(t_path)
        c_mat = _safe_load_matrix(c_path)
        if t_mat is None or c_mat is None:
            continue
        if not (_shape_ok(t_mat, expect_shape) and _shape_ok(c_mat, expect_shape)):
            continue
        rows.append(
            {
                "mode": "crossH",
                "h_mode": f"cross_oracle_{oracle}",
                "train_dataset": r.get("train_dataset"),
                "epoch": r.get("epoch"),
                "method": r.get("target_method"),
                "oracle_method": oracle,
                "shape_T": list(t_mat.shape),
                "shape_C": list(c_mat.shape),
                "T": _to_builtin_matrix(t_mat),
                "C": _to_builtin_matrix(c_mat),
            }
        )

    mixed_rows = _load_json_list(os.path.join(schemea_root, "mixedH", "mixedH_summary.json"))
    for r in mixed_rows:
        if str(r.get("status")) != "ok":
            continue
        ds = str(r.get("train_dataset"))
        ep = str(r.get("epoch"))
        target = str(r.get("target_method"))
        run_dir = os.path.join(schemea_root, "mixedH", ds, ep, f"target_{target}")
        expected_summary = os.path.join(run_dir, f"summary_{ds}_{ep}_{target}_mixedH.json")
        summary_json = _resolve_summary_path(r.get("summary_json"), [expected_summary])
        if not summary_json or not os.path.isfile(summary_json):
            continue
        t_path, c_path, _meta = _extract_from_summary_json(summary_json)
        t_path = _resolve_matrix_path(t_path, os.path.dirname(summary_json))
        c_path = _resolve_matrix_path(c_path, os.path.dirname(summary_json))
        t_mat = _safe_load_matrix(t_path)
        c_mat = _safe_load_matrix(c_path)
        if t_mat is None or c_mat is None:
            continue
        if not (_shape_ok(t_mat, expect_shape) and _shape_ok(c_mat, expect_shape)):
            continue
        rows.append(
            {
                "mode": "mixedH",
                "h_mode": "mixed",
                "train_dataset": r.get("train_dataset"),
                "epoch": r.get("epoch"),
                "method": r.get("target_method"),
                "oracle_method": None,
                "shape_T": list(t_mat.shape),
                "shape_C": list(c_mat.shape),
                "T": _to_builtin_matrix(t_mat),
                "C": _to_builtin_matrix(c_mat),
            }
        )

    if include_raw_rewrite:
        rr_glob = os.path.join(schemea_root, "raw_rewrite", "**", "raw_rewrite_summary_*.json")
        for p in sorted(glob.glob(rr_glob, recursive=True)):
            try:
                obj = _load_json(p)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            paths = obj.get("paths", {})
            if not isinstance(paths, dict):
                continue
            matrix_summary = paths.get("matrix_summary")
            matrix_summary = _to_abs(matrix_summary, os.path.dirname(p))
            if not matrix_summary or not os.path.isfile(matrix_summary):
                continue
            t_path, c_path, meta = _extract_from_summary_json(matrix_summary)
            t_path = _resolve_matrix_path(t_path, os.path.dirname(matrix_summary))
            c_path = _resolve_matrix_path(c_path, os.path.dirname(matrix_summary))
            t_mat = _safe_load_matrix(t_path)
            c_mat = _safe_load_matrix(c_path)
            if t_mat is None or c_mat is None:
                continue
            if not (_shape_ok(t_mat, expect_shape) and _shape_ok(c_mat, expect_shape)):
                continue
            rows.append(
                {
                    "mode": "raw_rewrite",
                    "h_mode": str(meta.get("oracle_mode_used", "raw_rewrite")),
                    "train_dataset": obj.get("train_dataset"),
                    "epoch": obj.get("epoch"),
                    "method": obj.get("feature_method"),
                    "oracle_method": meta.get("oracle_mode_used"),
                    "shape_T": list(t_mat.shape),
                    "shape_C": list(c_mat.shape),
                    "T": _to_builtin_matrix(t_mat),
                    "C": _to_builtin_matrix(c_mat),
                }
            )

    rows.sort(
        key=lambda x: (
            str(x.get("mode")),
            str(x.get("train_dataset")),
            str(x.get("epoch")),
            str(x.get("h_mode")),
            str(x.get("method")),
        )
    )
    return rows


def write_txt(path: str, payload: Dict[str, object]) -> None:
    rows = payload.get("rows", [])
    meta = payload.get("meta", {})
    if not isinstance(rows, list):
        rows = []
    if not isinstance(meta, dict):
        meta = {}

    lines: List[str] = []
    lines.append("SchemeA Matrix Bundle Summary")
    lines.append("=" * 80)
    lines.append("meta:")
    for k in sorted(meta.keys()):
        lines.append(f"  {k}: {meta[k]}")
    lines.append("")
    lines.append(f"rows_count: {len(rows)}")
    lines.append("")

    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        lines.append("-" * 80)
        lines.append(f"index: {i}")
        for k in ["mode", "h_mode", "train_dataset", "epoch", "method", "oracle_method", "shape_T", "shape_C"]:
            lines.append(f"{k}: {r.get(k)}")
        t_mat = r.get("T", [])
        c_mat = r.get("C", [])
        if isinstance(t_mat, list):
            lines.extend(_matrix_lines("T", t_mat))
        if isinstance(c_mat, list):
            lines.extend(_matrix_lines("C", c_mat))
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_shape_arg(shape_arg: str) -> Optional[Tuple[int, int]]:
    s = shape_arg.strip().lower()
    if not s or s in ("all", "none"):
        return None
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--shape must be like '5,5' or 'all', got: {shape_arg}")
    return int(parts[0]), int(parts[1])


def main() -> None:
    p = argparse.ArgumentParser(description="Export SchemeA matrices T/C into one JSON and one TXT.")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--schemea_root", type=str, default=None, help="Default: <result_root>/schemeA")
    p.add_argument("--output_root", type=str, default=None, help="Default: <schemea_root>/final_summary")
    p.add_argument("--json_name", type=str, default="schemeA_matrix_bundle_summary.json")
    p.add_argument("--txt_name", type=str, default="schemeA_matrix_bundle_summary.txt")
    p.add_argument("--shape", type=str, default="5,5", help="'5,5' by default, or 'all'")
    p.add_argument("--include_raw_rewrite", action="store_true", help="Also include raw_rewrite matrices.")
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    schemea_root = args.schemea_root or os.path.join(result_root, "schemeA")
    output_root = args.output_root or os.path.join(schemea_root, "final_summary")
    os.makedirs(output_root, exist_ok=True)

    expect_shape = parse_shape_arg(args.shape)
    rows = collect_rows(
        schemea_root=schemea_root,
        include_raw_rewrite=bool(args.include_raw_rewrite),
        expect_shape=expect_shape,
    )

    by_mode = Counter(str(r.get("mode")) for r in rows)
    by_hmode = Counter(str(r.get("h_mode")) for r in rows)
    payload: Dict[str, object] = {
        "meta": {
            "schemea_root": os.path.abspath(schemea_root),
            "shape_filter": list(expect_shape) if expect_shape else "all",
            "include_raw_rewrite": bool(args.include_raw_rewrite),
            "rows_count": len(rows),
            "counts_by_mode": dict(sorted(by_mode.items())),
            "counts_by_h_mode": dict(sorted(by_hmode.items())),
        },
        "rows": rows,
    }

    json_path = os.path.join(output_root, args.json_name)
    txt_path = os.path.join(output_root, args.txt_name)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    write_txt(txt_path, payload)

    print(os.path.abspath(json_path))
    print(os.path.abspath(txt_path))


if __name__ == "__main__":
    main()
