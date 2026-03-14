#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scheme A - Step 05
对 Scheme A 中已存在的 T 矩阵恢复 Gram 坐标，并补齐 shared-mode suite。
不做任何谱修正或负特征值截断。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAINF_ROOT_DEFAULT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(DATAINF_ROOT_DEFAULT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gram_scheme_a_utils import (  # type: ignore  # noqa: E402
    covariance_to_correlation,
    detect_datainf_root,
    resolve_result_root,
    save_coordinate_bundle,
    shared_mode_suite,
    spectral_diagnostics,
    write_rows_csv,
    write_rows_txt,
)


def discover_t_files(root: str) -> List[str]:
    out: List[str] = []
    for cur, _, files in os.walk(root):
        for fn in files:
            if fn.startswith("T_") and fn.endswith(".npy"):
                out.append(os.path.join(cur, fn))
    return sorted(out)


def infer_mode(path: str) -> str:
    norm = path.replace("\\", "/")
    if "/ownH/" in norm:
        return "ownH"
    if "/crossH/" in norm:
        return "crossH"
    if "/mixedH/" in norm:
        return "mixedH"
    if "/raw_rewrite/" in norm:
        return "raw_rewrite"
    return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description="Recover coordinates and shared suite for Scheme A matrices")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--input_root", type=str, default=None, help="默认: <result_root>/schemeA")
    p.add_argument("--output_subdir", type=str, default="coords")
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    input_root = args.input_root or os.path.join(result_root, "schemeA")

    t_files = discover_t_files(input_root)
    rows: List[Dict[str, object]] = []

    for t_path in t_files:
        fn = os.path.basename(t_path)
        tag = fn[len("T_") : -len(".npy")]
        matrix_dir = os.path.dirname(t_path)

        try:
            K = np.load(t_path)
        except Exception as e:
            rows.append({"T_path": os.path.abspath(t_path), "status": "failed_load", "error": str(e)})
            continue

        c_path = os.path.join(matrix_dir, f"C_{tag}.npy")
        if os.path.isfile(c_path):
            try:
                C = np.load(c_path)
                c_source = "existing_C"
            except Exception:
                C = covariance_to_correlation(K)
                c_source = "derived_from_T_due_to_C_load_error"
        else:
            C = covariance_to_correlation(K)
            c_source = "derived_from_T"

        suite = shared_mode_suite(C)
        suite_path = os.path.join(matrix_dir, f"shared_mode_suite_recovered_{tag}.json")
        with open(suite_path, "w", encoding="utf-8") as f:
            json.dump(suite, f, ensure_ascii=False, indent=2)

        suite_summary = {
            "tag": tag,
            "source_T_path": os.path.abspath(t_path),
            "source_C_path": os.path.abspath(c_path) if os.path.isfile(c_path) else None,
            "c_source": c_source,
            "shared_mode_suite": suite,
            "spectral_T": spectral_diagnostics(K),
            "spectral_C": spectral_diagnostics(np.nan_to_num(C, nan=0.0)),
            "note": "No eigenvalue clipping or spectrum cleaning has been applied.",
        }
        suite_summary_path = os.path.join(matrix_dir, f"suite_recover_summary_{tag}.json")
        with open(suite_summary_path, "w", encoding="utf-8") as f:
            json.dump(suite_summary, f, ensure_ascii=False, indent=2)

        object_names = [f"obj_{i}" for i in range(K.shape[0])]
        out_dir = os.path.join(matrix_dir, args.output_subdir)
        coord_bundle = save_coordinate_bundle(
            output_dir=out_dir,
            tag=tag,
            K=K,
            object_names=object_names,
            metadata={
                "source_T_path": os.path.abspath(t_path),
                "interpretation_note": "Z_hat is Gram-world coordinate realization only.",
            },
        )

        spec = spectral_diagnostics(K)
        rows.append(
            {
                "T_path": os.path.abspath(t_path),
                "mode": infer_mode(t_path),
                "status": "ok",
                "tag": tag,
                "suite_json": os.path.abspath(suite_path),
                "suite_summary_json": os.path.abspath(suite_summary_path),
                "coord_summary_json": coord_bundle["summary_json"],
                "Zhat_npy": coord_bundle["Zhat_npy"],
                "eig_min_real": spec.get("eig_min_real"),
                "eig_negative_count_real": spec.get("eig_negative_count_real"),
                "condition_number": spec.get("condition_number"),
            }
        )
        print(coord_bundle["summary_json"])

    summary_csv = os.path.join(input_root, "schemeA_coords_summary.csv")
    summary_json = os.path.join(input_root, "schemeA_coords_summary.json")
    summary_txt = os.path.join(input_root, "schemeA_coords_summary.txt")
    write_rows_csv(summary_csv, rows)
    write_rows_txt(summary_txt, rows, max_cols=16)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(os.path.abspath(summary_csv))
    print(os.path.abspath(summary_json))
    print(os.path.abspath(summary_txt))


if __name__ == "__main__":
    main()
