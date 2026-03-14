#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scheme A - Step 07
Task-level score-CKA / score-HSIC bridge (Gram view).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAINF_ROOT_DEFAULT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(DATAINF_ROOT_DEFAULT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gram_scheme_a_utils import (  # type: ignore  # noqa: E402
    DEFAULT_EPOCHS,
    DEFAULT_TRAIN_DATASETS,
    detect_datainf_root,
    gaussian_hsic_from_gram,
    linear_cka_from_gram,
    linear_hsic_from_gram,
    normalize_epoch_list,
    resolve_result_root,
    split_csv_arg,
    write_rows_csv,
    write_rows_txt,
)


def _pick_train_datasets(args) -> List[str]:
    if args.all_train_datasets:
        return list(DEFAULT_TRAIN_DATASETS)
    if args.train_dataset:
        return split_csv_arg(args.train_dataset, DEFAULT_TRAIN_DATASETS)
    return list(DEFAULT_TRAIN_DATASETS)


def _pick_epochs(args) -> List[str]:
    if args.all_epochs:
        return list(DEFAULT_EPOCHS)
    if args.epoch:
        return normalize_epoch_list(split_csv_arg(args.epoch, DEFAULT_EPOCHS))
    return list(DEFAULT_EPOCHS)


def _matrix_path(schemea_root: str, mode: str, train_dataset: str, epoch: str, method: str) -> Optional[str]:
    if mode == "own":
        p = os.path.join(schemea_root, "ownH", train_dataset, epoch, method, f"T_{train_dataset}_{epoch}_{method}_ownH.npy")
        return p if os.path.isfile(p) else None
    if mode == "mixed":
        p = os.path.join(schemea_root, "mixedH", train_dataset, epoch, f"target_{method}", f"T_{train_dataset}_{epoch}_{method}_mixedH.npy")
        return p if os.path.isfile(p) else None
    if mode == "oracle_sft":
        if method == "sft":
            return _matrix_path(schemea_root, "own", train_dataset, epoch, "sft")
        p = os.path.join(
            schemea_root,
            "crossH",
            train_dataset,
            epoch,
            "target_sdft__oracle_sft",
            f"T_{train_dataset}_{epoch}_sdft_under_sft_crossH.npy",
        )
        return p if os.path.isfile(p) else None
    if mode == "oracle_sdft":
        if method == "sdft":
            return _matrix_path(schemea_root, "own", train_dataset, epoch, "sdft")
        p = os.path.join(
            schemea_root,
            "crossH",
            train_dataset,
            epoch,
            "target_sft__oracle_sdft",
            f"T_{train_dataset}_{epoch}_sft_under_sdft_crossH.npy",
        )
        return p if os.path.isfile(p) else None
    return None


def _align_shape(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if A.shape == B.shape:
        return A, B
    n = min(A.shape[0], B.shape[0])
    return A[:n, :n], B[:n, :n]


def main() -> None:
    p = argparse.ArgumentParser(description="Scheme A task-level score CKA/HSIC")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--schemea_root", type=str, default=None, help="默认: <result_root>/schemeA")
    p.add_argument("--train_dataset", type=str, default="")
    p.add_argument("--all_train_datasets", action="store_true")
    p.add_argument("--epoch", type=str, default="")
    p.add_argument("--all_epochs", action="store_true")
    p.add_argument("--h_modes", type=str, default="own,mixed,oracle_sft,oracle_sdft")
    p.add_argument("--enable_gaussian_hsic", action="store_true")
    p.add_argument("--output_root", type=str, default=None)
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    schemea_root = args.schemea_root or os.path.join(result_root, "schemeA")
    output_root = args.output_root or os.path.join(result_root, "schemeA", "score_hsic")
    os.makedirs(output_root, exist_ok=True)

    train_datasets = _pick_train_datasets(args)
    epochs = _pick_epochs(args)
    h_modes = [x.strip() for x in args.h_modes.split(",") if x.strip()]

    rows: List[Dict[str, object]] = []

    for train_dataset in train_datasets:
        for epoch in epochs:
            for h_mode in h_modes:
                path_sft = _matrix_path(schemea_root, h_mode, train_dataset, epoch, "sft")
                path_sdft = _matrix_path(schemea_root, h_mode, train_dataset, epoch, "sdft")
                pair_name = f"{train_dataset}_{epoch}_{h_mode}_sft_vs_sdft"

                if not path_sft or not path_sdft:
                    rows.append(
                        {
                            "train_dataset": train_dataset,
                            "epoch": epoch,
                            "h_mode": h_mode,
                            "pair_name": pair_name,
                            "status": "missing_matrix",
                            "matrix_sft": path_sft,
                            "matrix_sdft": path_sdft,
                        }
                    )
                    continue

                try:
                    A = np.load(path_sft)
                    B = np.load(path_sdft)
                    A, B = _align_shape(A, B)
                    row: Dict[str, object] = {
                        "train_dataset": train_dataset,
                        "epoch": epoch,
                        "h_mode": h_mode,
                        "pair_name": pair_name,
                        "status": "ok",
                        "matrix_sft": os.path.abspath(path_sft),
                        "matrix_sdft": os.path.abspath(path_sdft),
                        "shape_used": str(A.shape),
                        "score_linear_hsic": linear_hsic_from_gram(A, B, centered=True),
                        "score_linear_cka": linear_cka_from_gram(A, B),
                    }
                    if args.enable_gaussian_hsic:
                        g = gaussian_hsic_from_gram(A, B)
                        row["score_gaussian_hsic"] = g["gaussian_hsic"]
                        row["sigma_x"] = g["sigma_x"]
                        row["sigma_y"] = g["sigma_y"]
                    rows.append(row)
                except Exception as e:
                    rows.append(
                        {
                            "train_dataset": train_dataset,
                            "epoch": epoch,
                            "h_mode": h_mode,
                            "pair_name": pair_name,
                            "status": "compute_error",
                            "matrix_sft": os.path.abspath(path_sft),
                            "matrix_sdft": os.path.abspath(path_sdft),
                            "error": str(e),
                        }
                    )

    out_csv = os.path.join(output_root, "score_cka_hsic_summary.csv")
    out_json = os.path.join(output_root, "score_cka_hsic_summary.json")
    out_txt = os.path.join(output_root, "score_cka_hsic_summary.txt")
    write_rows_csv(out_csv, rows)
    write_rows_txt(out_txt, rows, max_cols=16)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(os.path.abspath(out_csv))
    print(os.path.abspath(out_json))
    print(os.path.abspath(out_txt))


if __name__ == "__main__":
    main()
