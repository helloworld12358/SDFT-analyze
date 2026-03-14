#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scheme A - Step 03
Task-level cross-H Gram experiments.

固定两个方向：
- target=sft, oracle=sdft
- target=sdft, oracle=sft
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAINF_ROOT_DEFAULT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(DATAINF_ROOT_DEFAULT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gram_scheme_a_utils import (  # type: ignore  # noqa: E402
    DEFAULT_EPOCHS,
    DEFAULT_TASKS,
    DEFAULT_TRAIN_DATASETS,
    compute_pairwise_scores_via_cli,
    detect_datainf_root,
    normalize_epoch_list,
    resolve_checkpoint_path,
    resolve_result_root,
    resolve_sdft_root,
    resolve_train_dataset_path,
    resolve_grad_path,
    save_matrix_bundle,
    split_csv_arg,
    write_rows_csv,
    write_rows_txt,
    write_unavailable_note,
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


def _cross_pairs() -> List[Tuple[str, str]]:
    return [("sft", "sdft"), ("sdft", "sft")]


def main() -> None:
    p = argparse.ArgumentParser(description="Scheme A cross-H task-level Gram")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--train_dataset", type=str, default="")
    p.add_argument("--all_train_datasets", action="store_true")
    p.add_argument("--epoch", type=str, default="")
    p.add_argument("--all_epochs", action="store_true")
    p.add_argument("--task_names", type=str, default=",".join(DEFAULT_TASKS))
    p.add_argument("--base_model_path", type=str, default=None)
    p.add_argument("--damping", type=float, default=0.001)
    p.add_argument("--python_exe", type=str, default=sys.executable)
    p.add_argument("--output_root", type=str, default=None)
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    sdft_root = resolve_sdft_root(datainf_root)
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    output_root = args.output_root or os.path.join(result_root, "schemeA", "crossH")

    train_datasets = _pick_train_datasets(args)
    epochs = _pick_epochs(args)
    task_names = split_csv_arg(args.task_names, DEFAULT_TASKS)
    base_model_path = args.base_model_path or os.path.join(sdft_root, "model", "Llama-2-7b-chat-hf")

    rows: List[Dict[str, object]] = []

    for train_dataset in train_datasets:
        for epoch in epochs:
            for target_method, oracle_method in _cross_pairs():
                combo_dir = os.path.join(
                    output_root,
                    train_dataset,
                    epoch,
                    f"target_{target_method}__oracle_{oracle_method}",
                )
                tag = f"{train_dataset}_{epoch}_{target_method}_under_{oracle_method}_crossH"

                train_data_path = resolve_train_dataset_path(sdft_root, train_dataset, oracle_method)
                lora_path = resolve_checkpoint_path(sdft_root, epoch, train_dataset, oracle_method)
                grad_paths = {t: resolve_grad_path(datainf_root, epoch, target_method, train_dataset, t) for t in task_names}
                missing = [p for p in grad_paths.values() if not os.path.isfile(p)]

                if (not os.path.isfile(train_data_path)) or missing:
                    unavailable = write_unavailable_note(
                        os.path.join(combo_dir, f"unavailable_{tag}.json"),
                        reason="required train dataset or gradient cache missing for cross-H",
                        context={
                            "train_dataset": train_dataset,
                            "epoch": epoch,
                            "target_method": target_method,
                            "oracle_method": oracle_method,
                            "train_data_path": train_data_path,
                            "missing_grad_paths": missing,
                        },
                    )
                    rows.append(
                        {
                            "train_dataset": train_dataset,
                            "epoch": epoch,
                            "mode": "crossH",
                            "target_method": target_method,
                            "oracle_method": oracle_method,
                            "status": "unavailable",
                            "reason_file": os.path.abspath(unavailable),
                        }
                    )
                    continue

                run = compute_pairwise_scores_via_cli(
                    datainf_root=datainf_root,
                    output_dir=combo_dir,
                    base_model_path=base_model_path,
                    train_dataset_path=train_data_path,
                    grad_paths=grad_paths,
                    dataset_names=task_names,
                    lora_path=lora_path,
                    damping=args.damping,
                    python_exe=args.python_exe,
                )

                if run.matrix is None:
                    unavailable = write_unavailable_note(
                        os.path.join(combo_dir, f"unavailable_{tag}.json"),
                        reason="cross-H oracle produced no matrix",
                        context={
                            "failed_pairs": run.failed_pairs,
                            "pairwise_dir": run.pairwise_dir,
                        },
                    )
                    rows.append(
                        {
                            "train_dataset": train_dataset,
                            "epoch": epoch,
                            "mode": "crossH",
                            "target_method": target_method,
                            "oracle_method": oracle_method,
                            "status": "failed",
                            "reason_file": os.path.abspath(unavailable),
                            "failed_pairs": len(run.failed_pairs),
                        }
                    )
                    continue

                bundle = save_matrix_bundle(
                    output_dir=combo_dir,
                    tag=tag,
                    K=run.matrix,
                    object_names=task_names,
                    metadata={
                        "mode": "crossH",
                        "train_dataset": train_dataset,
                        "epoch": epoch,
                        "target_method": target_method,
                        "oracle_method": oracle_method,
                        "train_data_path": train_data_path,
                        "lora_path": lora_path,
                        "pairwise_dir": run.pairwise_dir,
                    },
                )
                with open(bundle["summary_json"], "r", encoding="utf-8") as f:
                    summary = json.load(f)
                suite = summary.get("shared_mode_suite", {})
                spec = summary.get("spectral_C", {})

                rows.append(
                    {
                        "train_dataset": train_dataset,
                        "epoch": epoch,
                        "mode": "crossH",
                        "target_method": target_method,
                        "oracle_method": oracle_method,
                        "status": "ok",
                        "summary_json": bundle["summary_json"],
                        "pairwise_dir": run.pairwise_dir,
                        "failed_pairs": len(run.failed_pairs),
                        "lambda1_C": suite.get("lambda1_C"),
                        "lambda1_minus_lambda2_C": suite.get("lambda1_minus_lambda2_C"),
                        "mean_offdiag_C": suite.get("mean_offdiag_C"),
                        "fro_offdiag_C": suite.get("fro_offdiag_C"),
                        "trace_C": suite.get("trace_C"),
                        "eig_min_real_C": spec.get("eig_min_real"),
                        "eig_negative_count_real_C": spec.get("eig_negative_count_real"),
                        "condition_number_C": spec.get("condition_number"),
                    }
                )
                print(bundle["summary_json"])

    summary_csv = os.path.join(output_root, "crossH_summary.csv")
    summary_json = os.path.join(output_root, "crossH_summary.json")
    summary_txt = os.path.join(output_root, "crossH_summary.txt")
    write_rows_csv(summary_csv, rows)
    write_rows_txt(summary_txt, rows, max_cols=18)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(os.path.abspath(summary_csv))
    print(os.path.abspath(summary_json))
    print(os.path.abspath(summary_txt))


if __name__ == "__main__":
    main()
