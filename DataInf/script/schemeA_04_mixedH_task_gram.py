#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scheme A - Step 04
Task-level mixed/common-H Gram experiments.

仅实现一种 common-H：
H_mix 由同一训练集下的 SFT 数据与 SDFT 数据直接拼接得到。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

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
    merge_records_for_mixed_h,
    normalize_epoch_list,
    normalize_method_list,
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


def _pick_methods(args) -> List[str]:
    return normalize_method_list(split_csv_arg(args.method, ["both"])) or ["sft", "sdft"]


def main() -> None:
    p = argparse.ArgumentParser(description="Scheme A mixed/common-H task-level Gram")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--train_dataset", type=str, default="")
    p.add_argument("--all_train_datasets", action="store_true")
    p.add_argument("--epoch", type=str, default="")
    p.add_argument("--all_epochs", action="store_true")
    p.add_argument("--method", type=str, default="both", help="sft/sdft/both")
    p.add_argument("--task_names", type=str, default=",".join(DEFAULT_TASKS))
    p.add_argument("--base_model_path", type=str, default=None)
    p.add_argument("--damping", type=float, default=0.001)
    p.add_argument("--python_exe", type=str, default=sys.executable)
    p.add_argument("--output_root", type=str, default=None)
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    sdft_root = resolve_sdft_root(datainf_root)
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    output_root = args.output_root or os.path.join(result_root, "schemeA", "mixedH")

    train_datasets = _pick_train_datasets(args)
    epochs = _pick_epochs(args)
    methods = _pick_methods(args)
    task_names = split_csv_arg(args.task_names, DEFAULT_TASKS)
    base_model_path = args.base_model_path or os.path.join(sdft_root, "model", "Llama-2-7b-chat-hf")

    rows: List[Dict[str, object]] = []

    for train_dataset in train_datasets:
        sft_train = resolve_train_dataset_path(sdft_root, train_dataset, "sft")
        sdft_distilled = resolve_train_dataset_path(sdft_root, train_dataset, "sdft")

        for epoch in epochs:
            mixed_cache = os.path.join(
                output_root,
                "_cache_mixed_train",
                train_dataset,
                epoch,
                f"mixed_{train_dataset}_sft_plus_sdft.json",
            )
            mixed_train_path, merge_err = merge_records_for_mixed_h(
                sft_train_path=sft_train,
                sdft_distilled_path=sdft_distilled,
                out_path=mixed_cache,
            )

            if merge_err or (not mixed_train_path) or (not os.path.isfile(mixed_train_path)):
                for method in methods:
                    combo_dir = os.path.join(output_root, train_dataset, epoch, f"target_{method}")
                    tag = f"{train_dataset}_{epoch}_{method}_mixedH"
                    unavailable = write_unavailable_note(
                        os.path.join(combo_dir, f"unavailable_{tag}.json"),
                        reason=f"mixed/common-H cannot be built: {merge_err or 'missing mixed dataset file'}",
                        context={
                            "train_dataset": train_dataset,
                            "epoch": epoch,
                            "target_method": method,
                            "sft_train_path": sft_train,
                            "sdft_distilled_path": sdft_distilled,
                            "expected_mixed_cache": mixed_cache,
                        },
                    )
                    rows.append(
                        {
                            "train_dataset": train_dataset,
                            "epoch": epoch,
                            "mode": "mixedH",
                            "target_method": method,
                            "status": "unavailable",
                            "reason_file": os.path.abspath(unavailable),
                        }
                    )
                continue

            for method in methods:
                combo_dir = os.path.join(output_root, train_dataset, epoch, f"target_{method}")
                tag = f"{train_dataset}_{epoch}_{method}_mixedH"
                grad_paths = {t: resolve_grad_path(datainf_root, epoch, method, train_dataset, t) for t in task_names}
                missing = [p for p in grad_paths.values() if not os.path.isfile(p)]

                if missing:
                    unavailable = write_unavailable_note(
                        os.path.join(combo_dir, f"unavailable_{tag}.json"),
                        reason="required gradient cache missing for mixed/common-H",
                        context={
                            "train_dataset": train_dataset,
                            "epoch": epoch,
                            "target_method": method,
                            "mixed_train_path": mixed_train_path,
                            "missing_grad_paths": missing,
                        },
                    )
                    rows.append(
                        {
                            "train_dataset": train_dataset,
                            "epoch": epoch,
                            "mode": "mixedH",
                            "target_method": method,
                            "status": "unavailable",
                            "reason_file": os.path.abspath(unavailable),
                        }
                    )
                    continue

                run = compute_pairwise_scores_via_cli(
                    datainf_root=datainf_root,
                    output_dir=combo_dir,
                    base_model_path=base_model_path,
                    train_dataset_path=mixed_train_path,
                    grad_paths=grad_paths,
                    dataset_names=task_names,
                    lora_path=None,
                    damping=args.damping,
                    python_exe=args.python_exe,
                )

                if run.matrix is None:
                    unavailable = write_unavailable_note(
                        os.path.join(combo_dir, f"unavailable_{tag}.json"),
                        reason="mixed/common-H oracle produced no matrix",
                        context={"failed_pairs": run.failed_pairs, "pairwise_dir": run.pairwise_dir},
                    )
                    rows.append(
                        {
                            "train_dataset": train_dataset,
                            "epoch": epoch,
                            "mode": "mixedH",
                            "target_method": method,
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
                        "mode": "mixedH",
                        "train_dataset": train_dataset,
                        "epoch": epoch,
                        "target_method": method,
                        "common_h_definition": "concat(sft_train, sdft_distilled)",
                        "mixed_train_path": mixed_train_path,
                        "oracle_lora_path": None,
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
                        "mode": "mixedH",
                        "target_method": method,
                        "status": "ok",
                        "summary_json": bundle["summary_json"],
                        "pairwise_dir": run.pairwise_dir,
                        "mixed_train_path": mixed_train_path,
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

    summary_csv = os.path.join(output_root, "mixedH_summary.csv")
    summary_json = os.path.join(output_root, "mixedH_summary.json")
    summary_txt = os.path.join(output_root, "mixedH_summary.txt")
    write_rows_csv(summary_csv, rows)
    write_rows_txt(summary_txt, rows, max_cols=18)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(os.path.abspath(summary_csv))
    print(os.path.abspath(summary_json))
    print(os.path.abspath(summary_txt))


if __name__ == "__main__":
    main()
