#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scheme A - Step 02
A1 的职责：优先收编已有 own-H 5x5 矩阵，不默认全量重算。
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
    DEFAULT_TASKS,
    DEFAULT_TRAIN_DATASETS,
    compute_pairwise_scores_via_cli,
    detect_datainf_root,
    load_existing_ownh_from_analysis,
    load_pairwise_matrix_any,
    normalize_epoch_list,
    normalize_method_list,
    resolve_checkpoint_path,
    resolve_existing_result_roots,
    resolve_grad_path,
    resolve_result_root,
    resolve_sdft_root,
    resolve_train_dataset_path,
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


def _shape_ok(mat: Optional[np.ndarray], n: int) -> bool:
    return mat is not None and isinstance(mat, np.ndarray) and mat.shape == (n, n)


def _attempt_recompute_ownh(
    datainf_root: str,
    sdft_root: str,
    train_dataset: str,
    epoch: str,
    method: str,
    task_names: List[str],
    combo_dir: str,
    base_model_path: str,
    damping: float,
    python_exe: str,
) -> Tuple[Optional[np.ndarray], Dict[str, object], str]:
    train_data_path = resolve_train_dataset_path(sdft_root, train_dataset, method)
    lora_path = resolve_checkpoint_path(sdft_root, epoch, train_dataset, method)
    grad_paths = {t: resolve_grad_path(datainf_root, epoch, method, train_dataset, t) for t in task_names}
    missing_grad = [p for p in grad_paths.values() if not os.path.isfile(p)]

    context = {
        "train_data_path": train_data_path,
        "lora_path": lora_path,
        "missing_grad_paths": missing_grad,
    }
    if not os.path.isfile(train_data_path) or missing_grad:
        return None, context, "required train dataset or gradient cache missing"

    run = compute_pairwise_scores_via_cli(
        datainf_root=datainf_root,
        output_dir=os.path.join(combo_dir, "_recompute_pairwise"),
        base_model_path=base_model_path,
        train_dataset_path=train_data_path,
        grad_paths=grad_paths,
        dataset_names=task_names,
        lora_path=lora_path,
        damping=damping,
        python_exe=python_exe,
    )
    context["pairwise_dir"] = run.pairwise_dir
    context["failed_pairs"] = run.failed_pairs
    if run.matrix is None:
        return None, context, "pairwise oracle produced no matrix"
    return run.matrix, context, "ok"


def main() -> None:
    p = argparse.ArgumentParser(description="Scheme A own-H collection (prefer existing analysis outputs)")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--train_dataset", type=str, default="")
    p.add_argument("--all_train_datasets", action="store_true")
    p.add_argument("--epoch", type=str, default="")
    p.add_argument("--all_epochs", action="store_true")
    p.add_argument("--method", type=str, default="both", help="sft/sdft/both")
    p.add_argument("--task_names", type=str, default=",".join(DEFAULT_TASKS))
    p.add_argument(
        "--existing_result_roots",
        type=str,
        default="",
        help="逗号分隔，可传入 DataInf/result 根目录；优先从这些目录读取 analysis/analysis_safe",
    )
    p.add_argument("--allow_recompute_missing", action="store_true")
    p.add_argument("--base_model_path", type=str, default=None)
    p.add_argument("--damping", type=float, default=0.001)
    p.add_argument("--python_exe", type=str, default=sys.executable)
    p.add_argument("--output_root", type=str, default=None)
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    sdft_root = resolve_sdft_root(datainf_root)
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    output_root = args.output_root or os.path.join(result_root, "schemeA", "ownH")

    train_datasets = _pick_train_datasets(args)
    epochs = _pick_epochs(args)
    methods = _pick_methods(args)
    task_names = split_csv_arg(args.task_names, DEFAULT_TASKS)
    extra_roots = [x.strip() for x in args.existing_result_roots.split(",") if x.strip()]
    result_roots = resolve_existing_result_roots(datainf_root, explicit_roots=extra_roots)

    base_model_path = args.base_model_path or os.path.join(sdft_root, "model", "Llama-2-7b-chat-hf")

    rows: List[Dict[str, object]] = []
    source_counts: Dict[str, int] = {}

    for train_dataset in train_datasets:
        for epoch in epochs:
            existing = load_existing_ownh_from_analysis(result_roots, train_dataset, epoch)
            collected: Dict[str, Dict[str, object]] = {}

            for method in methods:
                combo_dir = os.path.join(output_root, train_dataset, epoch, method)
                tag = f"{train_dataset}_{epoch}_{method}_ownH"

                t_mat = existing.get(method, {}).get("T")
                c_mat = existing.get(method, {}).get("C")
                source_type = None
                source_detail: Dict[str, object] = {
                    "analysis_log_path": existing.get(method, {}).get("analysis_log_path"),
                    "analysis_corr_path": existing.get(method, {}).get("analysis_corr_path"),
                    "source_root": existing.get(method, {}).get("source_root"),
                }

                if _shape_ok(t_mat, len(task_names)):
                    source_type = "existing_analysis_log"
                else:
                    t_mat = load_pairwise_matrix_any(
                        datainf_root=datainf_root,
                        model=train_dataset,
                        epoch=epoch,
                        method=method,
                        names=task_names,
                        extra_result_roots=result_roots,
                    )
                    if _shape_ok(t_mat, len(task_names)):
                        source_type = "existing_pairwise_matrix"

                if source_type is None and args.allow_recompute_missing:
                    recomputed, ctx, msg = _attempt_recompute_ownh(
                        datainf_root=datainf_root,
                        sdft_root=sdft_root,
                        train_dataset=train_dataset,
                        epoch=epoch,
                        method=method,
                        task_names=task_names,
                        combo_dir=combo_dir,
                        base_model_path=base_model_path,
                        damping=args.damping,
                        python_exe=args.python_exe,
                    )
                    source_detail["recompute_context"] = ctx
                    if _shape_ok(recomputed, len(task_names)):
                        t_mat = recomputed
                        source_type = "recomputed_missing_only"
                    else:
                        source_detail["recompute_message"] = msg

                if source_type is None or not _shape_ok(t_mat, len(task_names)):
                    unavailable = write_unavailable_note(
                        os.path.join(combo_dir, f"unavailable_{tag}.json"),
                        reason="own-H matrix missing and cannot be collected",
                        context={
                            "train_dataset": train_dataset,
                            "epoch": epoch,
                            "method": method,
                            "task_names": task_names,
                            "allow_recompute_missing": bool(args.allow_recompute_missing),
                            "result_roots_scanned": result_roots,
                            **source_detail,
                        },
                    )
                    rows.append(
                        {
                            "train_dataset": train_dataset,
                            "epoch": epoch,
                            "method": method,
                            "mode": "ownH",
                            "status": "unavailable",
                            "reason_file": os.path.abspath(unavailable),
                        }
                    )
                    continue

                c_override = c_mat if _shape_ok(c_mat, len(task_names)) else None
                bundle = save_matrix_bundle(
                    output_dir=combo_dir,
                    tag=tag,
                    K=t_mat,  # type: ignore[arg-type]
                    object_names=task_names,
                    metadata={
                        "mode": "ownH",
                        "train_dataset": train_dataset,
                        "epoch": epoch,
                        "method": method,
                        "source_type": source_type,
                        **source_detail,
                    },
                    C_override=c_override,  # keep existing corr if available
                )

                with open(bundle["summary_json"], "r", encoding="utf-8") as f:
                    summary = json.load(f)
                suite = summary.get("shared_mode_suite", {})
                spec = summary.get("spectral_C", {})

                rows.append(
                    {
                        "train_dataset": train_dataset,
                        "epoch": epoch,
                        "method": method,
                        "mode": "ownH",
                        "status": "ok",
                        "source_type": source_type,
                        "summary_json": bundle["summary_json"],
                        "T_npy": bundle.get("T_npy"),
                        "C_npy": bundle.get("C_npy"),
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
                source_counts[source_type] = source_counts.get(source_type, 0) + 1
                collected[method] = {"T": np.asarray(t_mat), "C": np.asarray(c_override) if c_override is not None else None}
                print(bundle["summary_json"])

            if "sft" in collected and "sdft" in collected:
                diff_dir = os.path.join(output_root, train_dataset, epoch, "compare_sft_minus_sdft")
                diff_tag = f"{train_dataset}_{epoch}_sft_minus_sdft_ownH_diff"
                diff_t = collected["sft"]["T"] - collected["sdft"]["T"]
                diff_c = None
                if collected["sft"].get("C") is not None and collected["sdft"].get("C") is not None:
                    diff_c = collected["sft"]["C"] - collected["sdft"]["C"]
                bundle = save_matrix_bundle(
                    output_dir=diff_dir,
                    tag=diff_tag,
                    K=diff_t,
                    object_names=task_names,
                    metadata={
                        "mode": "ownH_diff",
                        "diff_definition": "sft_minus_sdft",
                        "train_dataset": train_dataset,
                        "epoch": epoch,
                    },
                    C_override=diff_c,
                )
                rows.append(
                    {
                        "train_dataset": train_dataset,
                        "epoch": epoch,
                        "method": "sft_minus_sdft",
                        "mode": "ownH_diff",
                        "status": "ok",
                        "summary_json": bundle["summary_json"],
                    }
                )

    summary_csv = os.path.join(output_root, "ownH_summary.csv")
    summary_json = os.path.join(output_root, "ownH_summary.json")
    summary_txt = os.path.join(output_root, "ownH_summary.txt")
    collect_json = os.path.join(output_root, "ownH_collection_summary.json")

    write_rows_csv(summary_csv, rows)
    write_rows_txt(summary_txt, rows, max_cols=18)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(collect_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "result_roots_scanned": result_roots,
                "source_counts": source_counts,
                "allow_recompute_missing": bool(args.allow_recompute_missing),
                "task_names": task_names,
                "train_datasets": train_datasets,
                "epochs": epochs,
                "methods": methods,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(os.path.abspath(summary_csv))
    print(os.path.abspath(summary_json))
    print(os.path.abspath(summary_txt))
    print(os.path.abspath(collect_json))


if __name__ == "__main__":
    main()
