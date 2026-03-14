#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scheme A - Step 01
梳理现有代码、数据路径和结果缓存映射。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

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
    ensure_dir,
    resolve_existing_result_roots,
    resolve_result_root,
    resolve_sdft_root,
)


def _bool_exists(path: str) -> bool:
    return os.path.exists(path)


def _glob_count(pattern: str) -> int:
    return len(glob.glob(pattern, recursive=True))


def build_mapping(datainf_root: str, extra_result_roots: List[str]) -> Dict[str, object]:
    sdft_root = resolve_sdft_root(datainf_root)
    preferred_result_root = resolve_result_root(datainf_root, prefer_plural=True)
    result_roots = resolve_existing_result_roots(datainf_root, explicit_roots=extra_result_roots)

    key_files = {
        "pairwise_oracle": os.path.join(datainf_root, "src", "calc_dataset_similarity.py"),
        "pairwise_matrix_assembler": os.path.join(datainf_root, "src", "assemble_matrix.py"),
        "pairwise_direct_tagged": os.path.join(datainf_root, "src", "compute_pairwise_from_grads_tagged.py"),
        "pairwise_direct_tagged_local": os.path.join(datainf_root, "src", "compute_pairwise_from_grads_tagged_localresults.py"),
        "corr_analysis": os.path.join(datainf_root, "script", "analyze_pairwise_matrices_write_txt_with_corrs.py"),
        "eig_analysis": os.path.join(datainf_root, "script", "analyze_pairwise_matrices.py"),
        "batch_epoch0": os.path.join(datainf_root, "script", "gpu_scheduler_epoch_0.sh"),
        "batch_epoch1": os.path.join(datainf_root, "script", "gpu_scheduler_epoch_1.sh"),
        "batch_epoch5": os.path.join(datainf_root, "script", "gpu_scheduler_epoch_5.sh"),
        "batch_pairwise_all": os.path.join(datainf_root, "script", "run_pairwise_epoch0_1_5.sh"),
        "save_avg_grad": os.path.join(datainf_root, "src", "save_avg_grad_with_integrated_templates.py"),
        "distill_pair_mapping": os.path.join(sdft_root, "eval", "gen_distilled_data.py"),
        "embedding_hsic": os.path.join(sdft_root, "Mutual-Information", "compute_hsic.py"),
    }

    root_stats: List[Dict[str, object]] = []
    for root in result_roots:
        root_stats.append(
            {
                "root": root,
                "exists": os.path.isdir(root),
                "analysis_log_count": _glob_count(os.path.join(root, "*", "epoch_*", "analysis", "analysis_log.txt")),
                "analysis_corr_count": _glob_count(os.path.join(root, "*", "epoch_*", "analysis_safe", "analysis_corr_safe*.txt")),
                "pairwise_matrix_count": _glob_count(os.path.join(root, "*", "epoch_*", "*", "pairwise_matrix_*.npy")),
            }
        )

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "datainf_root": datainf_root,
        "sdft_root": sdft_root,
        "preferred_result_root": preferred_result_root,
        "result_roots_scanned": result_roots,
        "result_root_stats": root_stats,
        "enumeration": {
            "train_datasets": list(DEFAULT_TRAIN_DATASETS),
            "methods": list(DEFAULT_METHODS),
            "epochs": list(DEFAULT_EPOCHS),
            "task_level_test_tasks": list(DEFAULT_TASKS),
            "task_matrix_shape": "5x5",
        },
        "key_files": key_files,
        "key_files_exist": {k: _bool_exists(v) for k, v in key_files.items()},
        "answers": {
            "q1_ownH_matrix_scripts": [
                "DataInf/src/calc_dataset_similarity.py",
                "DataInf/src/assemble_matrix.py",
                "DataInf/script/analyze_pairwise_matrices.py",
                "DataInf/script/analyze_pairwise_matrices_write_txt_with_corrs.py",
            ],
            "q2_batch_traversal_scripts": [
                "DataInf/script/gpu_scheduler_epoch_0.sh",
                "DataInf/script/gpu_scheduler_epoch_1.sh",
                "DataInf/script/gpu_scheduler_epoch_5.sh",
                "DataInf/script/run_pairwise_epoch0_1_5.sh",
            ],
            "q3_grad_oracle_reading_paths": [
                "DataInf/output_grads/<epoch>/<method>/<train_dataset>/<task>.pt",
                "DataInf/output_grad/<epoch>/<method>/<train_dataset>/<task>.pt",
                "DataInf/src/result/output_grad/<epoch>/<method>/<train_dataset>/<task>.pt",
            ],
            "q4_matrix_output_paths": [
                "DataInf/result/<train_dataset>/<epoch>/<method>/pairwise_matrix_*.npy",
                "DataInf/results/<train_dataset>/<epoch>/<method>/pairwise_matrix_*.npy",
                "DataInf/result/<train_dataset>/<epoch>/analysis/analysis_log.txt",
                "DataInf/result/<train_dataset>/<epoch>/analysis_safe/analysis_corr_safe*.txt",
            ],
            "q5_eig_and_diff_scripts": [
                "DataInf/script/analyze_pairwise_matrices.py",
                "DataInf/script/aggregate_sft_sdft_diffs.sh",
                "DataInf/script/extract_top3_eigs_from_diffs.sh",
            ],
            "q6_cli_style": [
                "--train_dataset / --all_train_datasets",
                "--epoch / --all_epochs",
                "--method / --target_method",
                "--task_names (默认固定5个测试任务)",
            ],
        },
    }


def write_reports(mapping: Dict[str, object], out_dir: str) -> Dict[str, str]:
    ensure_dir(out_dir)
    json_path = os.path.join(out_dir, "schemeA_inventory_mapping.json")
    md_path = os.path.join(out_dir, "schemeA_inventory_mapping.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    lines: List[str] = []
    lines.append("# Scheme A 代码与路径梳理")
    lines.append("")
    lines.append(f"- 生成时间: {mapping['timestamp']}")
    lines.append(f"- DataInf 根目录: `{mapping['datainf_root']}`")
    lines.append(f"- SDFT 根目录: `{mapping['sdft_root']}`")
    lines.append(f"- 结果目录优先级: `{mapping['preferred_result_root']}`")
    lines.append("")
    lines.append("## 覆盖维度")
    enum = mapping["enumeration"]
    lines.append(f"- 训练集: `{','.join(enum['train_datasets'])}`")
    lines.append(f"- 方法: `{','.join(enum['methods'])}`")
    lines.append(f"- Epoch: `{','.join(enum['epochs'])}`")
    lines.append(f"- Task-level 测试任务: `{','.join(enum['task_level_test_tasks'])}`")
    lines.append(f"- Task-level 矩阵形状: `{enum['task_matrix_shape']}`")
    lines.append("")
    lines.append("## 结果根目录扫描")
    for item in mapping["result_root_stats"]:
        lines.append(
            f"- `{item['root']}` | exists={item['exists']} | analysis_log={item['analysis_log_count']} | analysis_corr={item['analysis_corr_count']} | pairwise_matrix={item['pairwise_matrix_count']}"
        )
    lines.append("")
    lines.append("## 定位结论（对应你的6个问题）")
    for key, vals in mapping["answers"].items():
        lines.append(f"- `{key}`")
        for v in vals:
            lines.append(f"  - `{v}`")
    lines.append("")
    lines.append("## 关键脚本存在性")
    for k, p in mapping["key_files"].items():
        ok = mapping["key_files_exist"].get(k, False)
        lines.append(f"- `{k}`: `{p}` [{'存在' if ok else '缺失'}]")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return {"json": json_path, "md": md_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheme A inventory and path mapping")
    parser.add_argument("--datainf_root", type=str, default=None)
    parser.add_argument("--extra_result_roots", type=str, default="")
    parser.add_argument("--output_root", type=str, default=None)
    args = parser.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    output_root = args.output_root or os.path.join(result_root, "schemeA", "inventory")
    extra_result_roots = [x.strip() for x in args.extra_result_roots.split(",") if x.strip()]

    mapping = build_mapping(datainf_root, extra_result_roots)
    out = write_reports(mapping, output_root)
    print(os.path.abspath(out["json"]))
    print(os.path.abspath(out["md"]))


if __name__ == "__main__":
    main()
