#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_avg_grad_cosines.py

修改说明：
- 已移除对 CSV 的写入（保留 --csv_name 参数以兼容外部调用，但不再使用）。
- 将每个比较对的详细信息写入 log 文件：包含完整文件路径、加载状态、张量大小、范数、cosine 值或错误说明。
- 以可读的块格式记录每个 pair，方便通过文件路径定位对应的相似度结果。

功能（保留）：
- 支持显式 --base_files / --target_files（笛卡尔积）
- 或者通过 --root / --datasets / --methods / --comparators 用模板生成路径
- 跳过缺失文件并在 log 中记录
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

import torch
import numpy as np


def load_tensor(path: Path) -> Optional[torch.Tensor]:
    """加载张量文件并转换为1D float32格式"""
    if not path.exists():
        return None
    try:
        t = torch.load(str(path), map_location="cpu")
        # make sure float32 and 1d
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = t.float().contiguous().view(-1)
        return t
    except Exception as e:
        print(f"[Warning] Failed to load tensor from {path}: {e}")
        return None


def cosine_similarity_cpu(a: torch.Tensor, b: torch.Tensor) -> float:
    """计算两个张量的余弦相似度，使用双精度确保数值稳定性"""
    a_np = a.to(dtype=torch.float64).cpu().numpy()
    b_np = b.to(dtype=torch.float64).cpu().numpy()
    if a_np.size != b_np.size:
        raise ValueError(f"Shape mismatch for cosine: {a_np.size} vs {b_np.size}")
    na = np.linalg.norm(a_np)
    nb = np.linalg.norm(b_np)
    if na == 0 or nb == 0:
        return float("nan")
    cs = float(np.dot(a_np, b_np) / (na * nb))
    cs = max(min(cs, 1.0), -1.0)
    return cs


def ensure_dir(path: Path):
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)


def generate_comparison_pairs(root: Path, datasets: List[str], methods: List[str],
                            comparators: List[str]) -> List[Tuple[Path, Path, str, str, str]]:
    """
    按照规则生成要比较的 (base_path, target_path, dataset, method, comparator_name) 列表：
      base_path = root/{dataset}/gradient_analysis_{method}/merged/avg_grad.pt
      target_path = root/{dataset}/{comp}/merged/avg_grad.pt   (comp 可能包含 {method}、{dataset})
    """
    pairs = []
    for dataset in datasets:
        for method in methods:
            base_path = root / dataset / f"gradient_analysis_{method}" / "merged" / "avg_grad.pt"
            for comp in comparators:
                comp_name = comp.format(method=method, dataset=dataset)
                target_path = root / dataset / comp_name / "merged" / "avg_grad.pt"
                pairs.append((base_path, target_path, dataset, method, comp_name))
    return pairs


def compute_and_save_cosines(
    pairs: List[Tuple[Path, Path, str, str, str]],
    results_dir: Path,
    csv_name: str = "cosine_results.csv",  # 保留参数以兼容外部调用，但不再使用
    log_name: str = "cosine_results.log",
):
    """计算所有配对的余弦相似度并把详细信息写入 log（不再输出 CSV）"""
    ensure_dir(results_dir)
    log_path = results_dir / log_name

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[Info] Starting computation at {timestamp}")
    print(f"[Info] Total pairs to process: {len(pairs)}")

    with log_path.open("w", encoding="utf-8") as lf:
        # 写入日志头部
        lf.write("Cosine Similarity Computation Log\n")
        lf.write(f"Started at: {timestamp}\n")
        lf.write(f"Total pairs: {len(pairs)}\n")
        lf.write("=" * 80 + "\n\n")

        tensor_cache = {}
        summary = {"total_pairs": len(pairs), "computed": 0, "skipped_missing": 0, "errors": 0}

        for i, (base_path, target_path, dataset, method, comp_name) in enumerate(pairs, 1):
            base_str = str(base_path)
            target_str = str(target_path)
            base_exists = base_path.exists()
            target_exists = target_path.exists()

            # 打印进度到 stdout，便于实时查看
            print(f"[{i}/{len(pairs)}] Processing {dataset}/{method} vs {comp_name}")

            # 日志块开始
            lf.write("-" * 80 + "\n")
            lf.write(f"[{i}/{len(pairs)}] Dataset: {dataset}, Method: {method}, Comparator: {comp_name}\n")
            lf.write(f"  Base path : {base_str}\n")
            lf.write(f"  Target path: {target_str}\n")
            lf.write(f"  Base exists: {base_exists}\n")
            lf.write(f"  Target exists: {target_exists}\n")

            # 缺失文件处理
            if not base_exists or not target_exists:
                lf.write(f"  STATUS: MISSING_FILES\n")
                lf.write("\n")
                summary["skipped_missing"] += 1
                continue

            try:
                # 加载或使用缓存的张量
                if base_str in tensor_cache:
                    base_t = tensor_cache[base_str]
                    lf.write(f"  Base loaded from cache: True\n")
                else:
                    base_t = load_tensor(base_path)
                    lf.write(f"  Base loaded from cache: False\n")
                    lf.write(f"  Base load success: {base_t is not None}\n")
                    if base_t is not None:
                        tensor_cache[base_str] = base_t

                if target_str in tensor_cache:
                    target_t = tensor_cache[target_str]
                    lf.write(f"  Target loaded from cache: True\n")
                else:
                    target_t = load_tensor(target_path)
                    lf.write(f"  Target loaded from cache: False\n")
                    lf.write(f"  Target load success: {target_t is not None}\n")
                    if target_t is not None:
                        tensor_cache[target_str] = target_t

                if base_t is None or target_t is None:
                    lf.write(f"  STATUS: LOAD_FAILED\n")
                    lf.write(f"    Base loaded: {base_t is not None}, Target loaded: {target_t is not None}\n")
                    lf.write("\n")
                    summary["errors"] += 1
                    continue

                # 检查张量形状
                base_numel = base_t.numel()
                target_numel = target_t.numel()
                lf.write(f"  Base tensor numel: {base_numel}\n")
                lf.write(f"  Target tensor numel: {target_numel}\n")

                if base_numel != target_numel:
                    lf.write(f"  STATUS: SHAPE_MISMATCH\n")
                    lf.write(f"    Base shape: {tuple(base_t.shape)} (numel: {base_numel})\n")
                    lf.write(f"    Target shape: {tuple(target_t.shape)} (numel: {target_numel})\n")
                    lf.write("\n")
                    summary["errors"] += 1
                    continue

                # 计算范数以便在日志中记录（可帮助诊断零向量等情况）
                try:
                    base_norm = float(torch.norm(base_t.to(dtype=torch.float64)).item())
                    target_norm = float(torch.norm(target_t.to(dtype=torch.float64)).item())
                except Exception:
                    # 兜底
                    base_norm = float(np.linalg.norm(base_t.to(dtype=torch.float64).cpu().numpy()))
                    target_norm = float(np.linalg.norm(target_t.to(dtype=torch.float64).cpu().numpy()))

                lf.write(f"  Base norm: {base_norm:.8e}\n")
                lf.write(f"  Target norm: {target_norm:.8e}\n")

                if base_norm == 0 or target_norm == 0:
                    # 如果有零范数，cosine 为 NaN
                    lf.write(f"  STATUS: ZERO_NORM -> cosine = NaN\n")
                    lf.write("\n")
                    summary["errors"] += 1
                    continue

                # 计算余弦相似度
                cos_sim = cosine_similarity_cpu(base_t, target_t)
                lf.write(f"  STATUS: SUCCESS\n")
                lf.write(f"    Cosine Similarity: {cos_sim:.8f}\n")
                lf.write("\n")
                summary["computed"] += 1

            except Exception as e:
                lf.write(f"  STATUS: ERROR\n")
                lf.write(f"    Error: {repr(e)}\n")
                lf.write("\n")
                summary["errors"] += 1

        # 写入总结
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lf.write("=" * 80 + "\n")
        lf.write("SUMMARY\n")
        lf.write("=" * 80 + "\n")
        lf.write(f"Completed at: {end_time}\n")
        lf.write(f"Total pairs processed: {summary['total_pairs']}\n")
        lf.write(f"Successfully computed: {summary['computed']}\n")
        lf.write(f"Skipped (missing files): {summary['skipped_missing']}\n")
        lf.write(f"Errors: {summary['errors']}\n")
        success_rate = (summary['computed'] / summary['total_pairs'] * 100) if summary['total_pairs'] > 0 else 0.0
        lf.write(f"Success rate: {success_rate:.1f}%\n")
        lf.write("\n")

    # stdout summary
    print(f"[Info] Log saved to {log_path}")
    print(f"[Info] Summary: {summary['computed']}/{summary['total_pairs']} successful, "
          f"{summary['skipped_missing']} missing, {summary['errors']} errors")


def parse_args():
    """解析命令行参数"""
    ap = argparse.ArgumentParser(description="Compute cosine similarities between avg_grad.pt files.")
    ap.add_argument("--root", type=str, default=None,
                   help="Analysis root directory (用于模板生成)。例如 /.../analysis")
    ap.add_argument("--datasets", type=str, nargs="*",
                   default=["alpaca", "gsm8k", "openfunction", "magicoder", "dolly", "lima", "openhermes"],
                   help="数据集名列表（用于模板生成），例如 alpaca gsm8k")
    ap.add_argument("--methods", type=str, nargs="*", default=["sdft", "sft"],
                   help="Method 列表，例如 sdft sft")
    ap.add_argument(
        "--comparators",
        type=str,
        nargs="*",
        default=[
            "gradient_analysis_{method}_alpacaeval",
            "gradient_analysis_{method}_gsm8ktest",
            "gradient_analysis_{method}_openfunctiontest",
        ],
        help="Comparators 模板列表，可含 {method} 和 {dataset} 占位符",
    )

    ap.add_argument("--base_files", type=str, nargs="*", default=[],
                   help="显式 base 文件路径（互斥于模板生成）。")
    ap.add_argument("--target_files", type=str, nargs="*", default=[],
                   help="显式 target 文件路径（互斥于模板生成）。")
    ap.add_argument("--results_dir", type=str, required=True,
                   help="保存结果的目录（log）。")
    # 保留 csv/log 名参数以兼容旧调用。csv_name 不再被使用/写入。
    ap.add_argument("--csv_name", type=str, default="cosine_results.csv",
                   help="（兼容）CSV 名称，但脚本目前已禁用 CSV 写入。")
    ap.add_argument("--log_name", type=str, default="cosine_results.log")
    return ap.parse_args()


def main():
    """主函数"""
    args = parse_args()

    results_dir = Path(args.results_dir)
    ensure_dir(results_dir)

    pairs = []

    if args.base_files and args.target_files:
        # 如果提供了显式文件列表，计算笛卡尔积
        print("[Info] Using explicit file lists")
        for b in args.base_files:
            for t in args.target_files:
                # comparator 字段用 target 的文件名或 base_vs_target 组合更直观
                comp_label = f"{Path(b).name}__vs__{Path(t).name}"
                pairs.append((Path(b), Path(t), "explicit", "explicit", comp_label))
    else:
        # 使用模板参数生成
        if args.root is None or len(args.datasets) == 0 or len(args.methods) == 0:
            print("[Error] No explicit base/target lists and template generation parameters missing (root/datasets/methods).")
            print("Either provide --base_files and --target_files, or provide --root, --datasets, --methods.")
            return

        root = Path(args.root)
        print(f"[Info] Using template generation with root: {root}")
        print(f"[Info] Datasets: {args.datasets}")
        print(f"[Info] Methods: {args.methods}")
        print(f"[Info] Comparators: {args.comparators}")

        pairs = generate_comparison_pairs(root, args.datasets, args.methods, args.comparators)

        expected_pairs = len(args.datasets) * len(args.methods) * len(args.comparators)
        print(f"[Info] Expected pairs: {expected_pairs}, Generated pairs: {len(pairs)}")

    if not pairs:
        print("[Error] No pairs to process!")
        return

    # 计算并保存结果（仅写 log）
    compute_and_save_cosines(pairs, results_dir, csv_name=args.csv_name, log_name=args.log_name)
    print("[Info] Computation completed!")


if __name__ == "__main__":
    main()
