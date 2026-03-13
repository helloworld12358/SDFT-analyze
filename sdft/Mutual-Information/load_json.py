#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from compute_hsic import calculate_hsic_from_json

# 项目根目录（根据你的环境调整）
PROJECT_ROOT = "/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft"

# predictions 根目录（绝对路径）
PREDICTIONS_ROOT = os.path.join(PROJECT_ROOT, "predictions")

def create_results_directory():
    """创建并返回绝对的 results 目录路径"""
    results_dir = os.path.join(PROJECT_ROOT, "Mutual-Information", "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def get_output_filename(json_file_path):
    """根据文件路径生成输出文件名"""
    # 规范化并拆分路径
    parts = os.path.normpath(json_file_path).replace('\\', '/').split('/')
    # 提取数据集名称
    dataset_name = next(
        (p for p in parts if p in [
            'alpaca','dolly','gsm8k','lima',
            'openfunction','magicoder','openhermes'
        ]),
        ""
    )
    # 反向遍历，优先匹配最后出现的 method（sdft 或 sft）
    training_method = next(
        (p for p in reversed(parts) if p in ['sdft','sft']),
        ""
    )
    if dataset_name and training_method:
        return f"{dataset_name}_{training_method}_hsic.log"
    # 兜底：使用原文件名
    base = os.path.splitext(os.path.basename(json_file_path))[0]
    return f"{base}_hsic.log"

def save_hsic_result(json_file_path, hsic_value, kernel_type, results_dir):
    """保存单个文件的 HSIC 结果到 log"""
    output_filename = get_output_filename(json_file_path)
    output_path = os.path.join(results_dir, output_filename)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "SUCCESS" if hsic_value is not None else "FAILED"
    hsic_str = f"{hsic_value:.6f}" if hsic_value is not None else "N/A"

    log_content = (
        f"File: {json_file_path}\n"
        f"Time: {timestamp}\n"
        f"Kernel: {kernel_type}\n"
        f"HSIC: {hsic_str}\n"
        f"Status: {status}\n"
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(log_content)
    return output_path

def generate_file_paths():
    """批量生成所有目标 JSONL 绝对路径"""
    datasets = ['alpaca', 'dolly', 'gsm8k', 'lima', 'openfunction', 'magicoder', 'openhermes']
    methods  = ['sdft', 'sft']
    paths = []
    for ds in datasets:
        for md in methods:
            fp = os.path.join(
                PREDICTIONS_ROOT,
                ds,
                md,
                "alpaca_eval",
                "generated_predictions.jsonl"
            )
            paths.append(fp)
    return paths

def batch_calculate_hsic(
    file_paths,
    device='cpu',
    batch_size=16,
    ktype='gaussian',
    verbose=False,
    save_results=True
):
    results_dir = create_results_directory() if save_results else None
    summary = []
    total = len(file_paths)
    print(f"[batch] 一共 {total} 个文件")

    for idx, fp in enumerate(file_paths, 1):
        print(f"[{idx}/{total}] {fp}")
        if not os.path.exists(fp):
            print("  → 文件不存在，跳过")
            summary.append((fp, None, 'missing'))
            continue
        try:
            val = calculate_hsic_from_json(
                json_file_path=fp,
                device=device,
                batch_size=batch_size,
                ktype=ktype,
                verbose=verbose
            )
            print(f"  → HSIC = {val:.6f}")
            if save_results:
                save_hsic_result(fp, val, ktype, results_dir)
            summary.append((fp, val, 'success'))
        except Exception as e:
            print(f"  → Error: {e}")
            if save_results:
                save_hsic_result(fp, None, ktype, results_dir)
            summary.append((fp, None, 'failed'))

    # 汇总
    print("\n=== 计算汇总 ===")
    succ = sum(1 for _,_,st in summary if st=='success')
    for fp, val, st in summary:
        tag = 'OK' if st=='success' else ('MISSING' if st=='missing' else 'FAIL')
        vs  = f"{val:.6f}" if val is not None else '--'
        print(f"{fp:<80} {tag:>7} {vs}")
    print(f"\n成功 {succ}/{total}\n")

def main():
    # 配置
    device     = 'cuda'   # 'cpu'或 'cuda'
    batch_size = 16
    ktype      = 'gaussian'
    verbose    = False
    save_res   = True

    # 批量模式，使用绝对路径列表
    fps = generate_file_paths()
    batch_calculate_hsic(
        file_paths=fps,
        device=device,
        batch_size=batch_size,
        ktype=ktype,
        verbose=verbose,
        save_results=save_res
    )

if __name__ == '__main__':
    main()
