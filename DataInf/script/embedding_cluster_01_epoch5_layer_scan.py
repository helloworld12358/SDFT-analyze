#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import pandas as pd
import torch
from transformers import AutoTokenizer

from embedding_cluster_utils import (
    DEFAULT_TRAIN_DATASETS,
    TASKS_5,
    build_label_arrays,
    build_model_specs_epoch5,
    compute_cluster_metrics,
    extract_last_token_representations,
    init_runtime_paths,
    load_model_with_optional_lora,
    parse_layers_spec,
    resolve_eval_dataset_paths,
    resolve_model_paths_for_spec,
    sample_prompt_pool,
    split_csv_arg,
    infer_layer_count,
    should_clear_cuda_cache,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Stage1: epoch5 layer scan for clustering metrics.")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default="")
    p.add_argument("--base_model_path", type=str, default="")
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--train_dataset", type=str, default="")
    p.add_argument("--all_train_datasets", action="store_true")
    p.add_argument("--include_base", action="store_true")
    p.add_argument("--tasks", type=str, default="alpaca_eval,gsm8k,humaneval,multiarith,openfunction")
    p.add_argument("--samples_per_task", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layers", type=str, default="all", help="all or e.g. 4,8,12-20,32")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--device", type=str, default="", help="default: auto if cuda available else cpu")
    p.add_argument("--prefer_auto_on_fail", action="store_true")
    args = p.parse_args()

    rt = init_runtime_paths(args.datainf_root, args.output_root, args.base_model_path, args.data_root)
    output_root = rt["output_root"]
    sdft_root = rt["sdft_root"]
    base_model_path = rt["base_model_path"]

    tasks = split_csv_arg(args.tasks, TASKS_5)
    if args.all_train_datasets:
        train_datasets = list(DEFAULT_TRAIN_DATASETS)
    else:
        td = args.train_dataset.strip()
        train_datasets = [td] if td else list(DEFAULT_TRAIN_DATASETS)

    device = args.device.strip() or ("auto" if torch.cuda.is_available() else "cpu")
    batch_size = max(1, int(args.batch_size))
    max_length = int(args.max_length)
    samples_per_task = max(1, int(args.samples_per_task))
    seed = int(args.seed)

    eval_paths = resolve_eval_dataset_paths(rt["data_root"])
    sample_rows, sample_stats = sample_prompt_pool(
        eval_paths=eval_paths,
        tasks=tasks,
        per_task=samples_per_task,
        seed=seed,
    )
    if not sample_rows:
        raise RuntimeError("sample pool is empty; check test dataset files")

    pool_dir = os.path.join(output_root, "sample_pool")
    os.makedirs(pool_dir, exist_ok=True)
    sample_pool_json = os.path.join(pool_dir, f"sample_pool_seed{seed}_n{samples_per_task}_5tasks.json")
    with open(sample_pool_json, "w", encoding="utf-8") as f:
        json.dump({"rows": sample_rows, "stats": sample_stats}, f, ensure_ascii=False, indent=2)

    y_int, _, _ = build_label_arrays(sample_rows, tasks)
    texts = [str(x["text"]) for x in sample_rows]

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    stage_dir = os.path.join(output_root, "stage1", "by_train_dataset")
    os.makedirs(stage_dir, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    unavailable: List[Dict[str, object]] = []

    # infer layer count once from base model
    base_model, base_err = load_model_with_optional_lora(
        base_model_path=base_model_path,
        lora_path=None,
        device=device,
        prefer_auto_on_fail=bool(args.prefer_auto_on_fail),
    )
    if base_model is None:
        raise RuntimeError(f"failed to load base model for layer count inference: {base_err}")
    n_layers = infer_layer_count(base_model, tokenizer, device=device)
    del base_model
    if should_clear_cuda_cache(device):
        torch.cuda.empty_cache()
    layers = parse_layers_spec(args.layers, n_layers)

    for train_dataset in train_datasets:
        model_specs = build_model_specs_epoch5(train_dataset=train_dataset, include_base=bool(args.include_base))
        ds_rows: List[Dict[str, object]] = []
        for spec in model_specs:
            model_tag = str(spec["model_tag"])
            base_path, lora_path, err_path = resolve_model_paths_for_spec(sdft_root, base_model_path, spec)
            if err_path:
                row = {
                    "train_dataset_shard": train_dataset,
                    "model_tag": model_tag,
                    "train_dataset": str(spec["train_dataset"]),
                    "method": str(spec["method"]),
                    "epoch": str(spec["epoch"]),
                    "layer": None,
                    "n_samples": len(sample_rows),
                    "status": "missing_checkpoint",
                    "reason": err_path,
                    "base_model_path": base_path,
                    "lora_path": lora_path or "",
                    "silhouette": None,
                    "davies_bouldin": None,
                    "calinski_harabasz": None,
                    "knn_purity": None,
                }
                ds_rows.append(row)
                unavailable.append(dict(row))
                continue

            model, load_err = load_model_with_optional_lora(
                base_model_path=base_path,
                lora_path=lora_path,
                device=device,
                prefer_auto_on_fail=bool(args.prefer_auto_on_fail),
            )
            if model is None:
                row = {
                    "train_dataset_shard": train_dataset,
                    "model_tag": model_tag,
                    "train_dataset": str(spec["train_dataset"]),
                    "method": str(spec["method"]),
                    "epoch": str(spec["epoch"]),
                    "layer": None,
                    "n_samples": len(sample_rows),
                    "status": "model_load_error",
                    "reason": load_err or "",
                    "base_model_path": base_path,
                    "lora_path": lora_path or "",
                    "silhouette": None,
                    "davies_bouldin": None,
                    "calinski_harabasz": None,
                    "knn_purity": None,
                }
                ds_rows.append(row)
                unavailable.append(dict(row))
                continue

            reps = extract_last_token_representations(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                layers=layers,
                batch_size=batch_size,
                device=device,
                max_length=max_length,
            )
            del model
            if should_clear_cuda_cache(device):
                torch.cuda.empty_cache()

            for layer in layers:
                x = reps.get(layer, None)
                if x is None or x.size == 0:
                    row = {
                        "train_dataset_shard": train_dataset,
                        "model_tag": model_tag,
                        "train_dataset": str(spec["train_dataset"]),
                        "method": str(spec["method"]),
                        "epoch": str(spec["epoch"]),
                        "layer": int(layer),
                        "n_samples": len(sample_rows),
                        "status": "empty_representation",
                        "reason": "layer representation is empty",
                        "base_model_path": base_path,
                        "lora_path": lora_path or "",
                        "silhouette": None,
                        "davies_bouldin": None,
                        "calinski_harabasz": None,
                        "knn_purity": None,
                    }
                    ds_rows.append(row)
                    unavailable.append(dict(row))
                    continue
                metrics = compute_cluster_metrics(x, y_int)
                row = {
                    "train_dataset_shard": train_dataset,
                    "model_tag": model_tag,
                    "train_dataset": str(spec["train_dataset"]),
                    "method": str(spec["method"]),
                    "epoch": str(spec["epoch"]),
                    "layer": int(layer),
                    "n_samples": int(x.shape[0]),
                    "status": "ok",
                    "reason": "",
                    "base_model_path": base_path,
                    "lora_path": lora_path or "",
                    "silhouette": metrics["silhouette"],
                    "davies_bouldin": metrics["davies_bouldin"],
                    "calinski_harabasz": metrics["calinski_harabasz"],
                    "knn_purity": metrics["knn_purity"],
                }
                ds_rows.append(row)

        ds_dir = os.path.join(stage_dir, train_dataset)
        os.makedirs(ds_dir, exist_ok=True)
        ds_csv = os.path.join(ds_dir, f"epoch5_layer_scan_{train_dataset}.csv")
        ds_json = os.path.join(ds_dir, f"epoch5_layer_scan_{train_dataset}.json")
        pd.DataFrame(ds_rows).to_csv(ds_csv, index=False)
        with open(ds_json, "w", encoding="utf-8") as f:
            json.dump(ds_rows, f, ensure_ascii=False, indent=2)
        all_rows.extend(ds_rows)
        print(os.path.abspath(ds_csv))
        print(os.path.abspath(ds_json))

    all_csv = os.path.join(output_root, "stage1", "epoch5_layer_scan_all.csv")
    all_json = os.path.join(output_root, "stage1", "epoch5_layer_scan_all.json")
    os.makedirs(os.path.dirname(all_csv), exist_ok=True)
    pd.DataFrame(all_rows).to_csv(all_csv, index=False)
    with open(all_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    unavail_json = os.path.join(output_root, "stage1", "unavailable_epoch5_layer_scan.json")
    with open(unavail_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "unavailable_count": len(unavailable),
                "unavailable": unavailable,
                "sample_pool_json": os.path.abspath(sample_pool_json),
                "sample_stats": sample_stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(os.path.abspath(all_csv))
    print(os.path.abspath(all_json))
    print(os.path.abspath(unavail_json))


if __name__ == "__main__":
    main()
