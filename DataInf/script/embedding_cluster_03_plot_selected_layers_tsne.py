#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoTokenizer

from embedding_cluster_utils import (
    DEFAULT_TRAIN_DATASETS,
    TASKS_5,
    build_label_arrays,
    build_model_specs_story,
    compute_cluster_metrics,
    extract_last_token_representations,
    init_runtime_paths,
    infer_layer_count,
    load_model_with_optional_lora,
    parse_layers_spec,
    resolve_eval_dataset_paths,
    resolve_model_paths_for_spec,
    sample_prompt_pool,
    should_clear_cuda_cache,
    split_csv_arg,
)


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(s))


def load_or_build_sample_pool(
    output_root: str,
    eval_paths: Dict[str, str],
    tasks: Sequence[str],
    samples_per_task: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], str]:
    pool_dir = os.path.join(output_root, "sample_pool")
    os.makedirs(pool_dir, exist_ok=True)
    pool_json = os.path.join(pool_dir, f"sample_pool_seed{seed}_n{samples_per_task}_5tasks.json")

    if os.path.isfile(pool_json):
        try:
            obj = json.load(open(pool_json, "r", encoding="utf-8"))
            rows = obj.get("rows", [])
            stats = obj.get("stats", [])
            if isinstance(rows, list) and rows:
                return rows, stats if isinstance(stats, list) else [], pool_json
        except Exception:
            pass

    rows, stats = sample_prompt_pool(
        eval_paths=eval_paths,
        tasks=tasks,
        per_task=samples_per_task,
        seed=seed,
    )
    with open(pool_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "stats": stats}, f, ensure_ascii=False, indent=2)
    return rows, stats, pool_json


def load_recommended_layers(output_root: str, top_k: int) -> List[int]:
    p = os.path.join(output_root, "stage2", "recommended_layers.json")
    if not os.path.isfile(p):
        return []
    try:
        obj = json.load(open(p, "r", encoding="utf-8"))
        arr = obj.get("recommended_layers", [])
        out = [int(x) for x in arr if str(x).strip()]
        return out[: max(1, top_k)]
    except Exception:
        return []


def resolve_selected_layers(
    layers_arg: str,
    output_root: str,
    top_k: int,
    n_layers: int,
) -> List[int]:
    s = (layers_arg or "").strip()
    if s:
        return parse_layers_spec(s, n_layers)
    rec = load_recommended_layers(output_root, top_k=top_k)
    if rec:
        return sorted(set(int(x) for x in rec if 1 <= int(x) <= n_layers))
    return [n_layers]


def reduce_for_tsne(x: np.ndarray, pca_dim: int, seed: int) -> np.ndarray:
    if x.ndim != 2:
        return x
    n, d = x.shape
    k = int(min(max(2, pca_dim), max(2, n - 1), d))
    if d <= k:
        return x
    pca = PCA(n_components=k, random_state=seed)
    return pca.fit_transform(x)


def run_tsne_2d(
    x: np.ndarray,
    seed: int,
    perplexity: float,
    n_iter: int,
    learning_rate: str,
    init: str,
    metric: str,
) -> Tuple[np.ndarray, float]:
    n = int(x.shape[0])
    upper = max(5.0, float((n - 1) / 3.0))
    perp = max(5.0, min(float(perplexity), upper))
    kwargs = dict(
        n_components=2,
        random_state=int(seed),
        perplexity=perp,
        learning_rate=learning_rate,
        init=init,
        metric=metric,
        verbose=0,
    )
    try:
        tsne = TSNE(max_iter=int(n_iter), **kwargs)
    except TypeError:
        tsne = TSNE(n_iter=int(n_iter), **kwargs)
    z = tsne.fit_transform(x)
    return z, perp


def centroid_separation_ratio(z2: np.ndarray, y_int: np.ndarray) -> float:
    uniq = sorted(int(x) for x in np.unique(y_int))
    if len(uniq) < 2:
        return float("nan")
    cents = []
    intra = []
    for u in uniq:
        sub = z2[y_int == u]
        if len(sub) == 0:
            continue
        c = sub.mean(axis=0)
        cents.append(c)
        intra.append(float(np.mean(np.linalg.norm(sub - c[None, :], axis=1))))
    if len(cents) < 2:
        return float("nan")
    cents_arr = np.stack(cents, axis=0)
    inter = []
    for i in range(len(cents_arr)):
        for j in range(i + 1, len(cents_arr)):
            inter.append(float(np.linalg.norm(cents_arr[i] - cents_arr[j])))
    if not inter:
        return float("nan")
    return float(np.mean(inter) / max(1e-12, float(np.mean(intra))))


def plot_tsne_scatter(
    z2: np.ndarray,
    rows: Sequence[Dict[str, object]],
    tasks: Sequence[str],
    out_png: str,
    title: str,
) -> None:
    color_map = {
        "alpaca_eval": "#1f77b4",
        "gsm8k": "#ff7f0e",
        "humaneval": "#2ca02c",
        "multiarith": "#d62728",
        "openfunction": "#9467bd",
    }
    plt.figure(figsize=(8, 6))
    for t in tasks:
        idx = [i for i, r in enumerate(rows) if str(r.get("task", "")) == t]
        if not idx:
            continue
        pts = z2[idx]
        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            s=16,
            alpha=0.75,
            c=color_map.get(t, None),
            label=t,
            edgecolors="none",
        )
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def refresh_stage3_merged_outputs(output_root: str) -> None:
    stage3_root = os.path.join(output_root, "stage3")
    paths = sorted(glob.glob(os.path.join(stage3_root, "by_train_dataset", "*", "story_tsne_summary_*.csv")))
    rows: List[pd.DataFrame] = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if not df.empty:
                rows.append(df)
        except Exception:
            continue
    if not rows:
        return
    merged = pd.concat(rows, axis=0, ignore_index=True)
    out_csv = os.path.join(stage3_root, "story_tsne_summary_all.csv")
    out_json = os.path.join(stage3_root, "story_tsne_summary_all.json")
    merged.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    print(os.path.abspath(out_csv))
    print(os.path.abspath(out_json))


def main() -> None:
    p = argparse.ArgumentParser(description="Stage3: generate t-SNE plots on selected layers.")
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
    p.add_argument("--layers", type=str, default="", help="explicit layers; empty means use stage2 recommended layers")
    p.add_argument("--top_k_layers", type=int, default=3, help="only used when --layers is empty")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--device", type=str, default="", help="default: auto if cuda available else cpu")
    p.add_argument("--prefer_auto_on_fail", action="store_true")
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--tsne_n_iter", type=int, default=1000)
    p.add_argument("--tsne_learning_rate", type=str, default="auto")
    p.add_argument("--tsne_init", type=str, default="pca")
    p.add_argument("--tsne_metric", type=str, default="euclidean")
    p.add_argument("--pca_dim", type=int, default=50)
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
    sample_rows, sample_stats, sample_pool_json = load_or_build_sample_pool(
        output_root=output_root,
        eval_paths=eval_paths,
        tasks=tasks,
        samples_per_task=samples_per_task,
        seed=seed,
    )
    if not sample_rows:
        raise RuntimeError("sample pool is empty; check test dataset files")
    y_int, _, _ = build_label_arrays(sample_rows, tasks)
    texts = [str(x.get("text", "")) for x in sample_rows]

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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

    selected_layers = resolve_selected_layers(
        layers_arg=args.layers,
        output_root=output_root,
        top_k=max(1, int(args.top_k_layers)),
        n_layers=n_layers,
    )
    if not selected_layers:
        raise RuntimeError("selected layers is empty")

    stage3_root = os.path.join(output_root, "stage3", "by_train_dataset")
    os.makedirs(stage3_root, exist_ok=True)

    all_unavailable: List[Dict[str, object]] = []
    for train_dataset in train_datasets:
        ds_dir = os.path.join(stage3_root, train_dataset)
        os.makedirs(ds_dir, exist_ok=True)

        model_specs = build_model_specs_story(train_dataset=train_dataset, include_base=bool(args.include_base))
        ds_rows: List[Dict[str, object]] = []
        ds_unavailable: List[Dict[str, object]] = []

        for spec in model_specs:
            model_tag = str(spec["model_tag"])
            base_path, lora_path, err_path = resolve_model_paths_for_spec(sdft_root, base_model_path, spec)
            if err_path:
                ds_unavailable.append(
                    {
                        "train_dataset_shard": train_dataset,
                        "model_tag": model_tag,
                        "status": "missing_checkpoint",
                        "reason": err_path,
                    }
                )
                continue

            model, load_err = load_model_with_optional_lora(
                base_model_path=base_path,
                lora_path=lora_path,
                device=device,
                prefer_auto_on_fail=bool(args.prefer_auto_on_fail),
            )
            if model is None:
                ds_unavailable.append(
                    {
                        "train_dataset_shard": train_dataset,
                        "model_tag": model_tag,
                        "status": "model_load_error",
                        "reason": load_err or "",
                    }
                )
                continue

            reps = extract_last_token_representations(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                layers=selected_layers,
                batch_size=batch_size,
                device=device,
                max_length=max_length,
            )
            del model
            if should_clear_cuda_cache(device):
                torch.cuda.empty_cache()

            for layer in selected_layers:
                x = reps.get(int(layer))
                if x is None or x.ndim != 2 or x.shape[0] != len(sample_rows):
                    ds_unavailable.append(
                        {
                            "train_dataset_shard": train_dataset,
                            "model_tag": model_tag,
                            "layer": int(layer),
                            "status": "invalid_representation",
                            "reason": "layer representation shape mismatch or empty",
                        }
                    )
                    continue
                x_use = reduce_for_tsne(x, pca_dim=max(2, int(args.pca_dim)), seed=seed)
                z2, perp_eff = run_tsne_2d(
                    x=x_use,
                    seed=seed,
                    perplexity=float(args.tsne_perplexity),
                    n_iter=max(250, int(args.tsne_n_iter)),
                    learning_rate=args.tsne_learning_rate.strip() or "auto",
                    init=args.tsne_init.strip() or "pca",
                    metric=args.tsne_metric.strip() or "euclidean",
                )

                coords = []
                for i, r in enumerate(sample_rows):
                    coords.append(
                        {
                            "train_dataset_shard": train_dataset,
                            "model_tag": model_tag,
                            "train_dataset": str(spec["train_dataset"]),
                            "method": str(spec["method"]),
                            "epoch": str(spec["epoch"]),
                            "layer": int(layer),
                            "task": str(r.get("task", "")),
                            "uid": str(r.get("uid", "")),
                            "sample_idx_src": int(r.get("sample_idx_src", -1)),
                            "tsne_x": float(z2[i, 0]),
                            "tsne_y": float(z2[i, 1]),
                        }
                    )
                coords_csv = os.path.join(ds_dir, f"tsne_points_{safe_name(model_tag)}_layer{int(layer):02d}.csv")
                pd.DataFrame(coords).to_csv(coords_csv, index=False)

                png = os.path.join(ds_dir, f"tsne_plot_{safe_name(model_tag)}_layer{int(layer):02d}.png")
                title = f"{train_dataset} | {model_tag} | layer {int(layer)}"
                plot_tsne_scatter(z2, sample_rows, tasks=tasks, out_png=png, title=title)

                metrics_orig = compute_cluster_metrics(x, y_int)
                metrics_tsne = compute_cluster_metrics(z2, y_int)
                row = {
                    "train_dataset_shard": train_dataset,
                    "model_tag": model_tag,
                    "train_dataset": str(spec["train_dataset"]),
                    "method": str(spec["method"]),
                    "epoch": str(spec["epoch"]),
                    "layer": int(layer),
                    "n_samples": int(x.shape[0]),
                    "hidden_dim": int(x.shape[1]),
                    "pca_dim_used": int(x_use.shape[1]),
                    "tsne_perplexity_effective": float(perp_eff),
                    "silhouette_orig": metrics_orig["silhouette"],
                    "knn_purity_orig": metrics_orig["knn_purity"],
                    "davies_bouldin_orig": metrics_orig["davies_bouldin"],
                    "calinski_harabasz_orig": metrics_orig["calinski_harabasz"],
                    "silhouette_tsne2d": metrics_tsne["silhouette"],
                    "knn_purity_tsne2d": metrics_tsne["knn_purity"],
                    "davies_bouldin_tsne2d": metrics_tsne["davies_bouldin"],
                    "calinski_harabasz_tsne2d": metrics_tsne["calinski_harabasz"],
                    "centroid_sep_ratio_tsne2d": centroid_separation_ratio(z2, y_int),
                    "points_csv": os.path.abspath(coords_csv),
                    "plot_png": os.path.abspath(png),
                    "sample_pool_json": os.path.abspath(sample_pool_json),
                    "status": "ok",
                    "reason": "",
                }
                ds_rows.append(row)

        ds_csv = os.path.join(ds_dir, f"story_tsne_summary_{train_dataset}.csv")
        ds_json = os.path.join(ds_dir, f"story_tsne_summary_{train_dataset}.json")
        pd.DataFrame(ds_rows).to_csv(ds_csv, index=False)
        with open(ds_json, "w", encoding="utf-8") as f:
            json.dump(ds_rows, f, ensure_ascii=False, indent=2)

        ds_unavail_json = os.path.join(ds_dir, f"unavailable_story_tsne_{train_dataset}.json")
        with open(ds_unavail_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "train_dataset_shard": train_dataset,
                    "unavailable_count": len(ds_unavailable),
                    "unavailable": ds_unavailable,
                    "sample_pool_json": os.path.abspath(sample_pool_json),
                    "sample_stats": sample_stats,
                    "selected_layers": selected_layers,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        all_unavailable.extend(ds_unavailable)

        print(os.path.abspath(ds_csv))
        print(os.path.abspath(ds_json))
        print(os.path.abspath(ds_unavail_json))

    unavail_json = os.path.join(output_root, "stage3", "unavailable_story_tsne_all.json")
    os.makedirs(os.path.dirname(unavail_json), exist_ok=True)
    with open(unavail_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "unavailable_count": len(all_unavailable),
                "unavailable": all_unavailable,
                "selected_layers": selected_layers,
                "sample_pool_json": os.path.abspath(sample_pool_json),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(os.path.abspath(unavail_json))

    refresh_stage3_merged_outputs(output_root)


if __name__ == "__main__":
    main()

