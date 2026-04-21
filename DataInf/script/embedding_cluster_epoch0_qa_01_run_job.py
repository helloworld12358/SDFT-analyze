#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

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
    compute_cluster_metrics,
    extract_token_representations_at_indices,
    infer_layer_count,
    init_runtime_paths,
    load_model_with_optional_lora,
    load_records,
    parse_layers_spec,
    should_clear_cuda_cache,
    smart_parse_example,
)


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(s))


def split_csv(s: str, default: Sequence[str]) -> List[str]:
    t = (s or "").strip()
    if not t:
        return list(default)
    out = [x.strip() for x in t.split(",") if x.strip()]
    return out if out else list(default)


def resolve_train_data_path(data_root: str, train_dataset: str, family: str) -> str:
    # family=sft -> *_train.json ; family=sdft -> distilled_*.json
    if family == "sft":
        cands = [
            os.path.join(data_root, train_dataset, f"{train_dataset}_train.json"),
            os.path.join(data_root, f"{train_dataset}_train.json"),
        ]
    else:
        cands = [
            os.path.join(data_root, train_dataset, f"distilled_{train_dataset}.json"),
            os.path.join(data_root, f"distilled_{train_dataset}.json"),
        ]
    for p in cands:
        if os.path.isfile(p):
            return p
    return cands[0]


def build_prompt_prefix(question: str) -> str:
    q = (question or "").strip()
    return f"### Instruction:\n{q}\n\n### Response:\n"


def build_qa_full_and_answer_last_idx(
    tokenizer: AutoTokenizer,
    question: str,
    answer: str,
    max_length: int,
) -> Tuple[Optional[str], Optional[int], Optional[Dict[str, int]]]:
    q = (question or "").strip()
    a = (answer or "").strip()
    if not q or not a:
        return None, None, None
    prefix = build_prompt_prefix(q)
    full = prefix + a

    pref_ids = tokenizer(
        prefix,
        add_special_tokens=True,
        truncation=(max_length > 0),
        max_length=max_length if max_length > 0 else None,
    )["input_ids"]
    full_ids = tokenizer(
        full,
        add_special_tokens=True,
        truncation=(max_length > 0),
        max_length=max_length if max_length > 0 else None,
    )["input_ids"]
    pref_len = len(pref_ids)
    full_len = len(full_ids)
    if full_len <= pref_len:
        return None, None, {"pref_len": pref_len, "full_len": full_len}
    ans_last_idx = full_len - 1
    return full, int(ans_last_idx), {"pref_len": pref_len, "full_len": full_len}


def build_balanced_pool(
    tokenizer: AutoTokenizer,
    data_root: str,
    train_datasets: Sequence[str],
    family: str,
    samples_per_class: int,
    seed: int,
    max_length: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    stats: List[Dict[str, object]] = []

    for ds_i, ds in enumerate(train_datasets):
        path = resolve_train_data_path(data_root, ds, family)
        recs = load_records(path) if os.path.isfile(path) else []
        idx = list(range(len(recs)))
        rng = random.Random(seed * 10007 + ds_i * 97 + (0 if family == "sft" else 1) * 131)
        rng.shuffle(idx)

        picked = 0
        tried = 0
        skip_empty = 0
        skip_trunc = 0
        task_rows: List[Dict[str, object]] = []
        for j in idx:
            if picked >= samples_per_class:
                break
            tried += 1
            rec = recs[j]
            q, a = smart_parse_example(rec if isinstance(rec, dict) else {})
            full, ans_last_idx, lens = build_qa_full_and_answer_last_idx(
                tokenizer=tokenizer,
                question=q,
                answer=a,
                max_length=max_length,
            )
            if full is None or ans_last_idx is None:
                if not q.strip() or not a.strip():
                    skip_empty += 1
                else:
                    skip_trunc += 1
                continue
            uid = (
                str(rec.get("id", rec.get("uid", rec.get("sample_id", f"{ds}_{j}"))))
                if isinstance(rec, dict)
                else f"{ds}_{j}"
            )
            task_rows.append(
                {
                    "family": family,
                    "class_label": ds,
                    "train_dataset": ds,
                    "uid": uid,
                    "sample_idx_src": int(j),
                    "text": full,
                    "answer_last_token_idx": int(ans_last_idx),
                    "pref_len": int(lens["pref_len"]) if lens else None,
                    "full_len": int(lens["full_len"]) if lens else None,
                    "dataset_path": path,
                }
            )
            picked += 1

        rows.extend(task_rows)
        stats.append(
            {
                "family": family,
                "class_label": ds,
                "dataset_path": path,
                "available_records": len(recs),
                "tried_records": tried,
                "selected": picked,
                "target_per_class": samples_per_class,
                "skip_empty_qa": skip_empty,
                "skip_trunc_or_no_answer_token": skip_trunc,
            }
        )

    rows = sorted(rows, key=lambda x: (str(x["class_label"]), int(x["sample_idx_src"]), str(x["uid"])))
    return rows, stats


def label_to_int(labels: Sequence[str], class_order: Sequence[str]) -> np.ndarray:
    mp = {c: i for i, c in enumerate(class_order)}
    out = [mp[str(x)] for x in labels]
    return np.asarray(out, dtype=np.int64)


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


def plot_tsne_scatter(
    z2: np.ndarray,
    labels: Sequence[str],
    class_order: Sequence[str],
    out_png: str,
    title: str,
) -> None:
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    cmap = {c: palette[i % len(palette)] for i, c in enumerate(class_order)}
    plt.figure(figsize=(8, 6))
    labels_np = np.asarray([str(x) for x in labels], dtype=object)
    for c in class_order:
        idx = np.where(labels_np == c)[0]
        if len(idx) == 0:
            continue
        pts = z2[idx]
        plt.scatter(pts[:, 0], pts[:, 1], s=14, alpha=0.75, c=cmap[c], label=c, edgecolors="none")
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def choose_tsne_layers(metrics_df: pd.DataFrame, top_k: int) -> List[int]:
    d = metrics_df.copy()
    d["silhouette_orig"] = pd.to_numeric(d["silhouette_orig"], errors="coerce")
    d = d.sort_values("silhouette_orig", ascending=False)
    layers = [int(x) for x in d["layer"].dropna().astype(int).tolist()]
    out: List[int] = []
    for x in layers:
        if x not in out:
            out.append(x)
        if len(out) >= max(1, top_k):
            break
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Epoch0 QA-last-token embedding clustering job (single family + seed).")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default="")
    p.add_argument("--base_model_path", type=str, default="")
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--family", type=str, choices=["sft", "sdft"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--train_datasets", type=str, default="")
    p.add_argument("--samples_per_class", type=int, default=500)
    p.add_argument("--layers", type=str, default="all")
    p.add_argument("--batch_size", type=int, default=0, help="<=0 means auto probe max safe batch")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_probe_batch", type=int, default=256)
    p.add_argument("--disable_auto_tune_batch", action="store_true")
    p.add_argument("--device", type=str, default="", help="default: auto if cuda available else cpu")
    p.add_argument("--prefer_auto_on_fail", action="store_true")
    p.add_argument("--disable_tsne", action="store_true")
    p.add_argument("--tsne_layers", type=str, default="", help="optional explicit layers for tsne, e.g. 8,16,24")
    p.add_argument("--tsne_top_k_layers", type=int, default=3, help="if tsne_layers empty, choose top-k by silhouette_orig")
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--tsne_n_iter", type=int, default=1000)
    p.add_argument("--tsne_learning_rate", type=str, default="auto")
    p.add_argument("--tsne_init", type=str, default="pca")
    p.add_argument("--tsne_metric", type=str, default="euclidean")
    p.add_argument("--pca_dim", type=int, default=50)
    args = p.parse_args()

    rt = init_runtime_paths(args.datainf_root, args.output_root, args.base_model_path, args.data_root)
    output_root = rt["output_root"]
    base_model_path = rt["base_model_path"]
    data_root = rt["data_root"]
    family = str(args.family).strip().lower()
    seed = int(args.seed)
    samples_per_class = max(1, int(args.samples_per_class))
    max_length = int(args.max_length)
    device = args.device.strip() or ("auto" if torch.cuda.is_available() else "cpu")
    batch_size = int(args.batch_size)
    max_probe_batch = max(1, int(args.max_probe_batch))
    auto_tune_batch = not bool(args.disable_auto_tune_batch)

    train_datasets = split_csv(args.train_datasets, DEFAULT_TRAIN_DATASETS)

    job_dir = os.path.join(output_root, "jobs", family, f"seed_{seed}")
    os.makedirs(job_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    sample_rows, sample_stats = build_balanced_pool(
        tokenizer=tokenizer,
        data_root=data_root,
        train_datasets=train_datasets,
        family=family,
        samples_per_class=samples_per_class,
        seed=seed,
        max_length=max_length,
    )
    if not sample_rows:
        raise RuntimeError("sample pool is empty; check training dataset files and parse logic")

    pool_json = os.path.join(job_dir, f"sample_pool_{family}_seed{seed}.json")
    with open(pool_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "family": family,
                "seed": seed,
                "train_datasets": train_datasets,
                "samples_per_class": samples_per_class,
                "rows": sample_rows,
                "stats": sample_stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    texts = [str(x["text"]) for x in sample_rows]
    answer_last_token_idx = [int(x["answer_last_token_idx"]) for x in sample_rows]
    labels = [str(x["class_label"]) for x in sample_rows]
    y_int = label_to_int(labels, class_order=train_datasets)

    model, load_err = load_model_with_optional_lora(
        base_model_path=base_model_path,
        lora_path=None,
        device=device,
        prefer_auto_on_fail=bool(args.prefer_auto_on_fail),
    )
    if model is None:
        raise RuntimeError(f"failed to load base model: {load_err}")
    n_layers = infer_layer_count(model, tokenizer, device=device)
    layers = parse_layers_spec(args.layers, n_layers)

    reps = extract_token_representations_at_indices(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        token_indices=answer_last_token_idx,
        layers=layers,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
        auto_tune_batch=auto_tune_batch,
        max_probe_batch=max_probe_batch,
    )
    del model
    if should_clear_cuda_cache(device):
        torch.cuda.empty_cache()

    metric_rows: List[Dict[str, object]] = []
    for layer in layers:
        x = reps.get(int(layer))
        if x is None or x.ndim != 2 or x.shape[0] != len(sample_rows):
            metric_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "layer": int(layer),
                    "status": "invalid_representation",
                    "reason": "representation empty or shape mismatch",
                    "n_samples": len(sample_rows),
                    "n_classes": len(train_datasets),
                    "silhouette_orig": None,
                    "knn_purity_orig": None,
                    "davies_bouldin_orig": None,
                    "calinski_harabasz_orig": None,
                }
            )
            continue
        m = compute_cluster_metrics(x, y_int)
        metric_rows.append(
            {
                "family": family,
                "seed": seed,
                "layer": int(layer),
                "status": "ok",
                "reason": "",
                "n_samples": int(x.shape[0]),
                "n_classes": len(train_datasets),
                "silhouette_orig": m["silhouette"],
                "knn_purity_orig": m["knn_purity"],
                "davies_bouldin_orig": m["davies_bouldin"],
                "calinski_harabasz_orig": m["calinski_harabasz"],
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_csv = os.path.join(job_dir, f"layer_metrics_{family}_seed{seed}.csv")
    metrics_json = os.path.join(job_dir, f"layer_metrics_{family}_seed{seed}.json")
    metrics_df.to_csv(metrics_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metric_rows, f, ensure_ascii=False, indent=2)

    tsne_summary_rows: List[Dict[str, object]] = []
    if not bool(args.disable_tsne):
        tsne_layers_arg = (args.tsne_layers or "").strip()
        if tsne_layers_arg:
            tsne_layers = parse_layers_spec(tsne_layers_arg, n_layers)
        else:
            tsne_layers = choose_tsne_layers(metrics_df[metrics_df["status"] == "ok"], top_k=max(1, int(args.tsne_top_k_layers)))

        tsne_dir = os.path.join(job_dir, "tsne")
        os.makedirs(tsne_dir, exist_ok=True)
        for layer in tsne_layers:
            x = reps.get(int(layer))
            if x is None or x.ndim != 2 or x.shape[0] != len(sample_rows):
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

            pts = []
            for i, r in enumerate(sample_rows):
                pts.append(
                    {
                        "family": family,
                        "seed": seed,
                        "layer": int(layer),
                        "class_label": str(r["class_label"]),
                        "uid": str(r["uid"]),
                        "sample_idx_src": int(r["sample_idx_src"]),
                        "tsne_x": float(z2[i, 0]),
                        "tsne_y": float(z2[i, 1]),
                    }
                )
            pts_csv = os.path.join(tsne_dir, f"tsne_points_{family}_seed{seed}_layer{int(layer):02d}.csv")
            pd.DataFrame(pts).to_csv(pts_csv, index=False)

            png = os.path.join(tsne_dir, f"tsne_plot_{family}_seed{seed}_layer{int(layer):02d}.png")
            title = f"{family} | seed={seed} | layer={int(layer)} | epoch0 | QA-last-token"
            plot_tsne_scatter(z2, labels=labels, class_order=train_datasets, out_png=png, title=title)

            m2 = compute_cluster_metrics(z2, y_int)
            tsne_summary_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "layer": int(layer),
                    "n_samples": len(sample_rows),
                    "n_classes": len(train_datasets),
                    "pca_dim_used": int(x_use.shape[1]),
                    "tsne_perplexity_effective": float(perp_eff),
                    "silhouette_tsne2d": m2["silhouette"],
                    "knn_purity_tsne2d": m2["knn_purity"],
                    "davies_bouldin_tsne2d": m2["davies_bouldin"],
                    "calinski_harabasz_tsne2d": m2["calinski_harabasz"],
                    "points_csv": os.path.abspath(pts_csv),
                    "plot_png": os.path.abspath(png),
                }
            )

    tsne_csv = os.path.join(job_dir, f"tsne_summary_{family}_seed{seed}.csv")
    tsne_json = os.path.join(job_dir, f"tsne_summary_{family}_seed{seed}.json")
    pd.DataFrame(tsne_summary_rows).to_csv(tsne_csv, index=False)
    with open(tsne_json, "w", encoding="utf-8") as f:
        json.dump(tsne_summary_rows, f, ensure_ascii=False, indent=2)

    print(os.path.abspath(pool_json))
    print(os.path.abspath(metrics_csv))
    print(os.path.abspath(metrics_json))
    print(os.path.abspath(tsne_csv))
    print(os.path.abspath(tsne_json))


if __name__ == "__main__":
    main()

