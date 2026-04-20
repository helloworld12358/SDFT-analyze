#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional

import pandas as pd

from embedding_cluster_utils import init_runtime_paths


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = float(x.mean()) if len(x) else float("nan")
    sd = float(x.std(ddof=0)) if len(x) else float("nan")
    if not math.isfinite(sd) or sd <= 1e-12:
        return pd.Series([0.0] * len(x), index=s.index)
    return (x - mu) / sd


def load_stage1_rows(output_root: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(output_root, "stage1", "by_train_dataset", "*", "epoch5_layer_scan_*.csv")))
    dfs: List[pd.DataFrame] = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if not df.empty:
                dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, axis=0, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Stage2: merge epoch5 scan and select most discriminative layers.")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default="")
    p.add_argument("--top_k_layers", type=int, default=3)
    args = p.parse_args()

    rt = init_runtime_paths(args.datainf_root, args.output_root, "", "")
    output_root = rt["output_root"]
    top_k = max(1, int(args.top_k_layers))

    df = load_stage1_rows(output_root)
    if df.empty:
        raise RuntimeError("no stage1 epoch5 scan csv found; run embedding_cluster_01_epoch5_layer_scan.py first")

    for c in ["train_dataset_shard", "method", "status"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    df = df[(df["status"] == "ok") & (df["method"].isin(["sft", "sdft"]))].copy()
    if df.empty:
        raise RuntimeError("no usable sft/sdft rows with status=ok in stage1 data")

    val_cols = ["silhouette", "davies_bouldin", "calinski_harabasz", "knn_purity"]
    for c in val_cols + ["layer"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["layer"])
    df["layer"] = df["layer"].astype(int)

    # build sft/sdft pair table by (train_dataset_shard, layer)
    key_cols = ["train_dataset_shard", "layer"]
    sft = df[df["method"] == "sft"][key_cols + val_cols].copy()
    sft = sft.rename(columns={c: f"sft_{c}" for c in val_cols})
    sdft = df[df["method"] == "sdft"][key_cols + val_cols].copy()
    sdft = sdft.rename(columns={c: f"sdft_{c}" for c in val_cols})
    merged = sft.merge(sdft, on=key_cols, how="inner")
    if merged.empty:
        raise RuntimeError("no matched sft/sdft rows by (train_dataset_shard, layer)")

    merged["delta_silhouette"] = merged["sdft_silhouette"] - merged["sft_silhouette"]
    merged["delta_knn_purity"] = merged["sdft_knn_purity"] - merged["sft_knn_purity"]
    merged["delta_calinski_harabasz"] = merged["sdft_calinski_harabasz"] - merged["sft_calinski_harabasz"]
    merged["delta_db_better"] = merged["sft_davies_bouldin"] - merged["sdft_davies_bouldin"]  # higher is better

    # normalize per train_dataset_shard to avoid scale mismatch
    score_rows: List[pd.DataFrame] = []
    for td, sub in merged.groupby("train_dataset_shard"):
        sub = sub.copy()
        sub["z_delta_silhouette"] = zscore(sub["delta_silhouette"])
        sub["z_delta_knn_purity"] = zscore(sub["delta_knn_purity"])
        sub["z_delta_calinski_harabasz"] = zscore(sub["delta_calinski_harabasz"])
        sub["z_delta_db_better"] = zscore(sub["delta_db_better"])
        sub["composite_score"] = (
            sub["z_delta_silhouette"]
            + sub["z_delta_knn_purity"]
            + sub["z_delta_calinski_harabasz"]
            + sub["z_delta_db_better"]
        ) / 4.0
        score_rows.append(sub)
    merged = pd.concat(score_rows, axis=0, ignore_index=True)

    agg = (
        merged.groupby("layer", as_index=False)
        .agg(
            n_train_datasets=("train_dataset_shard", "nunique"),
            mean_composite_score=("composite_score", "mean"),
            mean_delta_silhouette=("delta_silhouette", "mean"),
            mean_delta_knn_purity=("delta_knn_purity", "mean"),
            mean_delta_calinski_harabasz=("delta_calinski_harabasz", "mean"),
            mean_delta_db_better=("delta_db_better", "mean"),
            positive_rate_composite=("composite_score", lambda x: float((x > 0).mean())),
            positive_rate_silhouette=("delta_silhouette", lambda x: float((x > 0).mean())),
            positive_rate_knn=("delta_knn_purity", lambda x: float((x > 0).mean())),
        )
        .sort_values(["mean_composite_score", "positive_rate_composite", "mean_delta_silhouette"], ascending=False)
        .reset_index(drop=True)
    )

    top = agg.head(top_k).copy()
    recommended_layers = [int(x) for x in top["layer"].tolist()]

    out_dir = os.path.join(output_root, "stage2")
    os.makedirs(out_dir, exist_ok=True)
    long_csv = os.path.join(out_dir, "layer_deltas_long.csv")
    agg_csv = os.path.join(out_dir, "layer_rank_summary.csv")
    merged.to_csv(long_csv, index=False)
    agg.to_csv(agg_csv, index=False)

    rec_json = os.path.join(out_dir, "recommended_layers.json")
    with open(rec_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "top_k_layers": top_k,
                "recommended_layers": recommended_layers,
                "selection_rule": "rank by mean composite score over train datasets",
                "components": [
                    "delta_silhouette",
                    "delta_knn_purity",
                    "delta_calinski_harabasz",
                    "delta_db_better",
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    txt = os.path.join(out_dir, "layer_selection_summary.txt")
    lines: List[str] = []
    lines.append("# Layer Selection Summary (Epoch5)")
    lines.append("")
    lines.append(f"Top-K: {top_k}")
    lines.append(f"Recommended layers: {recommended_layers}")
    lines.append("")
    lines.append("|layer|mean_composite_score|mean_delta_silhouette|mean_delta_knn_purity|mean_delta_calinski_harabasz|mean_delta_db_better|positive_rate_composite|")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in top.iterrows():
        lines.append(
            f"|{int(r['layer'])}|{float(r['mean_composite_score']):.6f}|{float(r['mean_delta_silhouette']):.6f}|"
            f"{float(r['mean_delta_knn_purity']):.6f}|{float(r['mean_delta_calinski_harabasz']):.6f}|"
            f"{float(r['mean_delta_db_better']):.6f}|{float(r['positive_rate_composite']):.4f}|"
        )
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(os.path.abspath(long_csv))
    print(os.path.abspath(agg_csv))
    print(os.path.abspath(rec_json))
    print(os.path.abspath(txt))


if __name__ == "__main__":
    main()

