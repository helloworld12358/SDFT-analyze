#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from typing import List

import pandas as pd


def _normalize(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo = float(x.min())
    hi = float(x.max())
    if not pd.notna(lo) or not pd.notna(hi) or hi <= lo:
        return pd.Series([0.0] * len(x), index=x.index)
    return (x - lo) / (hi - lo)


def pick_layers(
    output_root: str,
    top_k: int,
    policy: str,
    min_layer: int,
    max_layer: int,
) -> pd.DataFrame:
    summary_dir = os.path.join(output_root, "summary")
    fam_csv = os.path.join(summary_dir, "layer_metrics_agg_by_family.csv")
    delta_csv = os.path.join(summary_dir, "layer_delta_sdft_minus_sft_agg.csv")

    if not os.path.isfile(fam_csv):
        raise FileNotFoundError(f"missing summary file: {fam_csv}")
    if not os.path.isfile(delta_csv):
        raise FileNotFoundError(f"missing summary file: {delta_csv}")

    fam = pd.read_csv(fam_csv)
    delta = pd.read_csv(delta_csv)

    fam["layer"] = pd.to_numeric(fam["layer"], errors="coerce").astype("Int64")
    fam["silhouette_mean"] = pd.to_numeric(fam["silhouette_mean"], errors="coerce")
    fam = fam.dropna(subset=["layer", "silhouette_mean"])
    fam["layer"] = fam["layer"].astype(int)

    pivot = fam.pivot_table(index="layer", columns="family", values="silhouette_mean", aggfunc="mean").reset_index()
    if "sft" not in pivot.columns or "sdft" not in pivot.columns:
        raise RuntimeError("layer_metrics_agg_by_family.csv does not contain both sft and sdft rows")

    pivot["common_sil"] = pivot[["sft", "sdft"]].min(axis=1)
    pivot["avg_sil"] = pivot[["sft", "sdft"]].mean(axis=1)

    d = delta.copy()
    d["layer"] = pd.to_numeric(d["layer"], errors="coerce").astype("Int64")
    d["mean_delta_silhouette"] = pd.to_numeric(d["mean_delta_silhouette"], errors="coerce")
    d = d.dropna(subset=["layer", "mean_delta_silhouette"])
    d["layer"] = d["layer"].astype(int)
    d["abs_delta_sil"] = d["mean_delta_silhouette"].abs()

    merged = pivot.merge(d[["layer", "mean_delta_silhouette", "abs_delta_sil"]], on="layer", how="left")
    merged = merged[(merged["layer"] >= int(min_layer)) & (merged["layer"] <= int(max_layer))].copy()
    if merged.empty:
        raise RuntimeError("no layers left after layer range filter")

    if policy == "common_quality":
        merged["score"] = merged["common_sil"]
        merged = merged.sort_values(["score", "avg_sil"], ascending=[False, False])
    elif policy == "delta_focus":
        merged["score"] = merged["abs_delta_sil"]
        merged = merged.sort_values(["score", "common_sil"], ascending=[False, False])
    elif policy == "balanced":
        ns_common = _normalize(merged["common_sil"])
        ns_delta = _normalize(merged["abs_delta_sil"])
        merged["score"] = 0.7 * ns_common + 0.3 * ns_delta
        merged = merged.sort_values(["score", "common_sil"], ascending=[False, False])
    else:
        raise ValueError(f"unknown policy: {policy}")

    out = merged.reset_index(drop=True).head(max(1, int(top_k))).copy()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Pick shared comparable layers for epoch0 QA t-SNE.")
    p.add_argument("--output_root", type=str, required=True, help="existing output root with summary/*.csv")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--policy", type=str, default="common_quality", choices=["common_quality", "delta_focus", "balanced"])
    p.add_argument("--min_layer", type=int, default=1)
    p.add_argument("--max_layer", type=int, default=32)
    p.add_argument("--out_json", type=str, default="")
    p.add_argument("--out_csv", type=str, default="")
    p.add_argument("--out_txt", type=str, default="")
    args = p.parse_args()

    picked = pick_layers(
        output_root=args.output_root,
        top_k=args.top_k,
        policy=args.policy,
        min_layer=int(args.min_layer),
        max_layer=int(args.max_layer),
    )

    layers: List[int] = [int(x) for x in picked["layer"].tolist()]
    layers_csv = ",".join(str(x) for x in layers)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "output_root": os.path.abspath(args.output_root),
                    "policy": args.policy,
                    "top_k": int(args.top_k),
                    "layers": layers,
                    "layers_csv": layers_csv,
                    "picked_rows": picked.to_dict(orient="records"),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        picked.to_csv(args.out_csv, index=False)
    if args.out_txt:
        os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
        with open(args.out_txt, "w", encoding="utf-8") as f:
            f.write(layers_csv + "\n")

    print(layers_csv)


if __name__ == "__main__":
    main()
