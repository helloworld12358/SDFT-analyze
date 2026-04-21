#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

import pandas as pd

from embedding_cluster_utils import init_runtime_paths


def gather_csv(pattern: str) -> List[str]:
    return sorted(glob.glob(pattern))


def safe_mean(s: pd.Series) -> float:
    if s is None or len(s) == 0:
        return float("nan")
    return float(pd.to_numeric(s, errors="coerce").mean())


def safe_pos_rate(s: pd.Series) -> float:
    if s is None or len(s) == 0:
        return float("nan")
    s2 = pd.to_numeric(s, errors="coerce")
    return float((s2 > 0).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate epoch0 QA-last-token embedding clustering results.")
    parser.add_argument("--datainf_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="")
    args = parser.parse_args()

    rt = init_runtime_paths(args.datainf_root, args.output_root, "", "")
    output_root = rt["output_root"]

    metric_paths = gather_csv(os.path.join(output_root, "jobs", "*", "seed_*", "layer_metrics_*.csv"))
    tsne_paths = gather_csv(os.path.join(output_root, "jobs", "*", "seed_*", "tsne_summary_*.csv"))

    if not metric_paths:
        raise RuntimeError("No job metric csv found. Please run embedding_cluster_epoch0_qa_01_run_job.py first.")

    metric_df = pd.concat([pd.read_csv(p) for p in metric_paths], axis=0, ignore_index=True)
    metric_df["family"] = metric_df["family"].astype(str)
    metric_df["seed"] = pd.to_numeric(metric_df["seed"], errors="coerce").astype("Int64")
    metric_df["layer"] = pd.to_numeric(metric_df["layer"], errors="coerce").astype("Int64")

    summary_dir = os.path.join(output_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    metric_all_csv = os.path.join(summary_dir, "layer_metrics_all_jobs.csv")
    metric_df.to_csv(metric_all_csv, index=False)

    ok = metric_df[metric_df["status"] == "ok"].copy()
    for c in ["silhouette_orig", "knn_purity_orig", "davies_bouldin_orig", "calinski_harabasz_orig"]:
        ok[c] = pd.to_numeric(ok[c], errors="coerce")

    agg_family = (
        ok.groupby(["family", "layer"], as_index=False)
        .agg(
            n_runs=("seed", "count"),
            silhouette_mean=("silhouette_orig", "mean"),
            silhouette_std=("silhouette_orig", "std"),
            knn_mean=("knn_purity_orig", "mean"),
            knn_std=("knn_purity_orig", "std"),
            db_mean=("davies_bouldin_orig", "mean"),
            db_std=("davies_bouldin_orig", "std"),
            ch_mean=("calinski_harabasz_orig", "mean"),
            ch_std=("calinski_harabasz_orig", "std"),
        )
        .sort_values(["family", "silhouette_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    agg_family_csv = os.path.join(summary_dir, "layer_metrics_agg_by_family.csv")
    agg_family.to_csv(agg_family_csv, index=False)

    key = ["seed", "layer"]
    val = ["silhouette_orig", "knn_purity_orig", "davies_bouldin_orig", "calinski_harabasz_orig"]
    sft = ok[ok["family"] == "sft"][key + val].copy().rename(columns={c: f"sft_{c}" for c in val})
    sdft = ok[ok["family"] == "sdft"][key + val].copy().rename(columns={c: f"sdft_{c}" for c in val})
    delta = sft.merge(sdft, on=key, how="inner")

    if not delta.empty:
        delta["delta_silhouette"] = delta["sdft_silhouette_orig"] - delta["sft_silhouette_orig"]
        delta["delta_knn"] = delta["sdft_knn_purity_orig"] - delta["sft_knn_purity_orig"]
        delta["delta_ch"] = delta["sdft_calinski_harabasz_orig"] - delta["sft_calinski_harabasz_orig"]
        # DB 越小越好，因此定义“更好差值”为 sft - sdft
        delta["delta_db_better"] = delta["sft_davies_bouldin_orig"] - delta["sdft_davies_bouldin_orig"]
    else:
        delta["delta_silhouette"] = []
        delta["delta_knn"] = []
        delta["delta_ch"] = []
        delta["delta_db_better"] = []

    delta_csv = os.path.join(summary_dir, "layer_delta_sdft_minus_sft_by_seed.csv")
    delta.to_csv(delta_csv, index=False)

    if not delta.empty:
        delta_agg = (
            delta.groupby("layer", as_index=False)
            .agg(
                n_runs=("seed", "count"),
                mean_delta_silhouette=("delta_silhouette", "mean"),
                mean_delta_knn=("delta_knn", "mean"),
                mean_delta_ch=("delta_ch", "mean"),
                mean_delta_db_better=("delta_db_better", "mean"),
                pos_rate_delta_silhouette=("delta_silhouette", lambda x: float((x > 0).mean())),
                pos_rate_delta_knn=("delta_knn", lambda x: float((x > 0).mean())),
                pos_rate_delta_ch=("delta_ch", lambda x: float((x > 0).mean())),
                pos_rate_delta_db_better=("delta_db_better", lambda x: float((x > 0).mean())),
            )
            .sort_values("mean_delta_silhouette", ascending=False)
            .reset_index(drop=True)
        )
    else:
        delta_agg = pd.DataFrame(
            columns=[
                "layer",
                "n_runs",
                "mean_delta_silhouette",
                "mean_delta_knn",
                "mean_delta_ch",
                "mean_delta_db_better",
                "pos_rate_delta_silhouette",
                "pos_rate_delta_knn",
                "pos_rate_delta_ch",
                "pos_rate_delta_db_better",
            ]
        )

    delta_agg_csv = os.path.join(summary_dir, "layer_delta_sdft_minus_sft_agg.csv")
    delta_agg.to_csv(delta_agg_csv, index=False)

    bad = delta[
        (pd.to_numeric(delta.get("delta_silhouette"), errors="coerce") <= 0)
        | (pd.to_numeric(delta.get("delta_knn"), errors="coerce") <= 0)
        | (pd.to_numeric(delta.get("delta_ch"), errors="coerce") <= 0)
        | (pd.to_numeric(delta.get("delta_db_better"), errors="coerce") <= 0)
    ].copy()
    bad = bad.sort_values(["seed", "layer"]).reset_index(drop=True)
    bad_csv = os.path.join(summary_dir, "counterexamples_orig_by_seed_layer.csv")
    bad.to_csv(bad_csv, index=False)

    if tsne_paths:
        tsne_df = pd.concat([pd.read_csv(p) for p in tsne_paths], axis=0, ignore_index=True)
    else:
        tsne_df = pd.DataFrame()
    tsne_csv = os.path.join(summary_dir, "tsne_summary_all_jobs.csv")
    tsne_df.to_csv(tsne_csv, index=False)

    md_path = os.path.join(summary_dir, "epoch0_qa_answer_lasttok_summary_zh.md")
    lines: List[str] = []
    lines.append("# Epoch0 QA（问题+答案拼接）A最后token 聚类汇总")
    lines.append("")
    lines.append("## 1. 实验范围")
    lines.append("- epoch: `epoch_0`（base模型，无adapter）")
    lines.append("- family: `sft` 与 `sdft`")
    lines.append("- 类别数: 7（对应7个训练数据集）")
    lines.append("- 表示位置: 每条样本仅取 `A` 的最后一个 token 的层表示")
    lines.append("")
    lines.append("## 2. 指标含义")
    lines.append("- `silhouette_orig`：越大越好，表示类内更紧、类间更远。")
    lines.append("- `knn_purity_orig`：越大越好，邻域标签一致性更强。")
    lines.append("- `davies_bouldin_orig`：越小越好，类间分离更好。")
    lines.append("- `calinski_harabasz_orig`：越大越好，类间方差相对类内方差更大。")
    lines.append("- 对比差值定义：")
    lines.append("  - `delta_silhouette = sdft - sft`（>0 表示 sdft 更好）")
    lines.append("  - `delta_knn = sdft - sft`（>0 表示 sdft 更好）")
    lines.append("  - `delta_ch = sdft - sft`（>0 表示 sdft 更好）")
    lines.append("  - `delta_db_better = sft - sdft`（>0 表示 sdft 更好，因为 DB 越小越好）")
    lines.append("")
    lines.append("## 3. 关键输出文件")
    lines.append(f"- 全部作业指标: `{os.path.abspath(metric_all_csv)}`")
    lines.append(f"- family分层均值: `{os.path.abspath(agg_family_csv)}`")
    lines.append(f"- sdft-sft差值（逐seed）: `{os.path.abspath(delta_csv)}`")
    lines.append(f"- sdft-sft差值（层聚合）: `{os.path.abspath(delta_agg_csv)}`")
    lines.append(f"- 高维反例全集: `{os.path.abspath(bad_csv)}`")
    lines.append(f"- t-SNE汇总: `{os.path.abspath(tsne_csv)}`")
    lines.append("")
    lines.append("## 4. 总体统计（高维原空间）")
    if delta.empty:
        lines.append("- 未找到可对齐的 sft/sdft 配对，无法计算差值。")
    else:
        lines.append(f"- 配对数: {len(delta)}")
        lines.append(f"- mean(ΔSilhouette): {safe_mean(delta['delta_silhouette']):.6f} ; pos_rate: {safe_pos_rate(delta['delta_silhouette']):.3f}")
        lines.append(f"- mean(ΔkNN): {safe_mean(delta['delta_knn']):.6f} ; pos_rate: {safe_pos_rate(delta['delta_knn']):.3f}")
        lines.append(f"- mean(ΔCH): {safe_mean(delta['delta_ch']):.6f} ; pos_rate: {safe_pos_rate(delta['delta_ch']):.3f}")
        lines.append(f"- mean(DB_better): {safe_mean(delta['delta_db_better']):.6f} ; pos_rate: {safe_pos_rate(delta['delta_db_better']):.3f}")
    lines.append("")
    lines.append("## 5. 反例统计（高维原空间）")
    if delta.empty:
        lines.append("- 无可统计项。")
    else:
        lines.append(f"- ΔSilhouette <= 0: {int((pd.to_numeric(delta['delta_silhouette'], errors='coerce') <= 0).sum())}")
        lines.append(f"- ΔkNN <= 0: {int((pd.to_numeric(delta['delta_knn'], errors='coerce') <= 0).sum())}")
        lines.append(f"- ΔCH <= 0: {int((pd.to_numeric(delta['delta_ch'], errors='coerce') <= 0).sum())}")
        lines.append(f"- DB_better <= 0: {int((pd.to_numeric(delta['delta_db_better'], errors='coerce') <= 0).sum())}")
        lines.append(f"- 任一指标<=0 的 pair 数: {len(bad)}")
    lines.append("")

    with open(md_path, "w", encoding="utf-8-sig", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    summary_json = os.path.join(summary_dir, "epoch0_qa_answer_lasttok_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metric_paths_count": len(metric_paths),
                "tsne_paths_count": len(tsne_paths),
                "rows": {
                    "metrics_all": int(len(metric_df)),
                    "metrics_ok": int(len(ok)),
                    "delta_pairs": int(len(delta)),
                    "counterexamples": int(len(bad)),
                },
                "overall": {
                    "mean_delta_silhouette": safe_mean(delta["delta_silhouette"]) if not delta.empty else None,
                    "mean_delta_knn": safe_mean(delta["delta_knn"]) if not delta.empty else None,
                    "mean_delta_ch": safe_mean(delta["delta_ch"]) if not delta.empty else None,
                    "mean_delta_db_better": safe_mean(delta["delta_db_better"]) if not delta.empty else None,
                    "pos_rate_delta_silhouette": safe_pos_rate(delta["delta_silhouette"]) if not delta.empty else None,
                    "pos_rate_delta_knn": safe_pos_rate(delta["delta_knn"]) if not delta.empty else None,
                    "pos_rate_delta_ch": safe_pos_rate(delta["delta_ch"]) if not delta.empty else None,
                    "pos_rate_delta_db_better": safe_pos_rate(delta["delta_db_better"]) if not delta.empty else None,
                },
                "outputs": {
                    "metric_all_csv": os.path.abspath(metric_all_csv),
                    "agg_family_csv": os.path.abspath(agg_family_csv),
                    "delta_csv": os.path.abspath(delta_csv),
                    "delta_agg_csv": os.path.abspath(delta_agg_csv),
                    "counterexample_csv": os.path.abspath(bad_csv),
                    "tsne_csv": os.path.abspath(tsne_csv),
                    "summary_md": os.path.abspath(md_path),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(os.path.abspath(metric_all_csv))
    print(os.path.abspath(agg_family_csv))
    print(os.path.abspath(delta_csv))
    print(os.path.abspath(delta_agg_csv))
    print(os.path.abspath(bad_csv))
    print(os.path.abspath(tsne_csv))
    print(os.path.abspath(md_path))
    print(os.path.abspath(summary_json))


if __name__ == "__main__":
    main()
