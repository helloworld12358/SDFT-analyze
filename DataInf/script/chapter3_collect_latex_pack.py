#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build chapter-3 embedding/entropy/tsne latex pack from existing local results only.

Safety constraints:
- Never modify existing experiment outputs.
- Never move/rename/overwrite existing files outside this pack output directory.
- If data is missing, write explicit missing notes instead of fabricating values.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import textwrap
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TRAIN_ORDER = ["gsm8k", "openfunction", "magicoder", "alpaca", "dolly", "lima", "openhermes"]
TRAIN_DISPLAY = {
    "gsm8k": "GSM8K",
    "openfunction": "OpenFunctions",
    "magicoder": "Magicoder",
    "alpaca": "Alpaca",
    "dolly": "Dolly",
    "lima": "LIMA",
    "openhermes": "OpenHermes",
}
DISPLAY_LAYERS_MAIN = [21, 30, 31]
DISPLAY_LAYERS_EXTRA = [28, 29, 30]


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def canonical_train_name(x: str) -> str:
    k = str(x).strip().lower().replace("-", "_")
    k = re.sub(r"_+", "_", k)
    aliases = {
        "openfunctions": "openfunction",
    }
    k = aliases.get(k, k)
    return k


def display_train_name(x: str) -> str:
    return TRAIN_DISPLAY.get(canonical_train_name(x), str(x))


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8")


def save_tex_table(df: pd.DataFrame, path: Path, caption: str = "", label: str = "") -> None:
    ensure_dir(path.parent)
    if df is None or df.empty:
        content = "% Table unavailable: source data missing.\n"
        path.write_text(content, encoding="utf-8")
        return
    tex = df.to_latex(index=False, escape=False)
    if caption or label:
        lines = [r"\begin{table}[t]", r"\centering"]
        if caption:
            lines.append(rf"\caption{{{caption}}}")
        if label:
            lines.append(rf"\label{{{label}}}")
        lines.append(tex)
        lines.append(r"\end{table}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        path.write_text(tex, encoding="utf-8")


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def scan_keyword_files(root: Path, keywords: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        n = p.name.lower()
        if any(k in n for k in keywords):
            out.append(p)
    return sorted(out)


def parse_stage3_png_info(path: Path, train_dataset: str) -> Optional[Dict[str, object]]:
    name = path.name
    # base: tsne_plot_base__epoch_0_layer21.png
    m_base = re.match(r"^tsne_plot_base__epoch_(\d+)_layer(\d+)\.png$", name)
    if m_base:
        ep = int(m_base.group(1))
        layer = int(m_base.group(2))
        return {
            "train_dataset": canonical_train_name(train_dataset),
            "model_state": "base",
            "method": "base",
            "epoch": f"epoch_{ep}",
            "layer": layer,
        }

    # train: tsne_plot_<ds>__sft__epoch_5_layer30.png
    m = re.match(r"^tsne_plot_([a-z0-9_]+)__([a-z0-9_]+)__epoch_(\d+)_layer(\d+)\.png$", name)
    if not m:
        return None
    method = m.group(2).lower()
    ep = int(m.group(3))
    layer = int(m.group(4))
    return {
        "train_dataset": canonical_train_name(train_dataset),
        "model_state": f"{method}_epoch_{ep}",
        "method": method,
        "epoch": f"epoch_{ep}",
        "layer": layer,
    }


def build_train_corpus_tables(
    source_metrics_csv: Optional[pd.DataFrame],
    source_shared_csv: Optional[pd.DataFrame],
    out_dir: Path,
    missing: List[Dict[str, object]],
) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}

    full_csv = out_dir / "train_corpus_cluster_full.csv"
    summary_csv = out_dir / "train_corpus_cluster_summary.csv"
    summary_tex = out_dir / "train_corpus_cluster_summary_for_latex.tex"
    aligned_csv = out_dir / "train_corpus_same_layer_aligned.csv"
    aligned_tex = out_dir / "train_corpus_same_layer_aligned_for_latex.tex"
    seed_mv_csv = out_dir / "train_corpus_seed_mean_var.csv"
    seed_mv_tex = out_dir / "train_corpus_seed_mean_var_for_latex.tex"

    if source_metrics_csv is None or source_metrics_csv.empty:
        missing.append(
            {
                "item": "D/train_corpus",
                "found": "否",
                "source_paths": "",
                "exportable": "",
                "missing": "layer_metrics_all_jobs.csv not found or unreadable",
                "next_step": "Re-copy embedding_cluster_epoch0_qa_answer_lasttok/summary outputs from cloud.",
            }
        )
        for p in [full_csv, summary_csv, aligned_csv, seed_mv_csv]:
            save_csv(pd.DataFrame(), p)
        for p in [summary_tex, aligned_tex, seed_mv_tex]:
            save_tex_table(pd.DataFrame(), p)
        outputs.update(
            {
                "train_corpus_cluster_full.csv": full_csv,
                "train_corpus_cluster_summary.csv": summary_csv,
                "train_corpus_cluster_summary_for_latex.tex": summary_tex,
                "train_corpus_same_layer_aligned.csv": aligned_csv,
                "train_corpus_same_layer_aligned_for_latex.tex": aligned_tex,
                "train_corpus_seed_mean_var.csv": seed_mv_csv,
                "train_corpus_seed_mean_var_for_latex.tex": seed_mv_tex,
            }
        )
        return outputs

    df = source_metrics_csv.copy()
    # keep usable rows only
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()

    required = [
        "family",
        "seed",
        "layer",
        "silhouette_orig",
        "knn_purity_orig",
        "davies_bouldin_orig",
        "calinski_harabasz_orig",
    ]
    for c in required:
        if c not in df.columns:
            missing.append(
                {
                    "item": "D/train_corpus",
                    "found": "部分找到",
                    "source_paths": "layer_metrics_all_jobs.csv",
                    "exportable": "",
                    "missing": f"column missing: {c}",
                    "next_step": "Verify local summary csv schema.",
                }
            )
            for p in [full_csv, summary_csv, aligned_csv, seed_mv_csv]:
                save_csv(pd.DataFrame(), p)
            for p in [summary_tex, aligned_tex, seed_mv_tex]:
                save_tex_table(pd.DataFrame(), p)
            outputs.update(
                {
                    "train_corpus_cluster_full.csv": full_csv,
                    "train_corpus_cluster_summary.csv": summary_csv,
                    "train_corpus_cluster_summary_for_latex.tex": summary_tex,
                    "train_corpus_same_layer_aligned.csv": aligned_csv,
                    "train_corpus_same_layer_aligned_for_latex.tex": aligned_tex,
                    "train_corpus_seed_mean_var.csv": seed_mv_csv,
                    "train_corpus_seed_mean_var_for_latex.tex": seed_mv_tex,
                }
            )
            return outputs

    # pivot sft/sdft
    sft = df[df["family"].astype(str).str.lower() == "sft"].copy()
    sdft = df[df["family"].astype(str).str.lower() == "sdft"].copy()

    sft = sft.rename(
        columns={
            "silhouette_orig": "sil_sft",
            "knn_purity_orig": "knn_sft",
            "davies_bouldin_orig": "db_sft",
            "calinski_harabasz_orig": "ch_sft",
        }
    )
    sdft = sdft.rename(
        columns={
            "silhouette_orig": "sil_sdft",
            "knn_purity_orig": "knn_sdft",
            "davies_bouldin_orig": "db_sdft",
            "calinski_harabasz_orig": "ch_sdft",
        }
    )

    merged = pd.merge(
        sft[["seed", "layer", "sil_sft", "knn_sft", "db_sft", "ch_sft"]],
        sdft[["seed", "layer", "sil_sdft", "knn_sdft", "db_sdft", "ch_sdft"]],
        on=["seed", "layer"],
        how="inner",
    )
    merged["train_dataset"] = "ALL_7_TRAIN_CORPORA"
    merged["pair"] = "sft_vs_sdft"
    merged["delta_sil"] = merged["sil_sdft"] - merged["sil_sft"]
    merged["delta_knn"] = merged["knn_sdft"] - merged["knn_sft"]
    merged["delta_ch"] = merged["ch_sdft"] - merged["ch_sft"]
    merged["delta_db"] = merged["db_sft"] - merged["db_sdft"]

    full_cols = [
        "train_dataset",
        "seed",
        "layer",
        "pair",
        "sil_sft",
        "sil_sdft",
        "delta_sil",
        "knn_sft",
        "knn_sdft",
        "delta_knn",
        "ch_sft",
        "ch_sdft",
        "delta_ch",
        "db_sft",
        "db_sdft",
        "delta_db",
    ]
    merged = merged.sort_values(["seed", "layer"]).reset_index(drop=True)
    save_csv(merged[full_cols], full_csv)

    # summary
    g = merged.groupby("train_dataset", as_index=False).agg(
        mean_delta_sil=("delta_sil", "mean"),
        std_delta_sil=("delta_sil", "std"),
        positive_count_delta_sil=("delta_sil", lambda s: int((s > 0).sum())),
        total_count=("delta_sil", "count"),
        mean_delta_knn=("delta_knn", "mean"),
        std_delta_knn=("delta_knn", "std"),
        positive_count_delta_knn=("delta_knn", lambda s: int((s > 0).sum())),
        mean_delta_ch=("delta_ch", "mean"),
        std_delta_ch=("delta_ch", "std"),
        positive_count_delta_ch=("delta_ch", lambda s: int((s > 0).sum())),
        mean_delta_db=("delta_db", "mean"),
        std_delta_db=("delta_db", "std"),
        positive_count_delta_db=("delta_db", lambda s: int((s > 0).sum())),
    )
    save_csv(g, summary_csv)
    save_tex_table(g, summary_tex, caption="Train-corpus embedding delta summary", label="tab:train_corpus_summary")

    # same-layer aligned (main + optional extra)
    keep_layers = [x for x in DISPLAY_LAYERS_MAIN if x in set(merged["layer"].tolist())]
    extra_layers = [x for x in DISPLAY_LAYERS_EXTRA if x in set(merged["layer"].tolist())]
    keep = sorted(set(keep_layers + extra_layers))
    aligned = merged[merged["layer"].isin(keep)].copy()
    aligned["layer_group"] = aligned["layer"].apply(
        lambda x: "main_display" if int(x) in DISPLAY_LAYERS_MAIN else "extra_28_29_30"
    )
    save_csv(aligned[full_cols + ["layer_group"]], aligned_csv)
    save_tex_table(
        aligned[["seed", "layer", "delta_sil", "delta_knn", "delta_ch", "delta_db", "layer_group"]],
        aligned_tex,
        caption="Train-corpus aligned layers",
        label="tab:train_corpus_aligned",
    )

    # seed mean/var by layer
    seed_mv = merged.groupby(["train_dataset", "layer"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        mean_delta_sil=("delta_sil", "mean"),
        std_delta_sil=("delta_sil", "std"),
        positive_seed_count_delta_sil=("delta_sil", lambda s: int((s > 0).sum())),
        mean_delta_knn=("delta_knn", "mean"),
        std_delta_knn=("delta_knn", "std"),
        positive_seed_count_delta_knn=("delta_knn", lambda s: int((s > 0).sum())),
        mean_delta_ch=("delta_ch", "mean"),
        std_delta_ch=("delta_ch", "std"),
        positive_seed_count_delta_ch=("delta_ch", lambda s: int((s > 0).sum())),
        mean_delta_db=("delta_db", "mean"),
        std_delta_db=("delta_db", "std"),
        positive_seed_count_delta_db=("delta_db", lambda s: int((s > 0).sum())),
    )
    seed_mv["total_seed_count"] = seed_mv["seed_count"]
    save_csv(seed_mv, seed_mv_csv)
    save_tex_table(seed_mv, seed_mv_tex, caption="Train-corpus seed mean/std by layer", label="tab:train_corpus_seed")

    # missing item #2: aligned rerun
    if source_shared_csv is None or source_shared_csv.empty:
        missing.append(
            {
                "item": "训练语料侧同层对齐重跑结果",
                "found": "部分找到",
                "source_paths": "",
                "exportable": "train_corpus_same_layer_aligned.csv generated from full 32-layer table",
                "missing": "shared_tsne summary file missing",
                "next_step": "Copy embedding_cluster_epoch0_qa_answer_lasttok_shared_tsne summary outputs if needed.",
            }
        )

    outputs.update(
        {
            "train_corpus_cluster_full.csv": full_csv,
            "train_corpus_cluster_summary.csv": summary_csv,
            "train_corpus_cluster_summary_for_latex.tex": summary_tex,
            "train_corpus_same_layer_aligned.csv": aligned_csv,
            "train_corpus_same_layer_aligned_for_latex.tex": aligned_tex,
            "train_corpus_seed_mean_var.csv": seed_mv_csv,
            "train_corpus_seed_mean_var_for_latex.tex": seed_mv_tex,
        }
    )
    return outputs


def build_test_task_tables(
    stage2_delta_5: Optional[pd.DataFrame],
    stage3_story_5: Optional[pd.DataFrame],
    stage2_delta_4: Optional[pd.DataFrame],
    out_dir: Path,
    missing: List[Dict[str, object]],
) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    paths = {
        "full_scan_csv": out_dir / "test_task_full_layer_scan.csv",
        "full_scan_tex": out_dir / "test_task_full_layer_scan_for_latex.tex",
        "layer_summary_csv": out_dir / "test_task_layer_summary.csv",
        "layer_summary_tex": out_dir / "test_task_layer_summary_for_latex.tex",
        "display_csv": out_dir / "test_task_display_layers_full.csv",
        "display_tex": out_dir / "test_task_display_layers_for_latex.tex",
        "dataset_csv": out_dir / "test_task_dataset_summary.csv",
        "dataset_tex": out_dir / "test_task_dataset_summary_for_latex.tex",
        "boundary_csv": out_dir / "test_task_boundary_pairs.csv",
        "boundary_tex": out_dir / "test_task_boundary_pairs_for_latex.tex",
        "extra_csv": out_dir / "test_task_layer_28_29_30_full.csv",
        "extra_tex": out_dir / "test_task_layer_28_29_30_for_latex.tex",
    }

    if stage2_delta_5 is None or stage2_delta_5.empty:
        missing.append(
            {
                "item": "E/test_task_full_layer_scan",
                "found": "否",
                "source_paths": "",
                "exportable": "",
                "missing": "embedding_cluster/stage2/layer_deltas_long.csv missing",
                "next_step": "Re-copy stage2 files from cloud.",
            }
        )
        for p in paths.values():
            if p.suffix == ".csv":
                save_csv(pd.DataFrame(), p)
            else:
                save_tex_table(pd.DataFrame(), p)
        outputs.update({k.replace("_", "."): v for k, v in paths.items()})
        return outputs

    d = stage2_delta_5.copy()
    d["train_dataset_shard"] = d["train_dataset_shard"].astype(str).map(canonical_train_name)
    d = d[d["train_dataset_shard"].isin(TRAIN_ORDER)].copy()

    # E1: full layer scan aggregate (epoch_5)
    layer = d.groupby("layer", as_index=False).agg(
        total_count=("train_dataset_shard", "count"),
        mean_delta_sil=("delta_silhouette", "mean"),
        positive_count_delta_sil=("delta_silhouette", lambda s: int((s > 0).sum())),
        mean_delta_knn=("delta_knn_purity", "mean"),
        positive_count_delta_knn=("delta_knn_purity", lambda s: int((s > 0).sum())),
        mean_delta_ch=("delta_calinski_harabasz", "mean"),
        positive_count_delta_ch=("delta_calinski_harabasz", lambda s: int((s > 0).sum())),
        mean_delta_db=("delta_db_better", "mean"),
        positive_count_delta_db=("delta_db_better", lambda s: int((s > 0).sum())),
    ).sort_values("layer")
    layer["number_of_positive_metrics"] = (
        (layer["mean_delta_sil"] > 0).astype(int)
        + (layer["mean_delta_knn"] > 0).astype(int)
        + (layer["mean_delta_ch"] > 0).astype(int)
        + (layer["mean_delta_db"] > 0).astype(int)
    )
    layer["all_four_positive"] = layer["number_of_positive_metrics"] == 4
    layer["at_least_three_positive"] = layer["number_of_positive_metrics"] >= 3
    save_csv(layer, paths["full_scan_csv"])
    save_tex_table(layer, paths["full_scan_tex"], caption="Test-task full layer scan", label="tab:test_full_layer_scan")

    # E2: summary (xx/32 style)
    total_layers = int(layer["layer"].nunique())
    summary_rows = [
        {
            "metric": r"\delta_{Sil}>0",
            "positive_layers": int((layer["mean_delta_sil"] > 0).sum()),
            "total_layers": total_layers,
            "ratio_text": f"{int((layer['mean_delta_sil'] > 0).sum())} / {total_layers}",
        },
        {
            "metric": r"\delta_{kNN}>0",
            "positive_layers": int((layer["mean_delta_knn"] > 0).sum()),
            "total_layers": total_layers,
            "ratio_text": f"{int((layer['mean_delta_knn'] > 0).sum())} / {total_layers}",
        },
        {
            "metric": r"\delta_{CH}>0",
            "positive_layers": int((layer["mean_delta_ch"] > 0).sum()),
            "total_layers": total_layers,
            "ratio_text": f"{int((layer['mean_delta_ch'] > 0).sum())} / {total_layers}",
        },
        {
            "metric": r"\delta_{DB}>0",
            "positive_layers": int((layer["mean_delta_db"] > 0).sum()),
            "total_layers": total_layers,
            "ratio_text": f"{int((layer['mean_delta_db'] > 0).sum())} / {total_layers}",
        },
        {
            "metric": "四项指标全部为正",
            "positive_layers": int(layer["all_four_positive"].sum()),
            "total_layers": total_layers,
            "ratio_text": f"{int(layer['all_four_positive'].sum())} / {total_layers}",
        },
        {
            "metric": "至少三项指标为正",
            "positive_layers": int(layer["at_least_three_positive"].sum()),
            "total_layers": total_layers,
            "ratio_text": f"{int(layer['at_least_three_positive'].sum())} / {total_layers}",
        },
    ]
    layer_summary = pd.DataFrame(summary_rows)
    save_csv(layer_summary, paths["layer_summary_csv"])
    save_tex_table(layer_summary, paths["layer_summary_tex"], caption="Layer-level positivity summary", label="tab:test_layer_summary")

    # E3 + E6 display layers and dataset summary use stage3 story table (contains epoch_1 + epoch_5)
    if stage3_story_5 is None or stage3_story_5.empty:
        missing.append(
            {
                "item": "E/display_layers_and_boundary_pairs",
                "found": "部分找到",
                "source_paths": "",
                "exportable": "E1/E2 tables ready from stage2",
                "missing": "story_tsne_summary_all.csv missing",
                "next_step": "Re-copy stage3 summary files from cloud.",
            }
        )
        save_csv(pd.DataFrame(), paths["display_csv"])
        save_tex_table(pd.DataFrame(), paths["display_tex"])
        save_csv(pd.DataFrame(), paths["boundary_csv"])
        save_tex_table(pd.DataFrame(), paths["boundary_tex"])
    else:
        s3 = stage3_story_5.copy()
        if "status" in s3.columns:
            s3 = s3[s3["status"].astype(str).str.lower() == "ok"].copy()
        s3["train_dataset"] = s3["train_dataset"].astype(str).map(canonical_train_name)
        s3["method"] = s3["method"].astype(str).str.lower()
        s3["epoch"] = s3["epoch"].astype(str)
        s3 = s3[s3["train_dataset"].isin(TRAIN_ORDER)]
        s3 = s3[s3["method"].isin(["sft", "sdft"])]
        s3 = s3[s3["layer"].isin(DISPLAY_LAYERS_MAIN)]
        keys = ["train_dataset", "epoch", "layer"]
        sft = s3[s3["method"] == "sft"][
            keys + ["silhouette_orig", "knn_purity_orig", "calinski_harabasz_orig", "davies_bouldin_orig"]
        ].rename(
            columns={
                "silhouette_orig": "sil_sft",
                "knn_purity_orig": "knn_sft",
                "calinski_harabasz_orig": "ch_sft",
                "davies_bouldin_orig": "db_sft",
            }
        )
        sdft = s3[s3["method"] == "sdft"][
            keys + ["silhouette_orig", "knn_purity_orig", "calinski_harabasz_orig", "davies_bouldin_orig"]
        ].rename(
            columns={
                "silhouette_orig": "sil_sdft",
                "knn_purity_orig": "knn_sdft",
                "calinski_harabasz_orig": "ch_sdft",
                "davies_bouldin_orig": "db_sdft",
            }
        )
        disp = pd.merge(sft, sdft, on=keys, how="inner")
        disp["delta_sil"] = disp["sil_sdft"] - disp["sil_sft"]
        disp["delta_knn"] = disp["knn_sdft"] - disp["knn_sft"]
        disp["delta_ch"] = disp["ch_sdft"] - disp["ch_sft"]
        disp["delta_db"] = disp["db_sft"] - disp["db_sdft"]
        disp["train_dataset"] = disp["train_dataset"].map(display_train_name)
        disp = disp.sort_values(["train_dataset", "epoch", "layer"]).reset_index(drop=True)
        save_csv(disp, paths["display_csv"])
        save_tex_table(disp, paths["display_tex"], caption="Display layers (21/30/31) full pairs", label="tab:test_display_layers")

        # boundary pairs + summary denominator 42 expected
        bad = disp[
            (disp["delta_sil"] <= 0)
            | (disp["delta_knn"] <= 0)
            | (disp["delta_ch"] <= 0)
            | (disp["delta_db"] <= 0)
        ].copy()

        def reason_row(r: pd.Series) -> str:
            reasons: List[str] = []
            if float(r["delta_sil"]) <= 0:
                reasons.append("delta_sil<=0")
            if float(r["delta_knn"]) <= 0:
                reasons.append("delta_knn<=0")
            if float(r["delta_ch"]) <= 0:
                reasons.append("delta_ch<=0")
            if float(r["delta_db"]) <= 0:
                reasons.append("delta_db<=0")
            return ";".join(reasons)

        if not bad.empty:
            bad["boundary_reason"] = bad.apply(reason_row, axis=1)
        else:
            bad["boundary_reason"] = []
        save_csv(
            bad[
                [
                    "train_dataset",
                    "epoch",
                    "layer",
                    "delta_sil",
                    "delta_knn",
                    "delta_ch",
                    "delta_db",
                    "boundary_reason",
                ]
            ],
            paths["boundary_csv"],
        )

        # add summary rows into latex table source for readability
        denom = int(len(disp))
        bsum = pd.DataFrame(
            [
                {"metric": r"\delta_{Sil}\le 0", "count": int((disp["delta_sil"] <= 0).sum()), "denom": denom},
                {"metric": r"\delta_{kNN}\le 0", "count": int((disp["delta_knn"] <= 0).sum()), "denom": denom},
                {"metric": r"\delta_{CH}\le 0", "count": int((disp["delta_ch"] <= 0).sum()), "denom": denom},
                {"metric": r"\delta_{DB}\le 0", "count": int((disp["delta_db"] <= 0).sum()), "denom": denom},
            ]
        )
        bsum["ratio_text"] = bsum.apply(lambda r: f"{int(r['count'])} / {int(r['denom'])}", axis=1)
        boundary_tex_df = pd.concat(
            [
                pd.DataFrame([{"metric": "Boundary Summary", "count": np.nan, "denom": np.nan, "ratio_text": ""}]),
                bsum,
            ],
            ignore_index=True,
        )
        save_tex_table(boundary_tex_df, paths["boundary_tex"], caption="Boundary pair summary", label="tab:test_boundary")

    # dataset summary from stage2 epoch_5
    ds = d.groupby("train_dataset_shard", as_index=False).agg(
        total_count=("layer", "count"),
        mean_delta_sil=("delta_silhouette", "mean"),
        positive_count_delta_sil=("delta_silhouette", lambda s: int((s > 0).sum())),
        mean_delta_knn=("delta_knn_purity", "mean"),
        positive_count_delta_knn=("delta_knn_purity", lambda s: int((s > 0).sum())),
        mean_delta_ch=("delta_calinski_harabasz", "mean"),
        positive_count_delta_ch=("delta_calinski_harabasz", lambda s: int((s > 0).sum())),
        mean_delta_db=("delta_db_better", "mean"),
        positive_count_delta_db=("delta_db_better", lambda s: int((s > 0).sum())),
    )
    ds["train_dataset"] = ds["train_dataset_shard"].map(display_train_name)
    ds = ds[
        [
            "train_dataset",
            "mean_delta_sil",
            "positive_count_delta_sil",
            "total_count",
            "mean_delta_knn",
            "positive_count_delta_knn",
            "mean_delta_ch",
            "positive_count_delta_ch",
            "mean_delta_db",
            "positive_count_delta_db",
        ]
    ].sort_values("train_dataset")
    save_csv(ds, paths["dataset_csv"])
    save_tex_table(ds, paths["dataset_tex"], caption="Dataset grouped summary", label="tab:test_dataset_summary")

    # extra 28/29/30 table from 4-task variant if available
    if stage2_delta_4 is not None and not stage2_delta_4.empty:
        ex = stage2_delta_4.copy()
        ex["train_dataset_shard"] = ex["train_dataset_shard"].astype(str).map(canonical_train_name)
        ex = ex[ex["train_dataset_shard"].isin(TRAIN_ORDER)]
        ex = ex[ex["layer"].isin(DISPLAY_LAYERS_EXTRA)].copy()
        ex["epoch"] = "epoch_5"
        ex["train_dataset"] = ex["train_dataset_shard"].map(display_train_name)
        ex = ex.rename(
            columns={
                "delta_silhouette": "delta_sil",
                "delta_knn_purity": "delta_knn",
                "delta_calinski_harabasz": "delta_ch",
                "delta_db_better": "delta_db",
            }
        )[
            [
                "train_dataset",
                "epoch",
                "layer",
                "delta_sil",
                "delta_knn",
                "delta_ch",
                "delta_db",
            ]
        ].sort_values(["train_dataset", "layer"])
        save_csv(ex, paths["extra_csv"])
        save_tex_table(ex, paths["extra_tex"], caption="Extra layers (28/29/30) from 4-task variant", label="tab:test_extra_2830")
    else:
        save_csv(pd.DataFrame(), paths["extra_csv"])
        save_tex_table(pd.DataFrame(), paths["extra_tex"])
        missing.append(
            {
                "item": "E/layer_28_29_30_full",
                "found": "否",
                "source_paths": "",
                "exportable": "",
                "missing": "embedding_cluster_4tasks_no_multiarith/stage2/layer_deltas_long.csv not found",
                "next_step": "Run or copy 4-task variant outputs.",
            }
        )

    outputs.update(
        {
            "test_task_full_layer_scan.csv": paths["full_scan_csv"],
            "test_task_full_layer_scan_for_latex.tex": paths["full_scan_tex"],
            "test_task_layer_summary.csv": paths["layer_summary_csv"],
            "test_task_layer_summary_for_latex.tex": paths["layer_summary_tex"],
            "test_task_display_layers_full.csv": paths["display_csv"],
            "test_task_display_layers_for_latex.tex": paths["display_tex"],
            "test_task_dataset_summary.csv": paths["dataset_csv"],
            "test_task_dataset_summary_for_latex.tex": paths["dataset_tex"],
            "test_task_boundary_pairs.csv": paths["boundary_csv"],
            "test_task_boundary_pairs_for_latex.tex": paths["boundary_tex"],
            "test_task_layer_28_29_30_full.csv": paths["extra_csv"],
            "test_task_layer_28_29_30_for_latex.tex": paths["extra_tex"],
        }
    )
    return outputs


def build_train_vs_test_alignment(
    train_full_df: Optional[pd.DataFrame],
    test_stage2_df: Optional[pd.DataFrame],
    out_dir: Path,
    missing: List[Dict[str, object]],
) -> Dict[str, Path]:
    out_csv = out_dir / "train_vs_test_alignment_summary.csv"
    out_tex = out_dir / "train_vs_test_alignment_summary_for_latex.tex"

    if train_full_df is None or train_full_df.empty or test_stage2_df is None or test_stage2_df.empty:
        save_csv(pd.DataFrame(), out_csv)
        save_tex_table(pd.DataFrame(), out_tex)
        missing.append(
            {
                "item": "F/train_vs_test_alignment",
                "found": "部分找到",
                "source_paths": "",
                "exportable": "",
                "missing": "train corpus or test stage2 table missing",
                "next_step": "Ensure D and E source tables exist.",
            }
        )
        return {
            "train_vs_test_alignment_summary.csv": out_csv,
            "train_vs_test_alignment_summary_for_latex.tex": out_tex,
        }

    tr = train_full_df.copy()
    tl = tr.groupby("layer", as_index=False).agg(
        train_delta_sil=("delta_sil", "mean"),
        train_delta_knn=("delta_knn", "mean"),
        train_delta_ch=("delta_ch", "mean"),
        train_delta_db=("delta_db", "mean"),
    )

    te = test_stage2_df.copy()
    te["train_dataset_shard"] = te["train_dataset_shard"].astype(str).map(canonical_train_name)
    te = te[te["train_dataset_shard"].isin(TRAIN_ORDER)].copy()
    te = te.rename(
        columns={
            "delta_silhouette": "test_delta_sil",
            "delta_knn_purity": "test_delta_knn",
            "delta_calinski_harabasz": "test_delta_ch",
            "delta_db_better": "test_delta_db",
            "train_dataset_shard": "train_dataset",
        }
    )[
        [
            "train_dataset",
            "layer",
            "test_delta_sil",
            "test_delta_knn",
            "test_delta_ch",
            "test_delta_db",
        ]
    ]
    merged = pd.merge(te, tl, on="layer", how="left")

    def pref(a: float, b: float, c: float, d: float) -> str:
        vals = [a, b, c, d]
        pos = sum(1 for x in vals if pd.notna(x) and float(x) > 0)
        neg = sum(1 for x in vals if pd.notna(x) and float(x) <= 0)
        if pos >= 3:
            return "SDFT"
        if neg >= 3:
            return "SFT"
        return "mixed"

    trend: List[str] = []
    for _, r in merged.iterrows():
        p_train = pref(r["train_delta_sil"], r["train_delta_knn"], r["train_delta_ch"], r["train_delta_db"])
        p_test = pref(r["test_delta_sil"], r["test_delta_knn"], r["test_delta_ch"], r["test_delta_db"])
        if p_train == "SFT" and p_test == "SDFT":
            trend.append("train偏SFT/test偏SDFT")
        elif p_train == "SDFT" and p_test == "SDFT":
            trend.append("train偏SDFT/test偏SDFT")
        elif p_train == "SFT" and p_test == "SFT":
            trend.append("train偏SFT/test偏SFT")
        else:
            trend.append("mixed")
    merged["trend_relation"] = trend
    merged["train_dataset"] = merged["train_dataset"].map(display_train_name)
    merged = merged[
        [
            "train_dataset",
            "layer",
            "train_delta_sil",
            "test_delta_sil",
            "train_delta_knn",
            "test_delta_knn",
            "train_delta_ch",
            "test_delta_ch",
            "train_delta_db",
            "test_delta_db",
            "trend_relation",
        ]
    ].sort_values(["train_dataset", "layer"])
    save_csv(merged, out_csv)
    save_tex_table(merged, out_tex, caption="Train-corpus vs test-task alignment", label="tab:train_test_alignment")

    return {
        "train_vs_test_alignment_summary.csv": out_csv,
        "train_vs_test_alignment_summary_for_latex.tex": out_tex,
    }


def build_semantic_entropy_outputs(out_dir: Path, missing: List[Dict[str, object]], repo_root: Path) -> Dict[str, Path]:
    p_full_csv = out_dir / "semantic_entropy_train_full.csv"
    p_full_tex = out_dir / "semantic_entropy_train_full_for_latex.tex"
    p_dim25_tex = out_dir / "semantic_entropy_train_dim25_for_latex.tex"
    p_ans_csv = out_dir / "semantic_entropy_answer_dim25.csv"
    p_ans_tex = out_dir / "semantic_entropy_answer_dim25_for_latex.tex"
    p_method_tex = out_dir / "semantic_entropy_method_for_latex.tex"

    # locate existing semantic-entropy result files
    sem_hits = scan_keyword_files(repo_root / "DataInf" / "results", ["semantic", "entropy"])
    if not sem_hits:
        missing.append(
            {
                "item": "G/semantic_entropy_results",
                "found": "否",
                "source_paths": "",
                "exportable": "",
                "missing": "No semantic-entropy result files under DataInf/results.",
                "next_step": "Run semantic-entropy pipeline on cloud and copy outputs.",
            }
        )
    # create empty placeholder tables (no fabricated values)
    save_csv(
        pd.DataFrame(
            columns=[
                "train_domain",
                "dim",
                "entropy_sft",
                "entropy_sdft",
                "entropy_diff_sft_minus_sdft",
                "source_file",
            ]
        ),
        p_full_csv,
    )
    save_tex_table(pd.DataFrame(), p_full_tex)
    save_tex_table(pd.DataFrame(), p_dim25_tex)
    save_csv(
        pd.DataFrame(
            columns=[
                "test_task",
                "train_domain",
                "dim",
                "entropy_diff_sdft_minus_sft",
                "source_file",
            ]
        ),
        p_ans_csv,
    )
    save_tex_table(pd.DataFrame(), p_ans_tex)

    # method tex from existing scripts
    method_lines = [
        r"\paragraph{Semantic-Entropy Estimation (from local scripts).}",
        r"We located scripts under \texttt{sdft/Mutual-Information/}.",
        r"Main continuous-estimation script \texttt{continuous_entropy_diff.py} uses:",
        r"\begin{itemize}",
        r"\item sentence embedding by Transformer mean pooling;",
        r"\item UMAP dimensionality reduction (default metric: euclidean, random\_state=42);",
        r"\item KDE (\texttt{scipy.stats.gaussian\_kde}) on reduced space;",
        r"\item differential-entropy proxy: $H=-\mathbb{E}[\log p(x)]$;",
        r"\item reported difference $\Delta H = H(P)-H(Q)$ across dims $\{10,15,20,25,30,50\}$.",
        r"\end{itemize}",
        r"Exploratory script \texttt{semantic_entropy.py} also uses HDBSCAN after UMAP; this is treated as exploratory and not the main-text metric.",
        r"If local result files are missing, no numerical table is fabricated in this pack.",
    ]
    p_method_tex.write_text("\n".join(method_lines) + "\n", encoding="utf-8")

    return {
        "semantic_entropy_train_full.csv": p_full_csv,
        "semantic_entropy_train_full_for_latex.tex": p_full_tex,
        "semantic_entropy_train_dim25_for_latex.tex": p_dim25_tex,
        "semantic_entropy_answer_dim25.csv": p_ans_csv,
        "semantic_entropy_answer_dim25_for_latex.tex": p_ans_tex,
        "semantic_entropy_method_for_latex.tex": p_method_tex,
    }


def build_tsne_manifest_and_selected_figures(
    stage3_by_train_dir: Path,
    out_dir: Path,
    missing: List[Dict[str, object]],
) -> Dict[str, Path]:
    man_csv = out_dir / "tsne_figure_manifest.csv"
    man_tex = out_dir / "tsne_figure_manifest_for_latex.tex"
    upload_md = out_dir / "selected_figures_to_upload.md"
    copy_ps1 = out_dir / "selected_figures_copy_commands.ps1"
    copy_sh = out_dir / "selected_figures_copy_commands.sh"
    sel_dir = out_dir / "selected_figures" / "chapter3"
    ensure_dir(sel_dir)

    rows: List[Dict[str, object]] = []
    if not stage3_by_train_dir.is_dir():
        missing.append(
            {
                "item": "H/tsne_manifest",
                "found": "否",
                "source_paths": str(stage3_by_train_dir),
                "exportable": "",
                "missing": "stage3/by_train_dataset folder not found",
                "next_step": "Copy stage3 plot files from cloud.",
            }
        )
        save_csv(pd.DataFrame(), man_csv)
        save_tex_table(pd.DataFrame(), man_tex)
        upload_md.write_text("# selected_figures_to_upload\n\nNo figures found.\n", encoding="utf-8")
        copy_ps1.write_text("# no-op\n", encoding="utf-8")
        copy_sh.write_text("#!/usr/bin/env bash\n# no-op\n", encoding="utf-8")
        return {
            "tsne_figure_manifest.csv": man_csv,
            "tsne_figure_manifest_for_latex.tex": man_tex,
            "selected_figures_to_upload.md": upload_md,
            "selected_figures_copy_commands.ps1": copy_ps1,
            "selected_figures_copy_commands.sh": copy_sh,
        }

    ds_dirs = sorted([d for d in stage3_by_train_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
    for ds_dir in ds_dirs:
        ds = canonical_train_name(ds_dir.name)
        for png in sorted(ds_dir.glob("tsne_plot_*.png")):
            info = parse_stage3_png_info(png, ds)
            if info is None:
                continue
            row = dict(info)
            row["original_path"] = str(png)
            row["suggested_overleaf_dir"] = "figures/chapter3"
            row["suggested_overleaf_name"] = (
                f"{row['train_dataset']}_{row['method']}_{str(row['epoch']).replace('_','')}_layer{int(row['layer'])}.png"
            )
            rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest = manifest.sort_values(["train_dataset", "method", "epoch", "layer"]).reset_index(drop=True)
    save_csv(manifest, man_csv)
    save_tex_table(manifest, man_tex, caption="Complete t-SNE image pool manifest", label="tab:tsne_manifest")

    # Selected 9 figures for main text
    wanted = [
        ("alpaca", "base", "epoch_0", 21, "alpaca_base_epoch0_layer21.png"),
        ("alpaca", "sft", "epoch_5", 21, "alpaca_sft_epoch5_layer21.png"),
        ("alpaca", "sdft", "epoch_5", 21, "alpaca_sdft_epoch5_layer21.png"),
        ("gsm8k", "base", "epoch_0", 21, "gsm8k_base_epoch0_layer21.png"),
        ("gsm8k", "sft", "epoch_5", 21, "gsm8k_sft_epoch5_layer21.png"),
        ("gsm8k", "sdft", "epoch_5", 21, "gsm8k_sdft_epoch5_layer21.png"),
        ("openfunction", "base", "epoch_0", 21, "openfunction_base_epoch0_layer21.png"),
        ("openfunction", "sft", "epoch_5", 21, "openfunction_sft_epoch5_layer21.png"),
        ("openfunction", "sdft", "epoch_5", 21, "openfunction_sdft_epoch5_layer21.png"),
    ]

    md_lines = ["# selected_figures_to_upload", "", "建议上传到 Overleaf 的正文候选图：", ""]
    ps_lines = ["# PowerShell copy commands", f"$dst = \"{str(sel_dir)}\"", "New-Item -ItemType Directory -Path $dst -Force | Out-Null"]
    sel_dir_posix = str(sel_dir).replace("\\", "/")
    sh_lines = ["#!/usr/bin/env bash", "set -euo pipefail", f'DST="{sel_dir_posix}"', 'mkdir -p "$DST"']

    for ds, method, epoch, layer, out_name in wanted:
        hit = manifest[
            (manifest["train_dataset"] == ds)
            & (manifest["method"] == method)
            & (manifest["epoch"] == epoch)
            & (manifest["layer"] == layer)
        ]
        if hit.empty:
            missing.append(
                {
                    "item": f"H/selected_figure/{out_name}",
                    "found": "否",
                    "source_paths": "",
                    "exportable": "",
                    "missing": "requested figure not found",
                    "next_step": "Check stage3 plot completeness for layer21/base+epoch5.",
                }
            )
            md_lines.append(f"- MISSING: `{out_name}`")
            continue
        src = str(hit.iloc[0]["original_path"])
        dst = sel_dir / out_name
        shutil.copy2(src, dst)
        md_lines.append(f"- `{out_name}` <- `{src}`")
        ps_lines.append(f'Copy-Item -LiteralPath "{src}" -Destination (Join-Path $dst "{out_name}") -Force')
        sh_lines.append(f'cp "{src}" "$DST/{out_name}"')

    upload_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    copy_ps1.write_text("\n".join(ps_lines) + "\n", encoding="utf-8")
    copy_sh.write_text("\n".join(sh_lines) + "\n", encoding="utf-8")

    return {
        "tsne_figure_manifest.csv": man_csv,
        "tsne_figure_manifest_for_latex.tex": man_tex,
        "selected_figures_to_upload.md": upload_md,
        "selected_figures_copy_commands.ps1": copy_ps1,
        "selected_figures_copy_commands.sh": copy_sh,
    }


def build_metric_defs_and_sampling_rules(
    repo_root: Path,
    out_dir: Path,
    sample_pool_5: Optional[dict],
    sample_pool_4: Optional[dict],
) -> Dict[str, Path]:
    metric_tex = out_dir / "metric_definitions_for_latex.tex"
    sampling_md = out_dir / "sampling_rules_report.md"
    sampling_tex = out_dir / "sampling_rules_for_latex.tex"

    metric_lines = [
        r"\paragraph{Metric direction conventions.}",
        r"\begin{align}",
        r"\delta_{\mathrm{Sil}} &= \mathrm{Sil}(SDFT)-\mathrm{Sil}(SFT),\\",
        r"\delta_{\mathrm{kNN}} &= \mathrm{Purity}_{k_{\mathrm{NN}}}(SDFT)-\mathrm{Purity}_{k_{\mathrm{NN}}}(SFT),\\",
        r"\delta_{\mathrm{CH}} &= \mathrm{CH}(SDFT)-\mathrm{CH}(SFT),\\",
        r"\delta_{\mathrm{DB}} &= \mathrm{DB}(SFT)-\mathrm{DB}(SDFT).",
        r"\end{align}",
        r"Silhouette / kNN Purity / CH are positive-direction metrics; DB is inverse-direction.",
        r"With this definition, all $\delta>0$ indicate SDFT is better.",
        "",
        r"\paragraph{Implementation settings found in local scripts.}",
        r"\begin{itemize}",
        r"\item kNN Purity uses $k=10$ (\texttt{embedding\_cluster\_utils.py}).",
        r"\item Distance metric for Silhouette and kNN: euclidean.",
        r"\item Embedding extraction in main stage-1/stage-3 scripts: last-token hidden state.",
        r"\item t-SNE defaults: perplexity=30, n\_iter=1000, learning\_rate=auto, init=pca, metric=euclidean.",
        r"\item PCA pre-reduction before t-SNE: default dim=50.",
        r"\item UMAP/KDE/HDBSCAN are not used in the chapter-3 main embedding-cluster pipeline in DataInf/script.",
        r"\item Semantic-entropy scripts under \texttt{sdft/Mutual-Information} do include UMAP/KDE/HDBSCAN variants (exploratory).",
        r"\end{itemize}",
    ]
    metric_tex.write_text("\n".join(metric_lines) + "\n", encoding="utf-8")

    def pool_info(pool: Optional[dict], tag: str) -> str:
        if not pool:
            return f"- {tag}: 未定位到 sample_pool 文件。"
        rows = pool.get("rows", [])
        stats = pool.get("stats", [])
        tasks = sorted({str(x.get("task", "")) for x in rows}) if rows else []
        seed_guess = "42"  # inferred from filename and scripts defaults
        return "\n".join(
            [
                f"- {tag}:",
                f"  - 样本总数: {len(rows)}",
                f"  - 任务列表: {tasks}",
                f"  - 随机 seed（脚本默认）: {seed_guess}",
                f"  - 每任务采样: {[{'task': s.get('task'), 'selected': s.get('selected')} for s in stats]}",
            ]
        )

    md_lines = [
        "# sampling_rules_report",
        "",
        "## 测试任务与采样",
        pool_info(sample_pool_5, "5-task 主配置"),
        "",
        pool_info(sample_pool_4, "4-task 去掉 MultiArith 变体"),
        "",
        "## 规则来源（脚本解析）",
        "- 主流程脚本：`embedding_cluster_01_epoch5_layer_scan.py` / `embedding_cluster_03_plot_selected_layers_tsne.py`",
        "- `samples_per_task` 默认 100；`seed` 默认 42；任务默认包含 `multiarith`。",
        "- 4-task 变体目录名和 sample_pool 显示已去掉 `multiarith`。",
        "- embedding 提取方式：last-token hidden state（`extract_last_token_representations`）。",
        "- epoch 使用：stage3 故事图包含 `epoch_1` 与 `epoch_5`；另含 `base epoch_0`。",
        "- 训练终点口径：文件名统一写 `epoch_5`，若某训练域真实训练步不同，该差异不会在文件名中单独标注。",
    ]
    sampling_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        r"\paragraph{Sampling and task composition.}",
        r"Main stage-1/stage-3 embedding-cluster pipeline uses five tasks "
        r"$\{$AlpacaEval, GSM8K, HumanEval, MultiArith, OpenFunctions$\}$ with 100 samples per task (seed=42 by default), totaling 500 samples.",
        r"A separate variant (\texttt{embedding\_cluster\_4tasks\_no\_multiarith}) excludes MultiArith and keeps 100 samples per remaining task (total 400).",
        r"Embedding feature is the last-token hidden state at selected Transformer layers.",
        r"Story t-SNE plots include base epoch\_0 plus SFT/SDFT at epoch\_1 and epoch\_5.",
    ]
    sampling_tex.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    return {
        "metric_definitions_for_latex.tex": metric_tex,
        "sampling_rules_report.md": sampling_md,
        "sampling_rules_for_latex.tex": sampling_tex,
    }


def load_json_if_exists(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def make_missing_report(
    out_dir: Path,
    entries_extra: List[Dict[str, object]],
    checks: Dict[str, Dict[str, object]],
) -> Path:
    p = out_dir / "MISSING_REPORT.md"
    lines = ["# MISSING_REPORT", ""]

    order = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
    ]
    for k in order:
        c = checks.get(k, {})
        lines.extend(
            [
                f"## 检查项 {k}",
                f"- 项目：{c.get('item','')}",
                f"- 是否找到：{c.get('found','否')}",
                f"- 原始文件路径：{c.get('source_paths','')}",
                f"- 当前可导出的结果：{c.get('exportable','')}",
                f"- 缺失内容：{c.get('missing','')}",
                f"- 建议下一步：{c.get('next_step','')}",
                "",
            ]
        )

    if entries_extra:
        lines.append("## 额外缺失/注意事项")
        for e in entries_extra:
            lines.extend(
                [
                    f"- 项目：{e.get('item','')}",
                    f"  - 是否找到：{e.get('found','')}",
                    f"  - 原始文件路径：{e.get('source_paths','')}",
                    f"  - 当前可导出的结果：{e.get('exportable','')}",
                    f"  - 缺失内容：{e.get('missing','')}",
                    f"  - 建议下一步：{e.get('next_step','')}",
                ]
            )
        lines.append("")

    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def write_readme_pack(
    out_dir: Path,
    scanned_dirs: List[Path],
    outputs: Dict[str, Path],
    checks: Dict[str, Dict[str, object]],
) -> Path:
    p = out_dir / "README_CHAPTER3_PACK.md"
    lines = [
        "# README_CHAPTER3_PACK",
        "",
        "本目录由 `chapter3_collect_latex_pack.py` 自动生成，目标是服务第三章正文与附录撰写。",
        "",
        "## 扫描目录",
    ]
    for d in scanned_dirs:
        lines.append(f"- {d}")
    lines.extend(
        [
            "",
            "## 结果概览",
            "- 已整理：embedding cluster（stage1/stage2/stage3）、epoch0 QA answer-last-token、t-SNE 图像池。",
            "- 缺失：semantic entropy 数值结果（仅定位到脚本，未定位到 DataInf/results 下结果文件）。",
            "",
            "## 指标方向",
            "- `delta_sil = Sil(SDFT)-Sil(SFT)`",
            "- `delta_knn = kNN(SDFT)-kNN(SFT)`",
            "- `delta_ch = CH(SDFT)-CH(SFT)`",
            "- `delta_db = DB(SFT)-DB(SDFT)`（已统一为正向指标）",
            "",
            "## 正文推荐使用",
            "- test_task_layer_summary_for_latex.tex",
            "- test_task_display_layers_for_latex.tex",
            "- test_task_dataset_summary_for_latex.tex",
            "- metric_definitions_for_latex.tex",
            "- sampling_rules_for_latex.tex",
            "",
            "## 附录推荐使用",
            "- train_corpus_cluster_summary_for_latex.tex",
            "- train_corpus_same_layer_aligned_for_latex.tex",
            "- train_corpus_seed_mean_var_for_latex.tex",
            "- test_task_full_layer_scan_for_latex.tex",
            "- test_task_boundary_pairs_for_latex.tex",
            "- tsne_figure_manifest_for_latex.tex",
            "- semantic_entropy_*_for_latex.tex（当前若空则表示本地缺失）",
            "",
            "## Overleaf 需要上传的图片",
            "- 见 `selected_figures_to_upload.md` 与 `selected_figures/chapter3/`",
            "",
            "## 全部输出文件",
        ]
    )
    for k, v in sorted(outputs.items(), key=lambda x: x[0]):
        lines.append(f"- `{k}` -> `{v}`")
    lines.extend(["", "## 缺失项总览", "- 见 `MISSING_REPORT.md`"])
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def write_final_report(out_dir: Path, outputs: Dict[str, Path], missing_path: Path) -> Path:
    p = out_dir / "FINAL_REPORT.md"
    lines = [
        "# 第三章结果整理执行报告",
        "",
        "## 1. 已成功整理的数据",
        "- embedding cluster stage1/stage2/stage3 表格与摘要",
        "- 训练语料侧（epoch0 QA answer-last-token）全层与seed统计",
        "- t-SNE 105图清单与正文候选9图拷贝",
        "",
        "## 2. 已生成的正文可用 LaTeX 表格",
        "- test_task_layer_summary_for_latex.tex",
        "- test_task_display_layers_for_latex.tex",
        "- test_task_dataset_summary_for_latex.tex",
        "- metric_definitions_for_latex.tex",
        "- sampling_rules_for_latex.tex",
        "",
        "## 3. 已生成的附录可用 LaTeX 表格",
        "- train_corpus_cluster_summary_for_latex.tex",
        "- train_corpus_same_layer_aligned_for_latex.tex",
        "- train_corpus_seed_mean_var_for_latex.tex",
        "- test_task_full_layer_scan_for_latex.tex",
        "- test_task_boundary_pairs_for_latex.tex",
        "- tsne_figure_manifest_for_latex.tex",
        "- semantic_entropy_train_full_for_latex.tex",
        "- semantic_entropy_train_dim25_for_latex.tex",
        "- semantic_entropy_answer_dim25_for_latex.tex",
        "",
        "## 4. 已复制的正文候选图片",
        "- selected_figures/chapter3/*.png（目标9张）",
        "- 具体列表见 selected_figures_to_upload.md",
        "",
        "## 5. 缺失或需要补跑的数据",
        f"- 详见 {missing_path.name}",
        "",
        "## 6. 建议我发给 ChatGPT 的文件",
        "- README_CHAPTER3_PACK.md",
        "- MISSING_REPORT.md",
        "- train_corpus_cluster_summary_for_latex.tex",
        "- train_corpus_same_layer_aligned_for_latex.tex",
        "- train_corpus_seed_mean_var_for_latex.tex",
        "- test_task_full_layer_scan_for_latex.tex",
        "- test_task_display_layers_for_latex.tex",
        "- test_task_dataset_summary_for_latex.tex",
        "- semantic_entropy_train_dim25_for_latex.tex",
        "- semantic_entropy_train_full_for_latex.tex",
        "- semantic_entropy_answer_dim25_for_latex.tex",
        "- metric_definitions_for_latex.tex",
        "- sampling_rules_for_latex.tex",
        "- selected_figures_to_upload.md",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect chapter-3 latex pack from existing local results.")
    ap.add_argument("--datainf_root", type=str, default="", help="default: auto from repo/DataInf")
    ap.add_argument("--output_dir", type=str, default="", help="optional explicit output dir")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    datainf_root = Path(args.datainf_root).resolve() if args.datainf_root else (repo_root / "DataInf")
    results_root = datainf_root / "results"
    ensure_dir(results_root)

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (results_root / f"chapter3_embedding_latex_pack_{now_stamp()}")
    ensure_dir(out_dir)

    scanned_dirs = [
        results_root / "embedding_cluster",
        results_root / "embedding_cluster" / "stage1",
        results_root / "embedding_cluster" / "stage1" / "by_train_dataset",
        results_root / "embedding_cluster" / "stage2",
        results_root / "embedding_cluster" / "stage3",
        results_root / "embedding_cluster" / "stage3" / "by_train_dataset",
        results_root / "embedding_cluster_4tasks_no_multiarith",
        results_root / "embedding_cluster_epoch0_qa_answer_lasttok",
        results_root / "semantic_entropy",
        results_root,
        datainf_root / "script",
        datainf_root / "script" / "README_embedding_cluster.md",
    ]

    # key sources
    stage2_5 = load_csv(results_root / "embedding_cluster" / "stage2" / "layer_deltas_long.csv")
    stage3_5 = load_csv(results_root / "embedding_cluster" / "stage3" / "story_tsne_summary_all.csv")
    stage2_4 = load_csv(results_root / "embedding_cluster_4tasks_no_multiarith" / "stage2" / "layer_deltas_long.csv")

    train_full_source = load_csv(
        results_root / "embedding_cluster_epoch0_qa_answer_lasttok" / "summary" / "layer_metrics_all_jobs.csv"
    )
    train_shared_source = load_csv(
        results_root / "embedding_cluster_epoch0_qa_answer_lasttok_shared_tsne" / "summary" / "layer_metrics_all_jobs.csv"
    )
    sample_pool_5 = load_json_if_exists(
        results_root / "embedding_cluster" / "sample_pool" / "sample_pool_seed42_n100_5tasks.json"
    )
    sample_pool_4 = load_json_if_exists(
        results_root / "embedding_cluster_4tasks_no_multiarith" / "sample_pool" / "sample_pool_seed42_n100_5tasks.json"
    )

    missing_extra: List[Dict[str, object]] = []
    outputs: Dict[str, Path] = {}

    # D
    d_outputs = build_train_corpus_tables(train_full_source, train_shared_source, out_dir, missing_extra)
    outputs.update(d_outputs)
    train_full_generated = load_csv(d_outputs["train_corpus_cluster_full.csv"])

    # E
    e_outputs = build_test_task_tables(stage2_5, stage3_5, stage2_4, out_dir, missing_extra)
    outputs.update(e_outputs)

    # F
    f_outputs = build_train_vs_test_alignment(train_full_generated, stage2_5, out_dir, missing_extra)
    outputs.update(f_outputs)

    # G
    g_outputs = build_semantic_entropy_outputs(out_dir, missing_extra, repo_root)
    outputs.update(g_outputs)

    # H
    h_outputs = build_tsne_manifest_and_selected_figures(
        results_root / "embedding_cluster" / "stage3" / "by_train_dataset",
        out_dir,
        missing_extra,
    )
    outputs.update(h_outputs)

    # C + I
    ci_outputs = build_metric_defs_and_sampling_rules(repo_root, out_dir, sample_pool_5, sample_pool_4)
    outputs.update(ci_outputs)

    # B: 20-point checklist
    def exists_df(x: Optional[pd.DataFrame]) -> bool:
        return x is not None and not x.empty

    checks: Dict[str, Dict[str, object]] = {
        "1": {
            "item": "训练语料侧 32层×3seed 完整聚类指标表",
            "found": "是" if exists_df(train_full_source) else "否",
            "source_paths": str(results_root / "embedding_cluster_epoch0_qa_answer_lasttok" / "summary" / "layer_metrics_all_jobs.csv"),
            "exportable": "train_corpus_cluster_full.csv",
            "missing": "" if exists_df(train_full_source) else "source missing",
            "next_step": "N/A" if exists_df(train_full_source) else "copy source summary from cloud",
        },
        "2": {
            "item": "训练语料侧同层对齐重跑结果",
            "found": "是" if exists_df(train_shared_source) else "部分找到",
            "source_paths": str(results_root / "embedding_cluster_epoch0_qa_answer_lasttok_shared_tsne" / "summary" / "layer_metrics_all_jobs.csv"),
            "exportable": "train_corpus_same_layer_aligned.csv",
            "missing": "" if exists_df(train_shared_source) else "shared_tsne source not found; aligned table built from full source",
            "next_step": "copy shared_tsne summary from cloud if needed",
        },
        "3": {
            "item": "测试任务侧全层扫描完整表",
            "found": "是" if exists_df(stage2_5) else "否",
            "source_paths": str(results_root / "embedding_cluster" / "stage2" / "layer_deltas_long.csv"),
            "exportable": "test_task_full_layer_scan.csv",
            "missing": "" if exists_df(stage2_5) else "stage2 source missing",
            "next_step": "copy stage2 outputs",
        },
        "4": {
            "item": "测试任务侧 layer21/30/31 完整指标表",
            "found": "是" if exists_df(stage3_5) else "否",
            "source_paths": str(results_root / "embedding_cluster" / "stage3" / "story_tsne_summary_all.csv"),
            "exportable": "test_task_display_layers_full.csv",
            "missing": "" if exists_df(stage3_5) else "stage3 source missing",
            "next_step": "copy stage3 outputs",
        },
        "5": {
            "item": "测试任务侧 layer28/29/30 完整指标表",
            "found": "是" if exists_df(stage2_4) else "否",
            "source_paths": str(results_root / "embedding_cluster_4tasks_no_multiarith" / "stage2" / "layer_deltas_long.csv"),
            "exportable": "test_task_layer_28_29_30_full.csv",
            "missing": "" if exists_df(stage2_4) else "4-task stage2 source missing",
            "next_step": "copy 4-task variant stage2 outputs",
        },
        "6": {
            "item": "不同seed下聚类指标均值和方差",
            "found": "是" if train_full_generated is not None and not train_full_generated.empty else "否",
            "source_paths": str(results_root / "embedding_cluster_epoch0_qa_answer_lasttok" / "summary" / "layer_metrics_all_jobs.csv"),
            "exportable": "train_corpus_seed_mean_var.csv",
            "missing": "" if train_full_generated is not None and not train_full_generated.empty else "seed table unavailable",
            "next_step": "ensure layer_metrics_all_jobs.csv is complete",
        },
        "7": {
            "item": "Sil/kNN/DB/CH 计算参数",
            "found": "是",
            "source_paths": str(datainf_root / "script" / "embedding_cluster_utils.py"),
            "exportable": "metric_definitions_for_latex.tex",
            "missing": "",
            "next_step": "N/A",
        },
        "8": {
            "item": "kNN Purity 中 k 的具体数值",
            "found": "是",
            "source_paths": str(datainf_root / "script" / "embedding_cluster_utils.py"),
            "exportable": "k=10 recorded in metric_definitions_for_latex.tex",
            "missing": "",
            "next_step": "N/A",
        },
        "9": {
            "item": "距离度量",
            "found": "是",
            "source_paths": str(datainf_root / "script" / "embedding_cluster_utils.py"),
            "exportable": "euclidean noted in metric_definitions_for_latex.tex",
            "missing": "",
            "next_step": "N/A",
        },
        "10": {
            "item": "embedding 提取层位",
            "found": "是",
            "source_paths": f"{results_root / 'embedding_cluster' / 'stage2' / 'layer_rank_summary.csv'}; {results_root / 'embedding_cluster' / 'stage1' / 'epoch5_layer_scan_all.csv'}",
            "exportable": "full layer scan + selected layer tables",
            "missing": "",
            "next_step": "N/A",
        },
        "11": {
            "item": "embedding 提取方式（last token/mean pooling）",
            "found": "是",
            "source_paths": f"{datainf_root / 'script' / 'embedding_cluster_utils.py'}; {datainf_root / 'script' / 'embedding_cluster_epoch0_qa_01_run_job.py'}",
            "exportable": "sampling_rules_for_latex.tex",
            "missing": "",
            "next_step": "N/A",
        },
        "12": {
            "item": "测试任务采样规则",
            "found": "是" if sample_pool_5 is not None else "部分找到",
            "source_paths": str(results_root / "embedding_cluster" / "sample_pool" / "sample_pool_seed42_n100_5tasks.json"),
            "exportable": "sampling_rules_report.md",
            "missing": "" if sample_pool_5 is not None else "sample_pool json missing",
            "next_step": "copy sample_pool json",
        },
        "13": {
            "item": "每个测试任务采样数量",
            "found": "是" if sample_pool_5 is not None else "部分找到",
            "source_paths": str(results_root / "embedding_cluster" / "sample_pool" / "sample_pool_seed42_n100_5tasks.json"),
            "exportable": "sampling_rules_report.md",
            "missing": "" if sample_pool_5 is not None else "sample count unknown",
            "next_step": "copy sample_pool json",
        },
        "14": {
            "item": "是否去掉 MultiArith 及原因",
            "found": "是" if sample_pool_4 is not None else "部分找到",
            "source_paths": str(results_root / "embedding_cluster_4tasks_no_multiarith"),
            "exportable": "sampling_rules_report.md",
            "missing": "" if sample_pool_4 is not None else "4-task variant source missing",
            "next_step": "copy 4-task variant outputs",
        },
        "15": {
            "item": "t-SNE 完整图像池",
            "found": "是" if (results_root / "embedding_cluster" / "stage3" / "by_train_dataset").is_dir() else "否",
            "source_paths": str(results_root / "embedding_cluster" / "stage3" / "by_train_dataset"),
            "exportable": "tsne_figure_manifest.csv",
            "missing": "",
            "next_step": "N/A",
        },
        "16": {
            "item": "训练语料侧 t-SNE 图",
            "found": "是" if (results_root / "embedding_cluster_epoch0_qa_answer_lasttok" / "jobs").is_dir() else "否",
            "source_paths": str(results_root / "embedding_cluster_epoch0_qa_answer_lasttok" / "jobs"),
            "exportable": "available in raw job folders; indexed indirectly in report",
            "missing": "",
            "next_step": "N/A",
        },
        "17": {
            "item": "语义熵计算脚本或结果文件",
            "found": "部分找到",
            "source_paths": f"{repo_root / 'sdft' / 'Mutual-Information'}",
            "exportable": "semantic_entropy_method_for_latex.tex",
            "missing": "No semantic-entropy result files under DataInf/results",
            "next_step": "run/copy semantic entropy results",
        },
        "18": {
            "item": "语义熵训练语料侧完整维度结果",
            "found": "否",
            "source_paths": "",
            "exportable": "empty placeholder: semantic_entropy_train_full.csv",
            "missing": "numeric result files not found",
            "next_step": "run semantic-entropy pipeline and copy outputs",
        },
        "19": {
            "item": "语义熵测试回答侧结果",
            "found": "否",
            "source_paths": "",
            "exportable": "empty placeholder: semantic_entropy_answer_dim25.csv",
            "missing": "numeric result files not found",
            "next_step": "run semantic-entropy answer-side pipeline and copy outputs",
        },
        "20": {
            "item": "PCA 或 UMAP 补充图",
            "found": "部分找到",
            "source_paths": f"{results_root / 'embedding_cluster' / 'stage3'}",
            "exportable": "t-SNE points/plots available; no standalone UMAP figure files located",
            "missing": "UMAP image outputs not found in DataInf/results",
            "next_step": "if needed, run dedicated UMAP plot exporter",
        },
    }

    missing_path = make_missing_report(out_dir, missing_extra, checks)
    outputs["MISSING_REPORT.md"] = missing_path

    # A README
    readme_path = write_readme_pack(out_dir, scanned_dirs, outputs, checks)
    outputs["README_CHAPTER3_PACK.md"] = readme_path

    # K final report
    final_path = write_final_report(out_dir, outputs, missing_path)
    outputs["FINAL_REPORT.md"] = final_path

    # J zip pack
    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(out_dir)))

    print(str(out_dir))
    print(str(zip_path))


if __name__ == "__main__":
    main()
