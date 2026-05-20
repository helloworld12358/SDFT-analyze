#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate thesis figure assets with terminology-only label fixes.

This script intentionally does NOT recompute experiment metrics.
For t-SNE figures, it re-plots from existing saved 2D coordinates CSV files.
For chapter-4 figures, it reuses fixed matrices/statistics from the existing plot script.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None

from chapter4_make_paper_figures import (
    build_gradient_statistics_figure,
    build_ownh_dolly_tc_heatmap,
    build_train_test_deltac_heatmap,
    ensure_output_dir,
    set_global_style,
)


TASK_COLOR_MAP = {
    "alpaca_eval": "#1f77b4",
    "gsm8k": "#ff7f0e",
    "humaneval": "#2ca02c",
    "multiarith": "#d62728",
    "openfunction": "#9467bd",
}

TASK_DISPLAY_NAME = {
    "alpaca_eval": "AlpacaEval",
    "gsm8k": "GSM8K",
    "humaneval": "HumanEval",
    "multiarith": "MultiArith",
    "openfunction": "OpenFunctions",
}


def _pick_endpoint_points_csv(ds_dir: Path, train_slug: str, method: str, layer: int) -> Path:
    pat = str(ds_dir / f"tsne_points_{train_slug}__{method}__epoch_*_layer{layer:02d}.csv")
    cands = sorted(glob.glob(pat))
    if not cands:
        raise FileNotFoundError(f"No points CSV found for {train_slug}/{method}/layer{layer:02d} under {ds_dir}")

    def _epoch_num(p: str) -> int:
        m = re.search(r"epoch_(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1

    cands = sorted(cands, key=lambda x: (_epoch_num(x), x))
    return Path(cands[-1])


def _pick_base_points_csv(ds_dir: Path, layer: int) -> Path:
    p = ds_dir / f"tsne_points_base__epoch_0_layer{layer:02d}.csv"
    if not p.is_file():
        raise FileNotFoundError(f"Missing base points CSV: {p}")
    return p


def _plot_from_points_csv(points_csv: Path, out_png: Path, title: str) -> None:
    df = pd.read_csv(points_csv)
    need = {"task", "tsne_x", "tsne_y"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing columns in {points_csv}: need {sorted(need)}")

    rows = df.to_dict(orient="records")
    tasks = list(TASK_DISPLAY_NAME.keys())

    plt.figure(figsize=(8, 6))
    for t in tasks:
        idx = [i for i, r in enumerate(rows) if str(r.get("task", "")) == t]
        if not idx:
            continue
        sub = df.iloc[idx]
        plt.scatter(
            sub["tsne_x"].to_numpy(),
            sub["tsne_y"].to_numpy(),
            s=16,
            alpha=0.75,
            c=TASK_COLOR_MAP.get(t, None),
            label=TASK_DISPLAY_NAME.get(t, t),
            edgecolors="none",
        )

    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def _build_chapter3_tsne(datainf_root: Path, output_fig_root: Path) -> List[str]:
    stage3_root = datainf_root / "results" / "embedding_cluster" / "stage3" / "by_train_dataset"
    out_dir = output_fig_root / "chapter3"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []

    specs = [
        ("alpaca", "Alpaca", "alpaca"),
        ("gsm8k", "GSM8K", "gsm8k"),
        ("openfunction", "OpenFunctions", "openfunction"),
    ]
    layer = 21

    for slug, display_name, file_prefix in specs:
        ds_dir = stage3_root / slug
        if not ds_dir.is_dir():
            raise FileNotFoundError(f"Missing stage3 directory: {ds_dir}")

        base_csv = _pick_base_points_csv(ds_dir, layer)
        sft_csv = _pick_endpoint_points_csv(ds_dir, slug, "sft", layer)
        sdft_csv = _pick_endpoint_points_csv(ds_dir, slug, "sdft", layer)

        targets = [
            (base_csv, out_dir / f"{file_prefix}_base_epoch0_layer21.png", f"{display_name}: Base, Layer 21"),
            (sft_csv, out_dir / f"{file_prefix}_sft_epoch5_layer21.png", f"{display_name}: SFT Endpoint, Layer 21"),
            (sdft_csv, out_dir / f"{file_prefix}_sdft_epoch5_layer21.png", f"{display_name}: SDFT Endpoint, Layer 21"),
        ]
        for src_csv, out_png, title in targets:
            _plot_from_points_csv(src_csv, out_png, title)
            generated.append(str(out_png.resolve()))

    return generated


def _build_appendix_tsne(datainf_root: Path, output_fig_root: Path) -> List[str]:
    stage3_root = datainf_root / "results" / "embedding_cluster" / "stage3" / "by_train_dataset"
    out_dir = output_fig_root / "appendix_tsne_selected"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []

    specs = [
        ("dolly", "Dolly"),
        ("lima", "LIMA"),
        ("magicoder", "Magicoder"),
        ("openhermes", "OpenHermes"),
    ]
    layer = 30

    for slug, display_name in specs:
        ds_dir = stage3_root / slug
        if not ds_dir.is_dir():
            raise FileNotFoundError(f"Missing stage3 directory: {ds_dir}")

        base_csv = _pick_base_points_csv(ds_dir, layer)
        sft_csv = _pick_endpoint_points_csv(ds_dir, slug, "sft", layer)
        sdft_csv = _pick_endpoint_points_csv(ds_dir, slug, "sdft", layer)

        targets = [
            (
                base_csv,
                out_dir / f"{display_name}__base__tsne_plot_base__epoch_0_layer30.png",
                f"{display_name}: Base, Layer 30",
            ),
            (
                sft_csv,
                out_dir / f"{display_name}__sft_final__tsne_plot_{slug}__sft__epoch_5_layer30.png",
                f"{display_name}: SFT Endpoint, Layer 30",
            ),
            (
                sdft_csv,
                out_dir / f"{display_name}__sdft_final__tsne_plot_{slug}__sdft__epoch_5_layer30.png",
                f"{display_name}: SDFT Endpoint, Layer 30",
            ),
        ]
        for src_csv, out_png, title in targets:
            _plot_from_points_csv(src_csv, out_png, title)
            generated.append(str(out_png.resolve()))

    return generated


def _pick_chinese_font(size: int) -> Optional["ImageFont.ImageFont"]:
    if ImageFont is None:
        return None
    cands = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for p in cands:
        try:
            if os.path.isfile(p):
                return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _fit_text_font(draw: "ImageDraw.ImageDraw", text: str, box_w: int, box_h: int) -> Optional["ImageFont.ImageFont"]:
    for sz in [78, 72, 66, 60, 56, 52, 48, 44, 40, 36, 32]:
        font = _pick_chinese_font(sz)
        if font is None:
            continue
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= int(box_w * 0.92) and th <= int(box_h * 0.78):
            return font
    return _pick_chinese_font(32)


def _fix_chapter2_pipeline_text(chapter2_src: Path, out_path: Path) -> Optional[str]:
    if Image is None or ImageDraw is None:
        return None
    if not chapter2_src.is_file():
        return None

    img = Image.open(chapter2_src).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # two target boxes (normalized coordinates), inner area only.
    top_box = (int(0.635 * w), int(0.345 * h), int(0.795 * w), int(0.438 * h))
    bot_box = (int(0.635 * w), int(0.715 * h), int(0.795 * w), int(0.808 * h))

    top_color = img.getpixel((int(0.72 * w), int(0.405 * h)))
    bot_color = img.getpixel((int(0.72 * w), int(0.775 * h)))

    radius = max(8, int(min(w, h) * 0.018))
    draw.rounded_rectangle(top_box, radius=radius, fill=top_color)
    draw.rounded_rectangle(bot_box, radius=radius, fill=bot_color)

    top_text = "SFT 微调模型"
    bot_text = "SDFT 微调模型"

    for box, text in [(top_box, top_text), (bot_box, bot_text)]:
        l, t, r, b = box
        box_w = r - l
        box_h = b - t
        font = _fit_text_font(draw, text, box_w, box_h)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = int((l + r - tw) / 2 - bbox[0])
        y = int((t + b - th) / 2 - bbox[1])
        draw.text((x, y), text, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return str(out_path.resolve())


def main() -> None:
    ap = argparse.ArgumentParser(description="Build thesis figures with terminology-only label fixes.")
    ap.add_argument("--datainf_root", type=str, default="")
    ap.add_argument("--output_root", type=str, default="")
    ap.add_argument(
        "--chapter2_pipeline_source",
        type=str,
        default="",
        help="Optional path to chapter2_pipeline.png. If provided, text labels are minimally replaced.",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    datainf_root = Path(args.datainf_root).resolve() if args.datainf_root else (repo_root / "DataInf")
    output_root = Path(args.output_root).resolve() if args.output_root else (datainf_root / "results" / "thesis_figure_text_fix")
    output_fig_root = output_root / "figures"
    output_fig_root.mkdir(parents=True, exist_ok=True)

    # Chapter-4 PDFs.
    set_global_style()
    chapter4_dir = output_fig_root
    ensure_output_dir(chapter4_dir)
    out_ch4 = [
        str(build_gradient_statistics_figure(chapter4_dir).resolve()),
        str(build_ownh_dolly_tc_heatmap(chapter4_dir).resolve()),
        str(build_train_test_deltac_heatmap(chapter4_dir).resolve()),
    ]

    # Chapter-3 and appendix t-SNE from existing saved coordinates only.
    out_ch3 = _build_chapter3_tsne(datainf_root, output_fig_root)
    out_app = _build_appendix_tsne(datainf_root, output_fig_root)

    out_ch2: Optional[str] = None
    if args.chapter2_pipeline_source:
        out_ch2 = _fix_chapter2_pipeline_text(
            Path(args.chapter2_pipeline_source).resolve(),
            output_fig_root / "chapter2_pipeline.png",
        )

    manifest = {
        "repo_root": str(repo_root),
        "datainf_root": str(datainf_root),
        "output_root": str(output_root),
        "chapter4_outputs": out_ch4,
        "chapter3_outputs": out_ch3,
        "appendix_tsne_outputs": out_app,
        "chapter2_pipeline_output": out_ch2,
        "notes": [
            "t-SNE figures are re-plotted from existing saved tsne_points_*.csv only.",
            "No t-SNE/UMAP recomputation is performed in this script.",
            "Chapter-4 figures reuse fixed matrices/statistics from chapter4_make_paper_figures.py.",
        ],
    }
    manifest_path = output_root / "thesis_figure_text_fix_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    for p in out_ch4 + out_ch3 + out_app:
        print(p)
    if out_ch2:
        print(out_ch2)
    print(str(manifest_path.resolve()))


if __name__ == "__main__":
    main()
