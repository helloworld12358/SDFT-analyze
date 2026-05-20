#!/usr/bin/env python
"""Build chapter-4 paper figures from fixed validated statistics/matrices.

Outputs:
  - figures/chapter4_gradient_statistics.pdf
  - figures/chapter4_ownH_dolly_TC_heatmap.pdf
  - figures/chapter4_train_test_deltaC_heatmap.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def set_global_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def build_gradient_statistics_figure(output_dir: Path) -> Path:
    metrics = [
        "Mean Gradient Norm",
        "Gradient-Norm Variance",
        "Total Gradient Variance",
    ]

    sft_init = np.array([4.898, 0.830, 7.294], dtype=float)
    sdft_init = np.array([5.328, 0.843, 7.793], dtype=float)

    sft_lora = np.array([13.394, 2.940, 19.565], dtype=float)
    sdft_lora = np.array([15.694, 3.030, 25.967], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    bar_w = 0.35
    x = np.arange(len(metrics))
    c_sft = "#4C78A8"
    c_sdft = "#F58518"

    # Panel A
    axes[0].bar(x - bar_w / 2, sft_init, width=bar_w, label="SFT", color=c_sft)
    axes[0].bar(x + bar_w / 2, sdft_init, width=bar_w, label="SDFT", color=c_sdft)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].set_ylabel("Value")
    axes[0].set_title("Panel A: Initial Model Parameter Space")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")
    axes[0].legend(loc="upper left")

    # Panel B
    axes[1].bar(x - bar_w / 2, sft_lora, width=bar_w, label="SFT", color=c_sft)
    axes[1].bar(x + bar_w / 2, sdft_lora, width=bar_w, label="SDFT", color=c_sdft)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].set_ylabel("Value")
    axes[1].set_title("Panel B: LoRA Parameter Space at Training End")
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")
    axes[1].legend(loc="upper left")

    out = output_dir / "chapter4_gradient_statistics.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _annotate_heatmap(ax, data: np.ndarray, fmt: str, text_color: str = "black") -> None:
    n_rows, n_cols = data.shape
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(
                j,
                i,
                format(data[i, j], fmt),
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )


def build_ownh_dolly_tc_heatmap(output_dir: Path) -> Path:
    labels = ["AlpacaEval", "GSM8K", "HumanEval", "MultiArith", "OpenFunctions"]

    t_sft = np.array(
        [
            [389.529, -3.237, 47.293, 1.361, 372.339],
            [-3.237, 247.821, 115.084, 333.758, 85.966],
            [47.293, 115.084, 260.962, 145.612, 125.453],
            [1.361, 333.758, 145.612, 929.127, 77.708],
            [372.339, 85.966, 125.453, 77.708, 2118.013],
        ],
        dtype=float,
    )
    t_sdft = np.array(
        [
            [325.932, 254.802, 242.559, 345.340, 294.077],
            [254.802, 463.369, 302.497, 625.070, 298.022],
            [242.559, 302.497, 397.928, 396.193, 274.538],
            [345.340, 625.070, 396.193, 1353.042, 390.053],
            [294.077, 298.022, 274.538, 390.053, 1172.230],
        ],
        dtype=float,
    )
    c_sft = np.array(
        [
            [1.000, -0.010, 0.148, 0.002, 0.410],
            [-0.010, 1.000, 0.453, 0.696, 0.119],
            [0.148, 0.453, 1.000, 0.296, 0.169],
            [0.002, 0.696, 0.296, 1.000, 0.055],
            [0.410, 0.119, 0.169, 0.055, 1.000],
        ],
        dtype=float,
    )
    c_sdft = np.array(
        [
            [1.000, 0.656, 0.674, 0.520, 0.476],
            [0.656, 1.000, 0.705, 0.789, 0.404],
            [0.674, 0.705, 1.000, 0.540, 0.402],
            [0.520, 0.789, 0.540, 1.000, 0.310],
            [0.476, 0.404, 0.402, 0.310, 1.000],
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11), constrained_layout=True)

    t_vmin = float(min(np.min(t_sft), np.min(t_sdft)))
    t_vmax = float(max(np.max(t_sft), np.max(t_sdft)))
    c_vmin, c_vmax = -0.1, 1.0

    im_t_sft = axes[0, 0].imshow(t_sft, cmap="YlOrRd", vmin=t_vmin, vmax=t_vmax, aspect="auto")
    im_t_sdft = axes[0, 1].imshow(t_sdft, cmap="YlOrRd", vmin=t_vmin, vmax=t_vmax, aspect="auto")
    im_c_sft = axes[1, 0].imshow(c_sft, cmap="YlGnBu", vmin=c_vmin, vmax=c_vmax, aspect="auto")
    im_c_sdft = axes[1, 1].imshow(c_sdft, cmap="YlGnBu", vmin=c_vmin, vmax=c_vmax, aspect="auto")

    axes[0, 0].set_title(r"SFT $I_H$ Matrix")
    axes[0, 1].set_title(r"SDFT $I_H$ Matrix")
    axes[1, 0].set_title(r"SFT $\rho_H$ Matrix")
    axes[1, 1].set_title(r"SDFT $\rho_H$ Matrix")

    for ax in axes.flat:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticklabels(labels)
        ax.tick_params(labelsize=9)

    _annotate_heatmap(axes[0, 0], t_sft, ".1f")
    _annotate_heatmap(axes[0, 1], t_sdft, ".1f")
    _annotate_heatmap(axes[1, 0], c_sft, ".2f")
    _annotate_heatmap(axes[1, 1], c_sdft, ".2f")

    cbar_t = fig.colorbar(im_t_sdft, ax=[axes[0, 0], axes[0, 1]], shrink=0.86, pad=0.02)
    cbar_t.set_label(r"$I_H$ Value", rotation=90)
    cbar_c = fig.colorbar(im_c_sdft, ax=[axes[1, 0], axes[1, 1]], shrink=0.86, pad=0.02)
    cbar_c.set_label(r"$\rho_H$ Value", rotation=90)

    out = output_dir / "chapter4_ownH_dolly_TC_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def build_train_test_deltac_heatmap(output_dir: Path) -> Path:
    rows = ["GSM8K", "OpenFunctions", "Magicoder", "Alpaca", "Dolly", "LIMA", "OpenHermes"]
    cols = ["AlpacaEval", "GSM8K", "HumanEval", "MultiArith", "OpenFunctions"]
    delta_c = np.array(
        [
            [0.0242, -0.0641, -0.0051, -0.1771, 0.0391],
            [0.0561, 0.0628, 0.0297, 0.0526, 0.0869],
            [0.0429, 0.0437, 0.0496, 0.0316, 0.0328],
            [0.1825, 0.0955, 0.1079, 0.0681, 0.1363],
            [0.3055, 0.2507, 0.2474, 0.1773, 0.2316],
            [0.0597, 0.0559, -0.0028, 0.0328, 0.1256],
            [0.0126, -0.0165, -0.0103, 0.0009, 0.0825],
        ],
        dtype=float,
    )

    vmax = float(np.max(np.abs(delta_c)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10.8, 6.6), constrained_layout=True)
    im = ax.imshow(delta_c, cmap="coolwarm", norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticklabels(rows)
    ax.set_title(r"Train-Test $\Delta \rho_H$ Heatmap (SDFT - SFT, Endpoint)")

    for i in range(delta_c.shape[0]):
        for j in range(delta_c.shape[1]):
            ax.text(j, i, f"{delta_c[i, j]:.3f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.02)
    cbar.set_label(r"$\Delta \rho_H$", rotation=90)

    out = output_dir / "chapter4_train_test_deltaC_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chapter-4 paper figures (PDF).")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str((Path(__file__).resolve().parents[2] / "figures")),
        help="Directory to write output PDF figures.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    ensure_output_dir(output_dir)
    set_global_style()

    outputs = [
        build_gradient_statistics_figure(output_dir),
        build_ownh_dolly_tc_heatmap(output_dir),
        build_train_test_deltac_heatmap(output_dir),
    ]

    for p in outputs:
        print(str(p))


if __name__ == "__main__":
    main()
