#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from loss_theory_utils import detect_datainf_root, ensure_dir, resolve_loss_theory_root


def load_csv_if_exists(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def main() -> None:
    p = argparse.ArgumentParser(description="Build final theory report from F2-F8 outputs.")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default="")
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    root = resolve_loss_theory_root(datainf_root, args.output_root)
    out_dir = ensure_dir(os.path.join(root, "analysis", "final_report"))

    tail_df = load_csv_if_exists(os.path.join(root, "analysis", "tail_shape", "tail_shape_summary.csv"))
    mgf_df = load_csv_if_exists(os.path.join(root, "analysis", "mgf_check", "mgf_fit_summary.csv"))
    bern_df = load_csv_if_exists(os.path.join(root, "analysis", "emp_bernstein", "emp_bernstein_coverage.csv"))
    rob_df = load_csv_if_exists(os.path.join(root, "analysis", "robust_mean", "robust_mean_summary.csv"))
    cond_df = load_csv_if_exists(os.path.join(root, "analysis", "conditional", "conditional_overview.json"))
    len_df = load_csv_if_exists(os.path.join(root, "analysis", "len_ablation", "len_ablation_summary.csv"))
    dep_df = load_csv_if_exists(os.path.join(root, "analysis", "dependence", "sample_block_bootstrap_compare.csv"))

    # load conditional overview json if csv load is empty
    cond_overview: Dict[str, float] = {}
    cond_json = os.path.join(root, "analysis", "conditional", "conditional_overview.json")
    if os.path.isfile(cond_json):
        try:
            cond_overview = json.load(open(cond_json, "r", encoding="utf-8"))
        except Exception:
            cond_overview = {}

    evidence: List[Dict[str, object]] = []
    labels: List[str] = []

    # Rule A: sub-gamma + Bernstein plausible
    if not tail_df.empty and not mgf_df.empty and not bern_df.empty:
        sample_tail = tail_df[tail_df["name"] == "sample_Lbar_i"]
        if not sample_tail.empty:
            tail_label = str(sample_tail.iloc[0].get("empirical_label", ""))
        else:
            tail_label = ""
        mean_cov = float(pd.to_numeric(bern_df["coverage_rate"], errors="coerce").dropna().mean()) if "coverage_rate" in bern_df.columns else float("nan")
        mgf_best = float(pd.to_numeric(mgf_df.get("subgamma_mse", np.nan), errors="coerce").dropna().min()) if "subgamma_mse" in mgf_df.columns else float("nan")
        if (tail_label in ("sub_exponential_or_sub_gamma_like", "sub_gaussian_like")) and np.isfinite(mean_cov) and mean_cov >= 0.75:
            labels.append("length-normalized sample loss is approximately sub-gamma; empirical Bernstein-style bounds are reasonable")
            evidence.append(
                {
                    "hypothesis": "sub-gamma/Bernstein",
                    "tail_label": tail_label,
                    "mean_empirical_coverage": mean_cov,
                    "best_subgamma_mse": mgf_best,
                }
            )

    # Rule B: heavy but finite variance -> robust means preferred
    if not rob_df.empty:
        tmp = rob_df.copy()
        tmp["p95_abs_dev"] = pd.to_numeric(tmp["p95_abs_dev"], errors="coerce")
        try:
            p95_mean = float(tmp[tmp["estimator"] == "mean"]["p95_abs_dev"].iloc[0])
            p95_catoni = float(tmp[tmp["estimator"] == "catoni"]["p95_abs_dev"].iloc[0])
            p95_mom = float(tmp[tmp["estimator"] == "median_of_means"]["p95_abs_dev"].iloc[0])
            if np.isfinite(p95_mean) and np.isfinite(p95_catoni) and p95_mean > 1.15 * min(p95_catoni, p95_mom):
                labels.append("overall distribution appears heavy with finite-variance signals; robust mean bounds are more reliable than plain empirical mean bounds")
                evidence.append(
                    {
                        "hypothesis": "heavy_finite_variance_robust_mean",
                        "p95_mean": p95_mean,
                        "p95_catoni": p95_catoni,
                        "p95_mom": p95_mom,
                    }
                )
        except Exception:
            pass

    # Rule C: conditional improvements
    if cond_overview:
        pooled_var = float(cond_overview.get("pooled_var", np.nan))
        by_domain_ratio = float(cond_overview.get("domain_variance_ratio", np.nan))
        by_len_ratio = float(cond_overview.get("length_variance_ratio", np.nan))
        if np.isfinite(by_domain_ratio) and np.isfinite(by_len_ratio) and (by_domain_ratio < 0.9 or by_len_ratio < 0.9):
            labels.append("unconditional distribution is heavier, but conditional on length/domain it becomes better behaved; use conditional Bernstein/sub-gamma interpretations")
            evidence.append(
                {
                    "hypothesis": "conditional_bernstein_subgamma",
                    "pooled_var": pooled_var,
                    "domain_variance_ratio": by_domain_ratio,
                    "length_variance_ratio": by_len_ratio,
                }
            )

    # Rule D: dependence
    if not dep_df.empty and "ordered_vs_naive_ratio" in dep_df.columns:
        ratios = pd.to_numeric(dep_df["ordered_vs_naive_ratio"], errors="coerce").dropna()
        if len(ratios) > 0 and float(np.mean(ratios)) >= 1.2:
            labels.append("token/sample dependence is non-negligible; use effective sample size or block/martingale interpretations")
            evidence.append(
                {
                    "hypothesis": "dependence_present",
                    "mean_ordered_vs_naive_ratio": float(np.mean(ratios)),
                    "p90_ordered_vs_naive_ratio": float(np.quantile(ratios, 0.90)),
                }
            )

    if not labels:
        labels.append("current evidence is mixed; no single hypothesis dominates yet")

    len_note = ""
    if not len_df.empty:
        try:
            tmp = len_df.set_index("target")
            cov_raw = float(tmp.loc["Li_raw_total", "emp_bernstein_coverage_delta_0p05"])
            cov_norm = float(tmp.loc["Lbar_i_normalized", "emp_bernstein_coverage_delta_0p05"])
            len_note = (
                f"Length ablation: coverage(raw Li)={cov_raw:.4f}, "
                f"coverage(normalized Lbar_i)={cov_norm:.4f}."
            )
        except Exception:
            len_note = "Length ablation exists but key fields were not fully readable."

    payload = {
        "output_root": os.path.abspath(root),
        "supported_hypotheses": labels,
        "evidence": evidence,
        "length_ablation_note": len_note,
        "input_artifacts": {
            "tail_shape": os.path.abspath(os.path.join(root, "analysis", "tail_shape", "tail_shape_summary.csv")),
            "mgf_check": os.path.abspath(os.path.join(root, "analysis", "mgf_check", "mgf_fit_summary.csv")),
            "emp_bernstein": os.path.abspath(os.path.join(root, "analysis", "emp_bernstein", "emp_bernstein_coverage.csv")),
            "robust_mean": os.path.abspath(os.path.join(root, "analysis", "robust_mean", "robust_mean_summary.csv")),
            "conditional": os.path.abspath(os.path.join(root, "analysis", "conditional", "conditional_overview.json")),
            "len_ablation": os.path.abspath(os.path.join(root, "analysis", "len_ablation", "len_ablation_summary.csv")),
            "dependence": os.path.abspath(os.path.join(root, "analysis", "dependence", "sample_block_bootstrap_compare.csv")),
        },
    }

    json_path = os.path.join(out_dir, "loss_theory_final_report.json")
    md_path = os.path.join(out_dir, "loss_theory_final_report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines: List[str] = []
    lines.append("# Loss Theory Final Report")
    lines.append("")
    lines.append("## Supported Hypotheses")
    lines.append("")
    for i, s in enumerate(labels, start=1):
        lines.append(f"{i}. {s}")
    lines.append("")
    lines.append("## Evidence")
    lines.append("")
    if evidence:
        for i, e in enumerate(evidence, start=1):
            lines.append(f"- E{i}: `{e}`")
    else:
        lines.append("- No strong single-cluster evidence yet.")
    lines.append("")
    lines.append("## Length Ablation")
    lines.append("")
    lines.append(f"- {len_note or 'No length-ablation note available.'}")
    lines.append("")
    lines.append("## Artifact Paths")
    lines.append("")
    for k, v in payload["input_artifacts"].items():
        lines.append(f"- `{k}`: `{v}`")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(os.path.abspath(json_path))
    print(os.path.abspath(md_path))


if __name__ == "__main__":
    main()

