#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect appendix materials from EXISTING results only.
No rerun, no LaTeX/body modification, no file rename of original artifacts.

Outputs (default): <repo>/DataInf/results/appendix_materials/
  - appendix_perf_delta.csv
  - appendix_perf_delta.md
  - appendix_tsne_paths.md
  - appendix_shared_geometry_perf.csv
  - appendix_shared_geometry_perf.md
  - appendix_missing_report.md
  - appendix_collect_manifest.json
  - appendix_tsne_selected/ (optional copied images)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


TRAIN_ORDER = ["GSM8K", "OpenFunctions", "Magicoder", "Alpaca", "Dolly", "LIMA", "OpenHermes"]
TASK_ORDER = ["GSM8K", "MultiArith", "OpenFunctions", "HumanEval", "AdvBench Raw", "AdvBench Jailbreak"]
TSNE_APPENDIX_DOMAINS = ["Dolly", "LIMA", "Magicoder", "OpenHermes"]

TRAIN_ALIASES = {
    "gsm8k": "GSM8K",
    "openfunction": "OpenFunctions",
    "openfunctions": "OpenFunctions",
    "magicoder": "Magicoder",
    "alpaca": "Alpaca",
    "dolly": "Dolly",
    "lima": "LIMA",
    "openhermes": "OpenHermes",
}


def normalize_key(x: str) -> str:
    s = str(x).strip().lower()
    s = s.replace("-", "_").replace(" ", "_").replace("/", "_")
    s = re.sub(r"_+", "_", s)
    return s


def canon_train(x: str) -> Optional[str]:
    return TRAIN_ALIASES.get(normalize_key(x))


def parse_epoch_num(x: object) -> Optional[int]:
    s = str(x).strip().lower()
    m = re.match(r"^epoch_(\d+)$", s)
    if m:
        return int(m.group(1))
    if s.isdigit():
        return int(s)
    return None


def pick_final_rows(df: pd.DataFrame, group_cols: List[str], prefer_hmode: bool = False) -> pd.DataFrame:
    df = df.copy()
    df["_epoch_num"] = df["epoch"].apply(parse_epoch_num) if "epoch" in df.columns else None
    if "_epoch_num" in df.columns:
        df = df[df["_epoch_num"].fillna(-1) > 1]

    if df.empty:
        return df

    if prefer_hmode and "h_mode" in df.columns:
        # Priority: own > mixed > cross_oracle_sft > cross_oracle_sdft > others
        rank_map = {
            "own": 0,
            "mixed": 1,
            "cross_oracle_sft": 2,
            "cross_oracle_sdft": 3,
        }
        df["_h_rank"] = df["h_mode"].astype(str).str.lower().map(rank_map).fillna(9)
        df = df.sort_values(group_cols + ["_h_rank", "_epoch_num"], ascending=[True] * len(group_cols) + [True, False])
    else:
        df = df.sort_values(group_cols + ["_epoch_num"], ascending=[True] * len(group_cols) + [False])

    keep_idx = []
    for _, g in df.groupby(group_cols, dropna=False):
        if g.empty:
            continue
        if prefer_hmode and "_h_rank" in g.columns:
            min_h = g["_h_rank"].min()
            g = g[g["_h_rank"] == min_h]
        max_ep = g["_epoch_num"].max()
        g = g[g["_epoch_num"] == max_ep]
        keep_idx.extend(list(g.index))

    out = df.loc[sorted(set(keep_idx))].copy()
    return out


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_markdown_table(df: pd.DataFrame, out_md: Path, title: str, ndigits: int = 6) -> None:
    ensure_parent(out_md)
    lines = [f"# {title}", ""]
    headers = list(df.columns)
    lines.append("| " + " | ".join(headers) + " |")
    align = ["---"] + ["---:" for _ in headers[1:]]
    lines.append("| " + " | ".join(align) + " |")
    for _, row in df.iterrows():
        cells: List[str] = []
        for i, c in enumerate(headers):
            v = row[c]
            if pd.isna(v):
                cells.append("")
            elif i == 0:
                cells.append(str(v))
            else:
                cells.append(f"{float(v):.{ndigits}f}")
        lines.append("| " + " | ".join(cells) + " |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_perf_delta(repo_root: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    missing: List[str] = []
    used: List[str] = []

    src = repo_root / "DataInf" / "results" / "schemeA" / "final_summary" / "schemeA_per_method_summary.csv"
    if not src.is_file():
        missing.append(f"missing source: {src}")
        out = pd.DataFrame({"train_domain": TRAIN_ORDER})
        for t in TASK_ORDER:
            out[t] = pd.NA
        return out, used, missing

    used.append(str(src))
    df = pd.read_csv(src)
    needed_cols = ["train_dataset", "method", "epoch"]
    for c in needed_cols:
        if c not in df.columns:
            missing.append(f"missing column in per_method_summary: {c}")

    # map columns to requested tasks
    task_col_candidates = {
        "GSM8K": ["metric_gsm8k"],
        "MultiArith": ["metric_multiarith"],
        "OpenFunctions": ["metric_openfunction"],
        "HumanEval": ["metric_humaneval"],
        "AdvBench Raw": ["metric_advbench_raw", "metric_safety_raw"],
        "AdvBench Jailbreak": ["metric_advbench_jailbreak", "metric_safety_jailbreak"],
    }

    for task, cands in task_col_candidates.items():
        if not any(c in df.columns for c in cands):
            missing.append(f"missing metric column for task={task}; candidates={cands}")

    if any(c not in df.columns for c in needed_cols):
        out = pd.DataFrame({"train_domain": TRAIN_ORDER})
        for t in TASK_ORDER:
            out[t] = pd.NA
        return out, used, missing

    df = df.copy()
    df["_train"] = df["train_dataset"].astype(str).map(lambda x: canon_train(x) or "")
    df["_method"] = df["method"].astype(str).str.lower()
    df = df[df["_train"].isin(TRAIN_ORDER) & df["_method"].isin(["sft", "sdft"])]

    if df.empty:
        missing.append("no valid rows after train/method filter")
        out = pd.DataFrame({"train_domain": TRAIN_ORDER})
        for t in TASK_ORDER:
            out[t] = pd.NA
        return out, used, missing

    final_df = pick_final_rows(df, group_cols=["_train", "_method"], prefer_hmode=False)
    if final_df.empty:
        missing.append("no final rows (epoch>1) found")

    log_cache: Dict[str, Dict[str, float]] = {}

    def parse_advbench_from_log(log_path: str) -> Dict[str, float]:
        """
        Best-effort parser for AdvBench raw/jailbreak metrics from existing perf logs.
        Returns keys in {"AdvBench Raw", "AdvBench Jailbreak"} when found.
        """
        if not log_path:
            return {}
        p = Path(log_path)
        if not p.is_file():
            return {}
        if log_path in log_cache:
            return log_cache[log_path]

        out: Dict[str, float] = {}
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = p.read_text(encoding="latin-1", errors="ignore")
        lines = text.splitlines()
        # Scan from tail to prioritize final report values.
        for line in reversed(lines):
            ll = line.lower()
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", line)
            if not nums:
                continue
            val = float(nums[-1])
            if (
                "advbench" in ll
                and "raw" in ll
            ) or (("safety" in ll or "adv" in ll) and "raw" in ll):
                out.setdefault("AdvBench Raw", val)
            if (
                "advbench" in ll
                and ("jailbreak" in ll or "jail" in ll)
            ) or (("safety" in ll or "adv" in ll) and ("jailbreak" in ll or "jail" in ll)):
                out.setdefault("AdvBench Jailbreak", val)
            if "AdvBench Raw" in out and "AdvBench Jailbreak" in out:
                break

        log_cache[log_path] = out
        return out

    # aggregate in case multiple h_mode rows remain
    agg_rows = []
    for (tr, md), g in final_df.groupby(["_train", "_method"], dropna=False):
        row = {"train_domain": tr, "method": md}
        for task, cands in task_col_candidates.items():
            vals = []
            for c in cands:
                if c in g.columns:
                    ser = pd.to_numeric(g[c], errors="coerce").dropna()
                    if not ser.empty:
                        vals.extend(list(ser.values))
            row[task] = float(pd.Series(vals).mean()) if vals else pd.NA

        # Fallback for AdvBench Raw/Jailbreak from perf logs.
        if pd.isna(row.get("AdvBench Raw")) or pd.isna(row.get("AdvBench Jailbreak")):
            log_vals_raw: List[float] = []
            log_vals_jb: List[float] = []
            if "perf_log_path" in g.columns:
                for lp in g["perf_log_path"].dropna().astype(str).tolist():
                    parsed = parse_advbench_from_log(lp)
                    if "AdvBench Raw" in parsed:
                        log_vals_raw.append(parsed["AdvBench Raw"])
                    if "AdvBench Jailbreak" in parsed:
                        log_vals_jb.append(parsed["AdvBench Jailbreak"])
            if pd.isna(row.get("AdvBench Raw")) and log_vals_raw:
                row["AdvBench Raw"] = float(pd.Series(log_vals_raw).mean())
            if pd.isna(row.get("AdvBench Jailbreak")) and log_vals_jb:
                row["AdvBench Jailbreak"] = float(pd.Series(log_vals_jb).mean())

        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)

    # delta = sdft - sft
    out_rows = []
    for tr in TRAIN_ORDER:
        r = {"train_domain": tr}
        g = agg[agg["train_domain"] == tr]
        gsft = g[g["method"] == "sft"]
        gsdft = g[g["method"] == "sdft"]
        for task in TASK_ORDER:
            vsft = pd.to_numeric(gsft[task], errors="coerce").dropna() if not gsft.empty else pd.Series(dtype=float)
            vsd = pd.to_numeric(gsdft[task], errors="coerce").dropna() if not gsdft.empty else pd.Series(dtype=float)
            if vsft.empty or vsd.empty:
                r[task] = pd.NA
                missing.append(f"missing delta value train={tr} task={task}")
            else:
                r[task] = float(vsd.mean() - vsft.mean())
        out_rows.append(r)

    out = pd.DataFrame(out_rows)
    return out, used, sorted(set(missing))


def _extract_layer(name: str) -> Optional[int]:
    m = re.search(r"layer(\d+)", name)
    return int(m.group(1)) if m else None


def _extract_epoch(name: str) -> Optional[int]:
    m = re.search(r"epoch_(\d+)", name)
    return int(m.group(1)) if m else None


def _pick_by_preferred_layers(files: List[Path], preferred_layers: List[int]) -> Optional[Path]:
    if not files:
        return None
    by_layer: Dict[int, List[Path]] = {}
    for p in files:
        l = _extract_layer(p.name)
        if l is None:
            continue
        by_layer.setdefault(l, []).append(p)
    for l in preferred_layers:
        if l in by_layer and by_layer[l]:
            return sorted(by_layer[l])[0]
    return sorted(files)[0]


def build_tsne_paths(repo_root: Path, output_dir: Path, copy_selected: bool) -> Tuple[Path, List[str], List[str], List[str]]:
    missing: List[str] = []
    notes: List[str] = []
    used: List[str] = []

    base_root = repo_root / "DataInf" / "results" / "embedding_cluster" / "stage3" / "by_train_dataset"
    if not base_root.is_dir():
        missing.append(f"missing tsne root: {base_root}")

    rec_layers_file = repo_root / "DataInf" / "results" / "embedding_cluster" / "stage2" / "recommended_layers.json"
    preferred_layers = [30, 31, 21]
    if rec_layers_file.is_file():
        used.append(str(rec_layers_file))
        try:
            j = json.loads(rec_layers_file.read_text(encoding="utf-8"))
            v = j.get("recommended_layers")
            if isinstance(v, list):
                tmp = []
                for x in v:
                    try:
                        tmp.append(int(x))
                    except Exception:
                        pass
                if tmp:
                    preferred_layers = tmp
        except Exception:
            notes.append(f"failed to parse {rec_layers_file}, fallback preferred_layers={preferred_layers}")

    lines: List[str] = []
    lines.append("# appendix_tsne_paths")
    lines.append("")
    lines.append("This file lists appendix t-SNE images for non-main-text train domains.")
    lines.append(f"Preferred layers priority: {preferred_layers}")
    lines.append("")

    selected_dir = output_dir / "appendix_tsne_selected"
    if copy_selected:
        selected_dir.mkdir(parents=True, exist_ok=True)

    for domain in TSNE_APPENDIX_DOMAINS:
        dl = domain.lower()
        ddir = base_root / dl
        if not ddir.is_dir():
            missing.append(f"missing domain dir: {ddir}")
            lines.append(f"## {domain}")
            lines.append("- base: MISSING")
            lines.append("- SFT final: MISSING")
            lines.append("- SDFT final: MISSING")
            lines.append("")
            continue

        used.append(str(ddir))

        base_files = sorted(ddir.glob("tsne_plot_base__epoch_0_layer*.png"))

        sft_all = sorted(ddir.glob(f"tsne_plot_{dl}__sft__epoch_*_layer*.png"))
        sdft_all = sorted(ddir.glob(f"tsne_plot_{dl}__sdft__epoch_*_layer*.png"))

        def keep_max_epoch(files: List[Path]) -> List[Path]:
            if not files:
                return []
            eps = [e for e in (_extract_epoch(p.name) for p in files) if e is not None]
            if not eps:
                return files
            me = max(eps)
            return [p for p in files if _extract_epoch(p.name) == me]

        sft_files = keep_max_epoch(sft_all)
        sdft_files = keep_max_epoch(sdft_all)

        # Try same-layer trio first
        p_base = p_sft = p_sdft = None
        chosen_layer = None
        for l in preferred_layers:
            b = [p for p in base_files if _extract_layer(p.name) == l]
            s = [p for p in sft_files if _extract_layer(p.name) == l]
            d = [p for p in sdft_files if _extract_layer(p.name) == l]
            if b and s and d:
                p_base, p_sft, p_sdft = sorted(b)[0], sorted(s)[0], sorted(d)[0]
                chosen_layer = l
                break

        if chosen_layer is None:
            p_base = _pick_by_preferred_layers(base_files, preferred_layers)
            p_sft = _pick_by_preferred_layers(sft_files, preferred_layers)
            p_sdft = _pick_by_preferred_layers(sdft_files, preferred_layers)
            lb = _extract_layer(p_base.name) if p_base else None
            ls = _extract_layer(p_sft.name) if p_sft else None
            ld = _extract_layer(p_sdft.name) if p_sdft else None
            notes.append(f"{domain}: no common layer for base/sft/sdft; picked base={lb}, sft={ls}, sdft={ld}")

        if p_base is None:
            missing.append(f"{domain}: missing base tsne")
        if p_sft is None:
            missing.append(f"{domain}: missing sft final tsne")
        if p_sdft is None:
            missing.append(f"{domain}: missing sdft final tsne")

        lines.append(f"## {domain}")
        lines.append(f"- base: {str(p_base) if p_base else 'MISSING'}")
        lines.append(f"- SFT final: {str(p_sft) if p_sft else 'MISSING'}")
        lines.append(f"- SDFT final: {str(p_sdft) if p_sdft else 'MISSING'}")
        lines.append(f"- chosen_layer: {chosen_layer if chosen_layer is not None else 'mixed'}")
        lines.append(f"- candidates: base={len(base_files)}, sft_final={len(sft_files)}, sdft_final={len(sdft_files)}")
        lines.append("")

        if copy_selected:
            for state, p in [("base", p_base), ("sft_final", p_sft), ("sdft_final", p_sdft)]:
                if p and p.is_file():
                    dst = selected_dir / f"{domain}__{state}__{p.name}"
                    shutil.copy2(p, dst)

    lines.append("## Excluded main-text domains")
    lines.append("- Alpaca")
    lines.append("- GSM8K")
    lines.append("- OpenFunctions")
    lines.append("")

    out_md = output_dir / "appendix_tsne_paths.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md, sorted(set(used)), sorted(set(missing)), sorted(set(notes))


def build_shared_geometry(repo_root: Path) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    missing: List[str] = []
    used: List[str] = []
    notes: List[str] = []

    src = repo_root / "DataInf" / "results" / "schemeA" / "final_summary" / "schemeA_sft_minus_sdft_summary.csv"
    if not src.is_file():
        missing.append(f"missing source: {src}")
        out = pd.DataFrame(
            {
                "train_domain": ["Alpaca", "Dolly", "GSM8K", "LIMA", "Magicoder", "OpenFunctions", "OpenHermes"],
                "Delta_lambda1_C": [pd.NA] * 7,
                "Delta_lambda_gap_C": [pd.NA] * 7,
                "Delta_C_off": [pd.NA] * 7,
                "Delta_C_off_Frobenius": [pd.NA] * 7,
                "mean_performance_gain": [pd.NA] * 7,
            }
        )
        return out, used, missing, notes

    used.append(str(src))
    df = pd.read_csv(src)
    for c in ["train_dataset", "epoch", "h_mode", "delta_lambda1_C", "delta_lambda1_minus_lambda2_C", "delta_mean_offdiag_C", "delta_fro_offdiag_C"]:
        if c not in df.columns:
            missing.append(f"missing column: {c}")

    if any(c not in df.columns for c in ["train_dataset", "epoch", "h_mode"]):
        out = pd.DataFrame(
            {
                "train_domain": ["Alpaca", "Dolly", "GSM8K", "LIMA", "Magicoder", "OpenFunctions", "OpenHermes"],
                "Delta_lambda1_C": [pd.NA] * 7,
                "Delta_lambda_gap_C": [pd.NA] * 7,
                "Delta_C_off": [pd.NA] * 7,
                "Delta_C_off_Frobenius": [pd.NA] * 7,
                "mean_performance_gain": [pd.NA] * 7,
            }
        )
        return out, used, missing, notes

    df = df.copy()
    df["_train"] = df["train_dataset"].astype(str).map(lambda x: canon_train(x) or "")
    df = df[df["_train"].isin(TRAIN_ORDER)]
    if df.empty:
        missing.append("no valid train rows in shared-geometry source")

    # Normalize delta direction to sdft_minus_sft.
    sign = 1.0
    if "delta_method" in df.columns:
        dm = [str(x).strip().lower() for x in df["delta_method"].dropna().tolist()]
        if dm and all("sft_minus_sdft" in x for x in dm):
            sign = -1.0
            notes.append("delta_method is sft_minus_sdft; all delta columns are sign-flipped to sdft_minus_sft.")

    final_df = pick_final_rows(df, group_cols=["_train"], prefer_hmode=True)
    if final_df.empty:
        missing.append("no final rows (epoch>1) found for shared geometry")

    # candidate performance gain columns
    perf_cols = [c for c in final_df.columns if c.startswith("delta_metric_")]
    if not perf_cols:
        missing.append("no delta_metric_* columns; cannot compute mean_performance_gain")
    else:
        notes.append("mean_performance_gain uses mean of columns: " + ", ".join(perf_cols))

    rows = []
    order = ["Alpaca", "Dolly", "GSM8K", "LIMA", "Magicoder", "OpenFunctions", "OpenHermes"]
    for tr in order:
        g = final_df[final_df["_train"] == tr]
        if g.empty:
            missing.append(f"missing shared-geometry row for {tr}")
            rows.append(
                {
                    "train_domain": tr,
                    "Delta_lambda1_C": pd.NA,
                    "Delta_lambda_gap_C": pd.NA,
                    "Delta_C_off": pd.NA,
                    "Delta_C_off_Frobenius": pd.NA,
                    "mean_performance_gain": pd.NA,
                }
            )
            continue

        # if still multiple rows, average
        def mean_col(name: str):
            if name not in g.columns:
                return pd.NA
            ser = pd.to_numeric(g[name], errors="coerce").dropna()
            return float(sign * ser.mean()) if not ser.empty else pd.NA

        mpg = pd.NA
        if perf_cols:
            vals = []
            for c in perf_cols:
                ser = pd.to_numeric(g[c], errors="coerce").dropna()
                if not ser.empty:
                    vals.extend([sign * float(v) for v in ser.values])
            if vals:
                mpg = float(pd.Series(vals).mean())
            else:
                missing.append(f"no numeric delta_metric_* for {tr}")

        rows.append(
            {
                "train_domain": tr,
                "Delta_lambda1_C": mean_col("delta_lambda1_C"),
                "Delta_lambda_gap_C": mean_col("delta_lambda1_minus_lambda2_C"),
                "Delta_C_off": mean_col("delta_mean_offdiag_C"),
                "Delta_C_off_Frobenius": mean_col("delta_fro_offdiag_C"),
                "mean_performance_gain": mpg,
            }
        )

    out = pd.DataFrame(rows)
    return out, used, sorted(set(missing)), notes


def write_missing_report(path: Path, sections: Dict[str, List[str]]) -> None:
    lines = ["# appendix_missing_report", ""]
    for sec, items in sections.items():
        lines.append(f"## {sec}")
        if items:
            for x in sorted(set(items)):
                lines.append(f"- {x}")
        else:
            lines.append("- None")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect appendix materials from existing files only.")
    ap.add_argument("--repo_root", type=str, default="", help="default: auto from script path")
    ap.add_argument("--output_dir", type=str, default="", help="default: <repo>/DataInf/results/appendix_materials")
    ap.add_argument("--copy_tsne_selected", type=int, default=1, help="1: copy selected images to appendix_tsne_selected")
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else script_path.parent.parent.parent
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (repo_root / "DataInf" / "results" / "appendix_materials")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Task 1
    perf_df, perf_used, perf_missing = build_perf_delta(repo_root)
    perf_csv = output_dir / "appendix_perf_delta.csv"
    perf_md = output_dir / "appendix_perf_delta.md"
    perf_df.to_csv(perf_csv, index=False, encoding="utf-8")
    write_markdown_table(perf_df, perf_md, title="SDFT-SFT Delta Table (final training setting)", ndigits=6)

    # Task 2
    tsne_md, tsne_used, tsne_missing, tsne_notes = build_tsne_paths(repo_root, output_dir, copy_selected=bool(int(args.copy_tsne_selected)))

    # Task 3
    geo_df, geo_used, geo_missing, geo_notes = build_shared_geometry(repo_root)
    geo_csv = output_dir / "appendix_shared_geometry_perf.csv"
    geo_md = output_dir / "appendix_shared_geometry_perf.md"
    geo_df.to_csv(geo_csv, index=False, encoding="utf-8")
    write_markdown_table(geo_df, geo_md, title="Shared geometry vs mean performance gain (train-domain level)", ndigits=6)

    # Missing report
    missing_md = output_dir / "appendix_missing_report.md"
    write_missing_report(
        missing_md,
        {
            "Task1_PerformanceDelta": perf_missing,
            "Task2_TSNE": tsne_missing + tsne_notes,
            "Task3_SharedGeometry": geo_missing + geo_notes,
        },
    )

    manifest = {
        "output_dir": str(output_dir),
        "files": {
            "appendix_perf_delta.csv": str(perf_csv),
            "appendix_perf_delta.md": str(perf_md),
            "appendix_tsne_paths.md": str(tsne_md),
            "appendix_shared_geometry_perf.csv": str(geo_csv),
            "appendix_shared_geometry_perf.md": str(geo_md),
            "appendix_missing_report.md": str(missing_md),
            "appendix_tsne_selected_dir": str(output_dir / "appendix_tsne_selected"),
        },
        "sources_used": sorted(set(perf_used + tsne_used + geo_used)),
        "rules": [
            "No experiment rerun",
            "No LaTeX/body modification",
            "No renaming original images",
            "Use final training setting (exclude epoch_1/base)",
        ],
        "note": "C_PF(Q,Q0) proxy is intentionally not processed.",
    }
    manifest_json = output_dir / "appendix_collect_manifest.json"
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(str(perf_csv))
    print(str(perf_md))
    print(str(tsne_md))
    print(str(geo_csv))
    print(str(geo_md))
    print(str(missing_md))
    print(str(manifest_json))


if __name__ == "__main__":
    main()
