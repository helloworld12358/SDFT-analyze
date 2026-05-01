#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract chapter-4 data package from existing local results only.
No existing experiment code/results are modified.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats  # type: ignore
except Exception:  # pragma: no cover
    scipy_stats = None


TRAIN_DOMAIN_ORDER = [
    "Alpaca",
    "Dolly",
    "GSM8K",
    "LIMA",
    "Magicoder",
    "OpenFunctions",
    "OpenHermes",
]

TEST_TASK_ORDER = ["AlpacaEval", "GSM8K", "HumanEval", "MultiArith", "OpenFunctions"]
HESSIAN_VIEW_ORDER = ["own-H", "cross-H", "mixed-H"]
STAGE_ORDER = ["initial_observation", "epoch_1", "final"]

ALLOWED_EXT = {
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".txt",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".npy",
    ".npz",
    ".parquet",
}

SEARCH_KEYWORDS = [
    "chapter4",
    "gradient",
    "grad",
    "grad_norm",
    "gradient_norm",
    "gradient_variance",
    "total_variance",
    "covariance",
    "cosine",
    "similarity",
    "hessian",
    "fisher",
    "datainf",
    "influence",
    "own",
    "own-h",
    "cross",
    "cross-h",
    "mixed",
    "mixed-h",
    "t_matrix",
    "c_matrix",
    "train_test",
    "rho",
    "rho_h",
    "offdiag",
    "lambda",
    "spectral",
    "eigen",
    "geometry",
    "performance_gain",
    "correlation",
]


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_df(path: Path, df: pd.DataFrame, columns: Sequence[str]) -> None:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = np.nan
    out = out[list(columns)]
    out.to_csv(path, index=False, encoding="utf-8-sig")


def normalize_train_domain(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower().replace("_", "")
    mapping = {
        "alpaca": "Alpaca",
        "dolly": "Dolly",
        "gsm8k": "GSM8K",
        "lima": "LIMA",
        "magicoder": "Magicoder",
        "openfunction": "OpenFunctions",
        "openfunctions": "OpenFunctions",
        "openhermes": "OpenHermes",
    }
    return mapping.get(s)


def normalize_stage(raw_stage: Any, stage_alias_log: Dict[str, List[str]]) -> Optional[str]:
    if raw_stage is None:
        return None
    s = str(raw_stage).strip().lower()
    s_clean = s.replace("-", "_")

    if s_clean in {"epoch_0", "epoch0", "initial", "start", "base", "epoch_0_observation"}:
        stage_alias_log["initial_observation"].append(str(raw_stage))
        return "initial_observation"
    if s_clean in {"epoch_1", "epoch1"}:
        stage_alias_log["epoch_1"].append(str(raw_stage))
        return "epoch_1"
    if s_clean in {"epoch_5", "epoch5", "final", "last", "trained", "terminal", "end", "endpoint"}:
        stage_alias_log["final"].append(str(raw_stage))
        return "final"

    if "epoch_5" in s_clean or "epoch5" in s_clean:
        stage_alias_log["final"].append(str(raw_stage))
        return "final"
    if "epoch_1" in s_clean or "epoch1" in s_clean:
        stage_alias_log["epoch_1"].append(str(raw_stage))
        return "epoch_1"
    if "epoch_0" in s_clean or "epoch0" in s_clean:
        stage_alias_log["initial_observation"].append(str(raw_stage))
        return "initial_observation"
    return None


def hessian_view_from_h_mode(h_mode: Any, method: Any) -> Optional[str]:
    hm = str(h_mode) if h_mode is not None else ""
    m = str(method).lower() if method is not None else ""
    if hm == "own":
        return "own-H"
    if hm == "mixed":
        return "mixed-H"
    if hm == "cross_oracle_sdft" and m == "sft":
        return "cross-H"
    if hm == "cross_oracle_sft" and m == "sdft":
        return "cross-H"
    return None


def sort_stage_key(stage: str) -> int:
    try:
        return STAGE_ORDER.index(stage)
    except Exception:
        return 999


def sort_view_key(view: str) -> int:
    try:
        return HESSIAN_VIEW_ORDER.index(view)
    except Exception:
        return 999


def matrix_to_df(mat: np.ndarray, row_names: Sequence[str], col_names: Sequence[str], row_label: str) -> pd.DataFrame:
    d = {row_label: list(row_names)}
    for j, c in enumerate(col_names):
        d[c] = [float(mat[i, j]) if np.isfinite(mat[i, j]) else np.nan for i in range(len(row_names))]
    return pd.DataFrame(d)


def upper_offdiag_values(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    vals: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            v = mat[i, j]
            if np.isfinite(v):
                vals.append(float(v))
    return np.asarray(vals, dtype=float)


def compute_corr(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if len(x) < 3:
        return None, None, None, None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None, None, None, None

    if scipy_stats is not None:
        try:
            pr = scipy_stats.pearsonr(x, y)
            sr = scipy_stats.spearmanr(x, y)
            return float(pr.statistic), float(pr.pvalue), float(sr.statistic), float(sr.pvalue)
        except Exception:
            pass

    px = pd.Series(x)
    py = pd.Series(y)
    return float(px.corr(py, method="pearson")), None, float(px.corr(py, method="spearman")), None


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = ensure_dir(repo_root / "chapter4_data_package")

    metadata: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": None,
        "searched_files": [],
        "output_sources": {},
        "aggregation_methods": {},
        "stage_alias_mapping": {},
        "missing_items_present": False,
        "legacy_fields_found": False,
        "aggregation_unknown_outputs": [],
    }
    missing_items: List[Dict[str, Any]] = []

    def add_missing(item_name: str, expected_output_file: str, reason: str, searched_directories: Optional[List[str]] = None, searched_keywords: Optional[List[str]] = None) -> None:
        missing_items.append(
            {
                "item_name": item_name,
                "expected_output_file": expected_output_file,
                "searched_keywords": searched_keywords or SEARCH_KEYWORDS,
                "searched_directories": searched_directories or [str(repo_root / "DataInf" / "results"), str(repo_root / "DataInf" / "script")],
                "reason": reason,
            }
        )

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
        metadata["git_commit"] = commit
    except Exception:
        metadata["git_commit"] = None

    searched_files: List[str] = []
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        if out_dir in p.parents:
            continue
        if p.suffix.lower() not in ALLOWED_EXT:
            continue
        lp = str(p).lower()
        if any(k in lp for k in SEARCH_KEYWORDS):
            searched_files.append(str(p.resolve()))
    searched_files.sort()
    metadata["searched_files"] = searched_files

    legacy_fields = ["V_align", "C_start", "C_end"]
    legacy_records: Dict[str, List[Dict[str, Any]]] = {k: [] for k in legacy_fields}
    for p_str in searched_files:
        p = Path(p_str)
        if p.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(p, nrows=5)
            except Exception:
                continue
            cols = list(df.columns)
            for f in legacy_fields:
                if f in cols:
                    sample_val = None
                    if len(df) > 0:
                        sample_val = df.iloc[0][f]
                        if isinstance(sample_val, float) and not math.isfinite(sample_val):
                            sample_val = None
                    legacy_records[f].append({"file": str(p.resolve()), "field_name": f, "sample_value": sample_val})
        elif p.suffix.lower() == ".json":
            try:
                obj = read_json(p)
            except Exception:
                continue

            def walk(o: Any, path_key: str = "") -> Iterable[Tuple[str, Any]]:
                if isinstance(o, dict):
                    for k, v in o.items():
                        nk = f"{path_key}.{k}" if path_key else k
                        yield nk, v
                        yield from walk(v, nk)
                elif isinstance(o, list):
                    for i, v in enumerate(o[:10]):
                        nk = f"{path_key}[{i}]"
                        yield nk, v
                        yield from walk(v, nk)

            for kpath, v in walk(obj):
                key_name = kpath.split(".")[-1]
                if key_name in legacy_fields:
                    legacy_records[key_name].append({"file": str(p.resolve()), "field_name": key_name, "sample_value": v})

    metadata["legacy_fields_found"] = any(len(v) > 0 for v in legacy_records.values())
    write_json(out_dir / "legacy_fields_detected.json", legacy_records)

    stage_alias_log: Dict[str, List[str]] = defaultdict(list)

    # gradient stats from existing chapter4 figure script (if available)
    grad_rows: List[Dict[str, Any]] = []
    grad_source_files: List[str] = []
    chap4_fig_script = repo_root / "DataInf" / "script" / "chapter4_make_paper_figures.py"
    if chap4_fig_script.exists():
        txt = chap4_fig_script.read_text(encoding="utf-8", errors="ignore")

        def parse_arr(name: str) -> Optional[List[float]]:
            m = re.search(rf"{re.escape(name)}\s*=\s*np\.array\s*\(\s*\[([^\]]+)\]", txt, flags=re.S)
            if not m:
                return None
            nums = [to_float(x.strip()) for x in m.group(1).split(",")]
            vals = [x for x in nums if x is not None]
            if len(vals) != 3:
                return None
            return [float(v) for v in vals]

        sft_init = parse_arr("sft_init")
        sdft_init = parse_arr("sdft_init")
        sft_lora = parse_arr("sft_lora")
        sdft_lora = parse_arr("sdft_lora")

        def push_rows(measurement_space: str, stage_raw: str, sft_vals: List[float], sdft_vals: List[float]) -> None:
            stage = normalize_stage(stage_raw, stage_alias_log)
            obs = "ALL_OBSERVED_TASKS"
            td = "ALL_TRAIN_DOMAINS"
            for method, arr in [("SFT", sft_vals), ("SDFT", sdft_vals)]:
                grad_rows.append(
                    {
                        "measurement_space": measurement_space,
                        "train_domain": td,
                        "observed_task": obs,
                        "stage": stage,
                        "method": method,
                        "mean_grad_norm": arr[0],
                        "grad_norm_variance": arr[1],
                        "grad_total_variance": arr[2],
                        "n_samples": np.nan,
                        "aggregation_level": "existing_summary_from_file",
                        "source_file": str(chap4_fig_script.resolve()),
                    }
                )

        if sft_init and sdft_init:
            push_rows("initial_model_parameter_space", "epoch_0", sft_init, sdft_init)
        if sft_lora and sdft_lora:
            push_rows("final_lora_parameter_space", "final", sft_lora, sdft_lora)
        grad_source_files = [str(chap4_fig_script.resolve())]

    grad_cols = [
        "measurement_space",
        "train_domain",
        "observed_task",
        "stage",
        "method",
        "mean_grad_norm",
        "grad_norm_variance",
        "grad_total_variance",
        "n_samples",
        "aggregation_level",
        "source_file",
    ]
    grad_df = pd.DataFrame(grad_rows)
    write_df(out_dir / "gradient_stats_full.csv", grad_df, grad_cols)
    metadata["output_sources"]["gradient_stats_full.csv"] = grad_source_files
    metadata["aggregation_methods"]["gradient_stats_full.csv"] = "existing_summary_from_file"
    metadata["aggregation_unknown_outputs"].append("gradient_stats_full.csv")

    delta_rows: List[Dict[str, Any]] = []
    if len(grad_df) > 0:
        key_cols = ["measurement_space", "train_domain", "observed_task", "stage"]
        for key, sub in grad_df.groupby(key_cols, dropna=False):
            if set(sub["method"]) >= {"SFT", "SDFT"}:
                sft = sub[sub["method"] == "SFT"].iloc[0]
                sdft = sub[sub["method"] == "SDFT"].iloc[0]
                delta_rows.append(
                    {
                        "measurement_space": key[0],
                        "train_domain": key[1],
                        "observed_task": key[2],
                        "stage": key[3],
                        "delta_mean_grad_norm": (to_float(sdft["mean_grad_norm"]) - to_float(sft["mean_grad_norm"])) if to_float(sdft["mean_grad_norm"]) is not None and to_float(sft["mean_grad_norm"]) is not None else np.nan,
                        "delta_grad_norm_variance": (to_float(sdft["grad_norm_variance"]) - to_float(sft["grad_norm_variance"])) if to_float(sdft["grad_norm_variance"]) is not None and to_float(sft["grad_norm_variance"]) is not None else np.nan,
                        "delta_grad_total_variance": (to_float(sdft["grad_total_variance"]) - to_float(sft["grad_total_variance"])) if to_float(sdft["grad_total_variance"]) is not None and to_float(sft["grad_total_variance"]) is not None else np.nan,
                        "sft_mean_grad_norm": sft["mean_grad_norm"],
                        "sdft_mean_grad_norm": sdft["mean_grad_norm"],
                        "sft_grad_norm_variance": sft["grad_norm_variance"],
                        "sdft_grad_norm_variance": sdft["grad_norm_variance"],
                        "sft_grad_total_variance": sft["grad_total_variance"],
                        "sdft_grad_total_variance": sdft["grad_total_variance"],
                        "n_samples_sft": sft["n_samples"],
                        "n_samples_sdft": sdft["n_samples"],
                        "source_file": sft["source_file"],
                    }
                )

    delta_cols = [
        "measurement_space",
        "train_domain",
        "observed_task",
        "stage",
        "delta_mean_grad_norm",
        "delta_grad_norm_variance",
        "delta_grad_total_variance",
        "sft_mean_grad_norm",
        "sdft_mean_grad_norm",
        "sft_grad_norm_variance",
        "sdft_grad_norm_variance",
        "sft_grad_total_variance",
        "sdft_grad_total_variance",
        "n_samples_sft",
        "n_samples_sdft",
        "source_file",
    ]
    delta_df = pd.DataFrame(delta_rows)
    write_df(out_dir / "gradient_stats_delta_full.csv", delta_df, delta_cols)
    metadata["output_sources"]["gradient_stats_delta_full.csv"] = grad_source_files
    metadata["aggregation_methods"]["gradient_stats_delta_full.csv"] = "derived_sdf_minus_sft"

    by_task_cols = [
        "measurement_space",
        "observed_task",
        "stage",
        "delta_mean_grad_norm",
        "delta_grad_norm_variance",
        "delta_grad_total_variance",
        "aggregation_method",
        "n_train_domains",
        "source_files",
    ]
    by_task_rows: List[Dict[str, Any]] = []
    valid_domain_delta = delta_df[delta_df["train_domain"].isin(TRAIN_DOMAIN_ORDER)] if len(delta_df) else pd.DataFrame()
    if len(valid_domain_delta) > 0:
        group_cols = ["measurement_space", "observed_task", "stage"]
        for key, sub in valid_domain_delta.groupby(group_cols, dropna=False):
            n_domains = sub["train_domain"].nunique()
            if n_domains > 0:
                by_task_rows.append(
                    {
                        "measurement_space": key[0],
                        "observed_task": key[1],
                        "stage": key[2],
                        "delta_mean_grad_norm": float(np.nanmean(sub["delta_mean_grad_norm"])),
                        "delta_grad_norm_variance": float(np.nanmean(sub["delta_grad_norm_variance"])),
                        "delta_grad_total_variance": float(np.nanmean(sub["delta_grad_total_variance"])),
                        "aggregation_method": "domain_equal_average",
                        "n_train_domains": int(n_domains),
                        "source_files": ";".join(sorted(set(sub["source_file"].dropna().astype(str).tolist()))),
                    }
                )
    else:
        add_missing(
            "gradient_stats_by_task_equal_domain_average",
            "gradient_stats_by_task_equal_domain_average.csv",
            "Per-train-domain gradient statistics were not found locally; only global figure-level aggregates are available.",
        )

    by_task_df = pd.DataFrame(by_task_rows)
    write_df(out_dir / "gradient_stats_by_task_equal_domain_average.csv", by_task_df, by_task_cols)
    metadata["output_sources"]["gradient_stats_by_task_equal_domain_average.csv"] = grad_source_files
    metadata["aggregation_methods"]["gradient_stats_by_task_equal_domain_average.csv"] = "domain_equal_average"

    # plain cosine directory + missing marker
    ensure_dir(out_dir / "plain_cosine_matrices")
    plain_candidates = [p for p in searched_files if "plain_cosine" in p.lower() or "cosine" in p.lower()]
    if len(plain_candidates) == 0:
        add_missing(
            "plain_cosine_matrices",
            "plain_cosine_matrices/*.csv",
            "No plain-gradient cosine matrix result files were found in local results.",
        )

    # task-task matrices from bundle
    tt_dir = ensure_dir(out_dir / "hessian_task_task_matrices")
    matrix_bundle = repo_root / "DataInf" / "results" / "schemeA" / "final_summary" / "schemeA_matrix_bundle_summary.json"
    tt_sources: List[str] = []
    task_task_store: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    if matrix_bundle.exists():
        payload = read_json(matrix_bundle)
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        for r in rows:
            if not isinstance(r, dict):
                continue
            td = normalize_train_domain(r.get("train_dataset"))
            st = normalize_stage(r.get("epoch"), stage_alias_log)
            method = str(r.get("method", "")).lower()
            view = hessian_view_from_h_mode(r.get("h_mode"), method)
            if td is None or st is None or view is None:
                continue
            if method not in {"sft", "sdft"}:
                continue
            T = np.asarray(r.get("T", []), dtype=float)
            C = np.asarray(r.get("C", []), dtype=float)
            if T.shape != (5, 5) or C.shape != (5, 5):
                continue
            key = (view, st, td, method.upper())
            task_task_store[key] = {"T": T, "C": C, "source": str(matrix_bundle.resolve())}
        tt_sources = [str(matrix_bundle.resolve())]
    else:
        add_missing("hessian_task_task_matrices", "hessian_task_task_matrices/*.csv", f"Missing matrix bundle: {matrix_bundle}")

    for view in HESSIAN_VIEW_ORDER:
        for st in STAGE_ORDER:
            for td in TRAIN_DOMAIN_ORDER:
                k_sft = (view, st, td, "SFT")
                k_sdft = (view, st, td, "SDFT")
                for method_lbl, key in [("SFT", k_sft), ("SDFT", k_sdft)]:
                    if key not in task_task_store:
                        continue
                    rec = task_task_store[key]
                    for metric_name, mk in [("I_H", "T"), ("rho_H", "C")]:
                        fpath = tt_dir / f"task_task__{view}__{st}__{td}__{metric_name}__{method_lbl}.csv"
                        matrix_to_df(rec[mk], TEST_TASK_ORDER, TEST_TASK_ORDER, "task").to_csv(fpath, index=False, encoding="utf-8-sig")
                if k_sft in task_task_store and k_sdft in task_task_store:
                    sft = task_task_store[k_sft]
                    sdft = task_task_store[k_sdft]
                    for metric_name, mk in [("I_H", "T"), ("rho_H", "C")]:
                        fpath = tt_dir / f"task_task__{view}__{st}__{td}__{metric_name}__Delta.csv"
                        matrix_to_df(sdft[mk] - sft[mk], TEST_TASK_ORDER, TEST_TASK_ORDER, "task").to_csv(fpath, index=False, encoding="utf-8-sig")

    metadata["output_sources"]["hessian_task_task_matrices/"] = tt_sources
    metadata["aggregation_methods"]["hessian_task_task_matrices/"] = "direct_read_or_delta_sdf_minus_sft"
    for view in HESSIAN_VIEW_ORDER:
        cnt = sum(1 for k in task_task_store if k[0] == view)
        if cnt == 0:
            add_missing(f"task_task_{view}", "hessian_task_task_matrices/*.csv", f"No {view} task-task matrices were found.")

    offdiag_rows: List[Dict[str, Any]] = []
    for view in HESSIAN_VIEW_ORDER:
        for st in STAGE_ORDER:
            for td in TRAIN_DOMAIN_ORDER:
                ks = (view, st, td, "SFT")
                kd = (view, st, td, "SDFT")
                if ks not in task_task_store or kd not in task_task_store:
                    continue
                vals = upper_offdiag_values(task_task_store[kd]["C"] - task_task_store[ks]["C"])
                if vals.size == 0:
                    continue
                offdiag_rows.append(
                    {
                        "hessian_view": view,
                        "stage": st,
                        "train_domain": td,
                        "metric": "rho_H",
                        "n_offdiag_positive": int(np.sum(vals > 0)),
                        "n_offdiag_total": int(vals.size),
                        "positive_ratio": float(np.sum(vals > 0) / vals.size),
                        "mean_delta_offdiag": float(np.mean(vals)),
                        "source_files": str(matrix_bundle.resolve()) if matrix_bundle.exists() else None,
                    }
                )

    offdiag_cols = ["hessian_view", "stage", "train_domain", "metric", "n_offdiag_positive", "n_offdiag_total", "positive_ratio", "mean_delta_offdiag", "source_files"]
    offdiag_df = pd.DataFrame(offdiag_rows)
    write_df(out_dir / "hessian_task_task_offdiag_summary.csv", offdiag_df, offdiag_cols)
    metadata["output_sources"]["hessian_task_task_offdiag_summary.csv"] = tt_sources
    metadata["aggregation_methods"]["hessian_task_task_offdiag_summary.csv"] = "per_train_domain"

    offdiag_aggr_rows: List[Dict[str, Any]] = []
    if len(offdiag_df) > 0:
        for key, sub in offdiag_df.groupby(["hessian_view", "stage", "metric"], dropna=False):
            total = int(sub["n_offdiag_total"].sum())
            pos = int(sub["n_offdiag_positive"].sum())
            n_domains = int(sub["train_domain"].nunique())
            offdiag_aggr_rows.append(
                {
                    "hessian_view": key[0],
                    "stage": key[1],
                    "metric": key[2],
                    "n_offdiag_positive": pos,
                    "n_offdiag_total": total,
                    "positive_ratio": float(pos / total) if total else np.nan,
                    "mean_delta_offdiag": float(np.nanmean(sub["mean_delta_offdiag"])),
                    "aggregation_method": "domain_equal_average",
                    "n_train_domains": n_domains,
                    "source_files": str(matrix_bundle.resolve()) if matrix_bundle.exists() else None,
                }
            )
    offdiag_aggr_cols = ["hessian_view", "stage", "metric", "n_offdiag_positive", "n_offdiag_total", "positive_ratio", "mean_delta_offdiag", "aggregation_method", "n_train_domains", "source_files"]
    offdiag_aggr_df = pd.DataFrame(offdiag_aggr_rows)
    write_df(out_dir / "hessian_task_task_offdiag_summary_aggregated.csv", offdiag_aggr_df, offdiag_aggr_cols)
    metadata["output_sources"]["hessian_task_task_offdiag_summary_aggregated.csv"] = tt_sources
    metadata["aggregation_methods"]["hessian_task_task_offdiag_summary_aggregated.csv"] = "domain_equal_average"

    # train-test own-H
    traintest_dir = ensure_dir(out_dir / "train_test_matrices")
    traintest_summary = repo_root / "DataInf" / "results" / "schemeA" / "train_test_rect" / "train_test_rect_summary.json"
    train_test_sources: List[str] = []
    train_test_store: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    if traintest_summary.exists():
        obj = read_json(traintest_summary)
        rows = obj.get("rows", []) if isinstance(obj, dict) else []
        for st in STAGE_ORDER:
            for m in ["SFT", "SDFT"]:
                train_test_store[(st, m)] = {
                    "T": np.full((len(TRAIN_DOMAIN_ORDER), len(TEST_TASK_ORDER)), np.nan, dtype=float),
                    "C": np.full((len(TRAIN_DOMAIN_ORDER), len(TEST_TASK_ORDER)), np.nan, dtype=float),
                }
        for r in rows:
            if not isinstance(r, dict) or str(r.get("status")) != "ok":
                continue
            td = normalize_train_domain(r.get("train_dataset"))
            st = normalize_stage(r.get("epoch"), stage_alias_log)
            m = str(r.get("method", "")).upper()
            if td not in TRAIN_DOMAIN_ORDER or st not in STAGE_ORDER or m not in {"SFT", "SDFT"}:
                continue
            tvals = [to_float(x) for x in r.get("T_row", [])]
            cvals = [to_float(x) for x in r.get("C_row", [])]
            if len(tvals) != len(TEST_TASK_ORDER) or len(cvals) != len(TEST_TASK_ORDER):
                continue
            i = TRAIN_DOMAIN_ORDER.index(td)
            train_test_store[(st, m)]["T"][i, :] = [np.nan if v is None else float(v) for v in tvals]
            train_test_store[(st, m)]["C"][i, :] = [np.nan if v is None else float(v) for v in cvals]
        train_test_sources = [str(traintest_summary.resolve())]
    else:
        add_missing("train_test_matrices_ownH", "train_test_matrices/*.csv", f"Missing train-test summary: {traintest_summary}")

    for st in STAGE_ORDER:
        sft = train_test_store.get((st, "SFT"))
        sdft = train_test_store.get((st, "SDFT"))
        if sft is None or sdft is None:
            continue
        for method_lbl, rec in [("SFT", sft), ("SDFT", sdft)]:
            for metric_name, mk in [("I_H_train_test", "T"), ("rho_H_train_test", "C")]:
                p = traintest_dir / f"train_test__own-H__{st}__{metric_name}__{method_lbl}.csv"
                matrix_to_df(rec[mk], TRAIN_DOMAIN_ORDER, TEST_TASK_ORDER, "train_domain").to_csv(p, index=False, encoding="utf-8-sig")
        for metric_name, mk in [("I_H_train_test", "T"), ("rho_H_train_test", "C")]:
            p = traintest_dir / f"train_test__own-H__{st}__{metric_name}__Delta.csv"
            matrix_to_df(sdft[mk] - sft[mk], TRAIN_DOMAIN_ORDER, TEST_TASK_ORDER, "train_domain").to_csv(p, index=False, encoding="utf-8-sig")

    metadata["output_sources"]["train_test_matrices/"] = train_test_sources
    metadata["aggregation_methods"]["train_test_matrices/"] = "direct_read_or_delta_sdf_minus_sft"
    add_missing("train_test_matrices_cross-H", "train_test_matrices/train_test__cross-H__*.csv", "No local cross-H train-test rectangular summary was found.")
    add_missing("train_test_matrices_mixed-H", "train_test_matrices/train_test__mixed-H__*.csv", "No local mixed-H train-test rectangular summary was found.")

    ttb_rows: List[Dict[str, Any]] = []
    for st in STAGE_ORDER:
        sft = train_test_store.get((st, "SFT"))
        sdft = train_test_store.get((st, "SDFT"))
        if sft is None or sdft is None:
            continue
        delta_c = sdft["C"] - sft["C"]
        for j, task in enumerate(TEST_TASK_ORDER):
            vals = delta_c[:, j]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            n_total = len(TRAIN_DOMAIN_ORDER)
            n_pos = int(np.sum(vals > 0))
            ttb_rows.append(
                {
                    "hessian_view": "own-H",
                    "stage": st,
                    "test_task": task,
                    "n_positive_train_domains": n_pos,
                    "n_total_train_domains": n_total,
                    "positive_ratio": float(n_pos / n_total) if n_total else np.nan,
                    "mean_delta_rho_H": float(np.mean(vals)),
                    "aggregation_method": "domain_equal_average",
                    "source_files": str(traintest_summary.resolve()) if traintest_summary.exists() else None,
                }
            )
    ttb_cols = ["hessian_view", "stage", "test_task", "n_positive_train_domains", "n_total_train_domains", "positive_ratio", "mean_delta_rho_H", "aggregation_method", "source_files"]
    ttb_df = pd.DataFrame(ttb_rows)
    write_df(out_dir / "train_test_by_task_summary.csv", ttb_df, ttb_cols)
    metadata["output_sources"]["train_test_by_task_summary.csv"] = train_test_sources
    metadata["aggregation_methods"]["train_test_by_task_summary.csv"] = "domain_equal_average"

    # spectral metrics from per-method summary
    per_method_csv = repo_root / "DataInf" / "results" / "schemeA" / "final_summary" / "schemeA_per_method_summary.csv"
    delta_csv = repo_root / "DataInf" / "results" / "schemeA" / "final_summary" / "schemeA_sft_minus_sdft_summary.csv"

    spectral_rows: List[Dict[str, Any]] = []
    spectral_sources: List[str] = []
    if per_method_csv.exists():
        d = pd.read_csv(per_method_csv)
        for _, r in d.iterrows():
            method_raw = str(r.get("method", "")).lower()
            view = hessian_view_from_h_mode(r.get("h_mode"), method_raw)
            if not view:
                continue
            td = normalize_train_domain(r.get("train_dataset"))
            st = normalize_stage(r.get("epoch"), stage_alias_log)
            if td not in TRAIN_DOMAIN_ORDER or st not in STAGE_ORDER:
                continue
            method = method_raw.upper()
            if method not in {"SFT", "SDFT"}:
                continue
            spectral_rows.append(
                {
                    "hessian_view": view,
                    "stage": st,
                    "train_domain": td,
                    "method": method,
                    "lambda1_rho_H": to_float(r.get("lambda1_C")),
                    "lambda_gap_rho_H": to_float(r.get("lambda1_minus_lambda2_C")),
                    "rho_off": to_float(r.get("mean_offdiag_C")),
                    "rho_off_fro": to_float(r.get("fro_offdiag_C")),
                    "source_file": str(r.get("summary_json")) if isinstance(r.get("summary_json"), str) and str(r.get("summary_json")) else str(per_method_csv.resolve()),
                }
            )
        spectral_sources.append(str(per_method_csv.resolve()))
    else:
        add_missing("spectral_metrics_full", "spectral_metrics_full.csv", f"Missing per-method summary file: {per_method_csv}")

    spectral_cols = ["hessian_view", "stage", "train_domain", "method", "lambda1_rho_H", "lambda_gap_rho_H", "rho_off", "rho_off_fro", "source_file"]
    spectral_df = pd.DataFrame(spectral_rows)
    if len(spectral_df) > 0:
        spectral_df["_view_order"] = spectral_df["hessian_view"].apply(sort_view_key)
        spectral_df["_stage_order"] = spectral_df["stage"].apply(sort_stage_key)
        spectral_df["_domain_order"] = spectral_df["train_domain"].apply(lambda x: TRAIN_DOMAIN_ORDER.index(x) if x in TRAIN_DOMAIN_ORDER else 999)
        spectral_df = spectral_df.sort_values(["_view_order", "_stage_order", "_domain_order", "method"]).drop(columns=["_view_order", "_stage_order", "_domain_order"])
    write_df(out_dir / "spectral_metrics_full.csv", spectral_df, spectral_cols)
    metadata["output_sources"]["spectral_metrics_full.csv"] = spectral_sources
    metadata["aggregation_methods"]["spectral_metrics_full.csv"] = "per_train_domain"

    spec_delta_rows: List[Dict[str, Any]] = []
    if len(spectral_df) > 0:
        for key, sub in spectral_df.groupby(["hessian_view", "stage", "train_domain"], dropna=False):
            if set(sub["method"].astype(str)) < {"SFT", "SDFT"}:
                continue
            sft = sub[sub["method"] == "SFT"].iloc[0]
            sdft = sub[sub["method"] == "SDFT"].iloc[0]

            def dlt(a: Any, b: Any) -> Optional[float]:
                va = to_float(a)
                vb = to_float(b)
                if va is None or vb is None:
                    return None
                return va - vb

            spec_delta_rows.append(
                {
                    "hessian_view": key[0],
                    "stage": key[1],
                    "train_domain": key[2],
                    "delta_lambda1_rho_H": dlt(sdft["lambda1_rho_H"], sft["lambda1_rho_H"]),
                    "delta_lambda_gap_rho_H": dlt(sdft["lambda_gap_rho_H"], sft["lambda_gap_rho_H"]),
                    "delta_rho_off": dlt(sdft["rho_off"], sft["rho_off"]),
                    "delta_rho_off_fro": dlt(sdft["rho_off_fro"], sft["rho_off_fro"]),
                    "sft_lambda1_rho_H": sft["lambda1_rho_H"],
                    "sdft_lambda1_rho_H": sdft["lambda1_rho_H"],
                    "sft_lambda_gap_rho_H": sft["lambda_gap_rho_H"],
                    "sdft_lambda_gap_rho_H": sdft["lambda_gap_rho_H"],
                    "sft_rho_off": sft["rho_off"],
                    "sdft_rho_off": sdft["rho_off"],
                    "sft_rho_off_fro": sft["rho_off_fro"],
                    "sdft_rho_off_fro": sdft["rho_off_fro"],
                    "source_files": ";".join(sorted(set([str(sft["source_file"]), str(sdft["source_file"])]))),
                }
            )

    spec_delta_cols = [
        "hessian_view",
        "stage",
        "train_domain",
        "delta_lambda1_rho_H",
        "delta_lambda_gap_rho_H",
        "delta_rho_off",
        "delta_rho_off_fro",
        "sft_lambda1_rho_H",
        "sdft_lambda1_rho_H",
        "sft_lambda_gap_rho_H",
        "sdft_lambda_gap_rho_H",
        "sft_rho_off",
        "sdft_rho_off",
        "sft_rho_off_fro",
        "sdft_rho_off_fro",
        "source_files",
    ]
    spec_delta_df = pd.DataFrame(spec_delta_rows)
    if len(spec_delta_df) > 0:
        spec_delta_df["_view_order"] = spec_delta_df["hessian_view"].apply(sort_view_key)
        spec_delta_df["_stage_order"] = spec_delta_df["stage"].apply(sort_stage_key)
        spec_delta_df["_domain_order"] = spec_delta_df["train_domain"].apply(lambda x: TRAIN_DOMAIN_ORDER.index(x) if x in TRAIN_DOMAIN_ORDER else 999)
        spec_delta_df = spec_delta_df.sort_values(["_view_order", "_stage_order", "_domain_order"]).drop(columns=["_view_order", "_stage_order", "_domain_order"])
    write_df(out_dir / "spectral_metrics_delta_full.csv", spec_delta_df, spec_delta_cols)
    metadata["output_sources"]["spectral_metrics_delta_full.csv"] = spectral_sources
    metadata["aggregation_methods"]["spectral_metrics_delta_full.csv"] = "derived_sdf_minus_sft"

    ref_rows: List[Dict[str, Any]] = []
    if len(spec_delta_df) > 0:
        for key, sub in spec_delta_df.groupby(["hessian_view", "stage"], dropna=False):
            ref_rows.append(
                {
                    "hessian_view": key[0],
                    "stage": key[1],
                    "delta_lambda1_rho_H_mean": float(np.nanmean(sub["delta_lambda1_rho_H"])),
                    "delta_lambda_gap_rho_H_mean": float(np.nanmean(sub["delta_lambda_gap_rho_H"])),
                    "delta_rho_off_mean": float(np.nanmean(sub["delta_rho_off"])),
                    "aggregation_method": "domain_equal_average",
                    "n_train_domains": int(sub["train_domain"].nunique()),
                    "source_files": ";".join(sorted(set(sub["source_files"].dropna().astype(str).tolist()))),
                }
            )
    ref_cols = ["hessian_view", "stage", "delta_lambda1_rho_H_mean", "delta_lambda_gap_rho_H_mean", "delta_rho_off_mean", "aggregation_method", "n_train_domains", "source_files"]
    ref_df = pd.DataFrame(ref_rows)
    if len(ref_df) > 0:
        ref_df["_view_order"] = ref_df["hessian_view"].apply(sort_view_key)
        ref_df["_stage_order"] = ref_df["stage"].apply(sort_stage_key)
        ref_df = ref_df.sort_values(["_view_order", "_stage_order"]).drop(columns=["_view_order", "_stage_order"])
    write_df(out_dir / "spectral_metrics_reference_view_summary.csv", ref_df, ref_cols)
    metadata["output_sources"]["spectral_metrics_reference_view_summary.csv"] = spectral_sources
    metadata["aggregation_methods"]["spectral_metrics_reference_view_summary.csv"] = "domain_equal_average"

    # correlation input + summary
    corr_input_rows: List[Dict[str, Any]] = []
    if delta_csv.exists() and len(spec_delta_df) > 0:
        ddelta = pd.read_csv(delta_csv)
        perf_lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for _, r in ddelta.iterrows():
            hm = str(r.get("h_mode", ""))
            if hm == "own":
                view = "own-H"
            elif hm == "mixed":
                view = "mixed-H"
            elif hm == "cross_oracle_sdft":
                view = "cross-H"
            else:
                continue
            td = normalize_train_domain(r.get("train_dataset"))
            st = normalize_stage(r.get("epoch"), stage_alias_log)
            if td not in TRAIN_DOMAIN_ORDER or st not in STAGE_ORDER:
                continue
            metric_cols = [
                "delta_metric_accuracy_generic",
                "delta_metric_multiarith",
                "delta_metric_openfunction",
                "delta_metric_gsm8k",
                "delta_metric_safety",
                "delta_metric_humaneval",
            ]
            vals = [to_float(r.get(c)) for c in metric_cols]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            perf_lookup[(view, st, td)] = {
                "mean_performance_gain": -float(np.mean(vals)),  # source is sft-sdft
                "source": str(delta_csv.resolve()),
            }

        for _, r in spec_delta_df.iterrows():
            key = (str(r["hessian_view"]), str(r["stage"]), str(r["train_domain"]))
            if key not in perf_lookup:
                continue
            perf = perf_lookup[key]
            corr_input_rows.append(
                {
                    "hessian_view": key[0],
                    "stage": key[1],
                    "train_domain": key[2],
                    "mean_performance_gain": perf["mean_performance_gain"],
                    "delta_lambda1_rho_H": r["delta_lambda1_rho_H"],
                    "delta_lambda_gap_rho_H": r["delta_lambda_gap_rho_H"],
                    "delta_rho_off": r["delta_rho_off"],
                    "delta_rho_off_fro": r["delta_rho_off_fro"],
                    "source_files": ";".join(sorted(set([perf["source"], str(r["source_files"])]))),
                }
            )
    else:
        add_missing("geometry_performance_correlation_input", "geometry_performance_correlation_input.csv", "Either spectral delta table or schemeA_sft_minus_sdft_summary.csv is missing.")

    corr_input_cols = ["hessian_view", "stage", "train_domain", "mean_performance_gain", "delta_lambda1_rho_H", "delta_lambda_gap_rho_H", "delta_rho_off", "delta_rho_off_fro", "source_files"]
    corr_input_df = pd.DataFrame(corr_input_rows)
    if len(corr_input_df) > 0:
        corr_input_df["_view_order"] = corr_input_df["hessian_view"].apply(sort_view_key)
        corr_input_df["_stage_order"] = corr_input_df["stage"].apply(sort_stage_key)
        corr_input_df["_domain_order"] = corr_input_df["train_domain"].apply(lambda x: TRAIN_DOMAIN_ORDER.index(x) if x in TRAIN_DOMAIN_ORDER else 999)
        corr_input_df = corr_input_df.sort_values(["_view_order", "_stage_order", "_domain_order"]).drop(columns=["_view_order", "_stage_order", "_domain_order"])
    write_df(out_dir / "geometry_performance_correlation_input.csv", corr_input_df, corr_input_cols)
    metadata["output_sources"]["geometry_performance_correlation_input.csv"] = [str(delta_csv.resolve()), str(per_method_csv.resolve())]
    metadata["aggregation_methods"]["geometry_performance_correlation_input.csv"] = "per_train_domain"

    corr_rows: List[Dict[str, Any]] = []
    if len(corr_input_df) > 0:
        for (view, st), sub in corr_input_df.groupby(["hessian_view", "stage"], dropna=False):
            for metric in ["delta_lambda1_rho_H", "delta_lambda_gap_rho_H", "delta_rho_off", "delta_rho_off_fro"]:
                x = sub[metric].astype(float).to_numpy()
                y = sub["mean_performance_gain"].astype(float).to_numpy()
                mask = np.isfinite(x) & np.isfinite(y)
                pr, pp, sr, sp = compute_corr(x[mask], y[mask])
                corr_rows.append(
                    {
                        "hessian_view": view,
                        "stage": st,
                        "metric": metric,
                        "pearson_r": pr,
                        "pearson_p": pp,
                        "spearman_rho": sr,
                        "spearman_p": sp,
                        "n_train_domains": int(np.sum(mask)),
                        "source_files": ";".join(sorted(set(sub["source_files"].dropna().astype(str).tolist()))),
                    }
                )
    else:
        add_missing("geometry_performance_correlation_summary", "geometry_performance_correlation_summary.csv", "Correlation input table is empty.")

    corr_cols = ["hessian_view", "stage", "metric", "pearson_r", "pearson_p", "spearman_rho", "spearman_p", "n_train_domains", "source_files"]
    corr_df = pd.DataFrame(corr_rows)
    if len(corr_df) > 0:
        corr_df["_view_order"] = corr_df["hessian_view"].apply(sort_view_key)
        corr_df["_stage_order"] = corr_df["stage"].apply(sort_stage_key)
        corr_df = corr_df.sort_values(["_view_order", "_stage_order", "metric"]).drop(columns=["_view_order", "_stage_order"])
    write_df(out_dir / "geometry_performance_correlation_summary.csv", corr_df, corr_cols)
    metadata["output_sources"]["geometry_performance_correlation_summary.csv"] = [str(delta_csv.resolve()), str(per_method_csv.resolve())]
    metadata["aggregation_methods"]["geometry_performance_correlation_summary.csv"] = "per_train_domain"

    if len(grad_df) == 0:
        add_missing("gradient_stats_full", "gradient_stats_full.csv", "No local gradient statistics source was found.")
    elif not any(x in TRAIN_DOMAIN_ORDER for x in grad_df["train_domain"].dropna().astype(str).tolist()):
        add_missing("gradient_stats_per_train_domain_and_task", "gradient_stats_full.csv", "Only global aggregate values were found; per-train-domain and per-observed-task rows are unavailable locally.")

    metadata["stage_alias_mapping"] = {k: sorted(set(v)) for k, v in stage_alias_log.items()}
    metadata["missing_items_present"] = len(missing_items) > 0

    write_json(out_dir / "missing_items.json", missing_items)
    write_json(out_dir / "metadata.json", metadata)

    output_files = sorted([p.name for p in out_dir.glob("*") if p.is_file()])
    readme_lines: List[str] = []
    readme_lines.append("# Chapter4 Data Package")
    readme_lines.append("")
    readme_lines.append(f"Generated at: {metadata['generated_at']}")
    readme_lines.append(f"Git commit: {metadata['git_commit']}")
    readme_lines.append("")
    readme_lines.append("## 1. Generated files")
    for fn in output_files:
        readme_lines.append(f"- {fn}")
    readme_lines.append("")
    readme_lines.append("## 2. Mapping to chapter-4 content")
    readme_lines.append("- gradient_stats_full.csv / gradient_stats_delta_full.csv: gradient statistics (from existing chapter4 figure script aggregate values).")
    readme_lines.append("- hessian_task_task_matrices/: task-task I_H/rho_H matrices for own-H/cross-H/mixed-H by stage/domain/method + Delta.")
    readme_lines.append("- hessian_task_task_offdiag_summary*.csv: off-diagonal Delta(rho_H) summaries.")
    readme_lines.append("- train_test_matrices/: 7x5 own-H train-test matrices (I_H_train_test/rho_H_train_test) + Delta.")
    readme_lines.append("- train_test_by_task_summary.csv: per-task domain-equal average summaries for own-H.")
    readme_lines.append("- spectral_metrics_full.csv / spectral_metrics_delta_full.csv: lambda/offdiag geometry metrics and SDFT-SFT deltas.")
    readme_lines.append("- spectral_metrics_reference_view_summary.csv: view-level domain-equal averages.")
    readme_lines.append("- geometry_performance_correlation_input.csv / geometry_performance_correlation_summary.csv: per-domain inputs and correlation stats.")
    readme_lines.append("- metadata.json / missing_items.json / legacy_fields_detected.json: provenance + missing + legacy field detection.")
    readme_lines.append("")
    readme_lines.append("## 3. Read vs derived")
    readme_lines.append("- Directly read: matrix bundle, per-method summary, delta summary, train-test rectangular summary, chapter4 figure constants.")
    readme_lines.append("- Derived: all Delta tables computed as SDFT - SFT in this package.")
    readme_lines.append("- Domain-equal averages: files whose aggregation_method is domain_equal_average.")
    readme_lines.append("")
    readme_lines.append("## 4. Missing or partial")
    if missing_items:
        for m in missing_items:
            readme_lines.append(f"- {m['item_name']}: {m['reason']}")
    else:
        readme_lines.append("- None")
    readme_lines.append("")
    readme_lines.append("## 5. Aggregation unknown")
    if metadata.get("aggregation_unknown_outputs"):
        for x in metadata["aggregation_unknown_outputs"]:
            readme_lines.append(f"- {x}")
    else:
        readme_lines.append("- None")
    readme_lines.append("")
    readme_lines.append("## 6. Legacy fields detection")
    readme_lines.append(f"- legacy fields found: {metadata['legacy_fields_found']}")
    readme_lines.append("- see legacy_fields_detected.json for details")
    readme_lines.append("")
    readme_lines.append("## 7. Visualizations")
    readme_lines.append("- No new visualization is generated by this extractor.")
    readme_lines.append("- Existing chapter-4 PDFs found in repo: figures/chapter4_gradient_statistics.pdf, figures/chapter4_ownH_dolly_TC_heatmap.pdf, figures/chapter4_train_test_deltaC_heatmap.pdf")
    (out_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    generated_output_files = len([p for p in out_dir.glob("*") if p.is_file()])
    print(f"FOUND_SOURCE_FILES={len(searched_files)}")
    print(f"GENERATED_OUTPUT_FILES={generated_output_files}")
    print(f"MISSING_ITEM_CLASSES={len(missing_items)}")
    print(f"LEGACY_FIELDS_FOUND={metadata['legacy_fields_found']}")
    print(f"OUTPUT_DIR={out_dir.resolve()}")


if __name__ == "__main__":
    main()
