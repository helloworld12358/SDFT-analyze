#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MODELS: List[str] = [
    "gsm8k",
    "openfunction",
    "magicoder",
    "alpaca",
    "dolly",
    "lima",
    "openhermes",
]
DEFAULT_TRAIN_DATASETS: List[str] = list(DEFAULT_MODELS)
DEFAULT_EPOCHS: List[str] = ["epoch_0", "epoch_1", "epoch_5"]
DEFAULT_METHODS: List[str] = ["sft", "sdft"]
DEFAULT_TASKS: List[str] = ["alpaca_eval", "gsm8k", "humaneval", "multiarith", "openfunction"]

_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
# Compatible with:
#   shape: (5, 5)
#   shape = (5,5)
#   shape：(5，5)
_SHAPE_RE = re.compile(
    r"shape\s*[:=：]\s*[（(]\s*(\d+)\s*[,，]\s*(\d+)\s*[)）]",
    re.IGNORECASE,
)


def _normalize_header_text(s: str) -> str:
    return (
        s.strip()
        .lower()
        .replace("＝", "=")
        .replace("—", "-")
        .replace("－", "-")
    )


def _is_section_header_line(s: str) -> bool:
    low = _normalize_header_text(s)
    return low.startswith("===") or low.startswith("---")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def detect_datainf_root(datainf_root: Optional[str] = None) -> str:
    if datainf_root:
        return os.path.abspath(datainf_root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, ".."))


def resolve_sdft_root(datainf_root: str) -> str:
    return os.path.normpath(os.path.join(datainf_root, "..", "sdft"))


def split_csv_arg(s: Optional[str], default: Sequence[str]) -> List[str]:
    if s is None:
        return list(default)
    out = [x.strip() for x in s.split(",") if x.strip()]
    return out if out else list(default)


def normalize_epoch_tag(epoch: str) -> str:
    e = epoch.strip()
    if e in DEFAULT_EPOCHS:
        return e
    if e in ("0", "1", "5"):
        return f"epoch_{e}"
    if e.startswith("epoch") and "_" not in e:
        tail = e[len("epoch") :]
        if tail in ("0", "1", "5"):
            return f"epoch_{tail}"
    return e


def normalize_epoch_list(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for it in items:
        e = normalize_epoch_tag(it)
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def normalize_method_token(method: str) -> List[str]:
    m = method.strip().lower()
    if m in ("both", "all", "sft,sdft", "sdft,sft"):
        return ["sft", "sdft"]
    if m in ("sft", "sdft"):
        return [m]
    return []


def normalize_method_list(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for it in items:
        for m in normalize_method_token(it):
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


def choose_existing_dir(paths: Sequence[str]) -> Optional[str]:
    for p in paths:
        if os.path.isdir(p):
            return p
    return None


def resolve_result_root(datainf_root: str, prefer_plural: bool = True) -> str:
    plural = os.path.join(datainf_root, "results")
    singular = os.path.join(datainf_root, "result")
    if prefer_plural:
        if os.path.isdir(plural):
            return plural
        if os.path.isdir(singular):
            return singular
        return plural
    if os.path.isdir(singular):
        return singular
    if os.path.isdir(plural):
        return plural
    return singular


def resolve_existing_result_roots(datainf_root: str, explicit_roots: Optional[Sequence[str]] = None) -> List[str]:
    roots: List[str] = []

    def _append(p: Optional[str]) -> None:
        if not p:
            return
        q = os.path.abspath(p)
        if q not in roots:
            roots.append(q)

    _append(os.path.join(datainf_root, "result"))
    _append(os.path.join(datainf_root, "results"))

    env_roots = os.environ.get("SCHEMEA_EXISTING_RESULT_ROOTS", "").strip()
    if env_roots:
        for p in [x.strip() for x in env_roots.split(",") if x.strip()]:
            _append(p)

    if explicit_roots:
        for p in explicit_roots:
            _append(p)

    expanded: List[str] = []
    for p in roots:
        expanded.append(p)
        if os.path.basename(p) not in ("result", "results"):
            expanded.append(os.path.join(p, "result"))
            expanded.append(os.path.join(p, "results"))

    dedup: List[str] = []
    seen = set()
    for p in expanded:
        q = os.path.abspath(p)
        if q not in seen:
            seen.add(q)
            dedup.append(q)
    return dedup


def resolve_grad_path(datainf_root: str, epoch: str, method: str, model: str, task: str) -> str:
    cands = [
        os.path.join(datainf_root, "output_grads", epoch, method, model, f"{task}.pt"),
        os.path.join(datainf_root, "output_grad", epoch, method, model, f"{task}.pt"),
        os.path.join(datainf_root, "src", "result", "output_grad", epoch, method, model, f"{task}.pt"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return cands[0]


def resolve_checkpoint_path(sdft_root: str, epoch: str, model: str, method: str) -> Optional[str]:
    if epoch == "epoch_0":
        return None
    if epoch == "epoch_1":
        p = os.path.join(sdft_root, "epoch1_checkpoints", model, method)
        return p if os.path.isdir(p) else None
    p = os.path.join(sdft_root, "checkpoints", model, method)
    return p if os.path.isdir(p) else None


def resolve_train_dataset_path(sdft_root: str, model: str, method: str) -> str:
    fn = f"{model}_train.json" if method == "sft" else f"distilled_{model}.json"
    return os.path.join(sdft_root, "data", model, fn)


def load_json_records(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("data", "records", "examples", "items"):
            if isinstance(obj.get(key), list):
                return obj[key]
        return [obj]
    return []


def load_jsonl_records(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
    return out


def load_records_any(path: str) -> List[dict]:
    if not os.path.isfile(path):
        return []
    if path.endswith(".jsonl") or path.endswith(".ndjson"):
        return load_jsonl_records(path)
    return load_json_records(path)


def save_records_json(path: str, records: Sequence[dict]) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(records), f, ensure_ascii=False, indent=2)
    return path


def merge_records_for_mixed_h(sft_train_path: str, sdft_distilled_path: str, out_path: str) -> Tuple[Optional[str], Optional[str]]:
    if not os.path.isfile(sft_train_path):
        return None, f"missing sft train dataset: {sft_train_path}"
    if not os.path.isfile(sdft_distilled_path):
        return None, f"missing sdft distilled dataset: {sdft_distilled_path}"
    a = load_records_any(sft_train_path)
    b = load_records_any(sdft_distilled_path)
    if not a and not b:
        return None, "both input datasets are empty"
    save_records_json(out_path, list(a) + list(b))
    return out_path, None

def _extract_floats(line: str) -> List[float]:
    vals = _FLOAT_RE.findall(line)
    out: List[float] = []
    for v in vals:
        try:
            out.append(float(v))
        except Exception:
            continue
    return out


def _parse_shape(line: str) -> Optional[Tuple[int, int]]:
    m = _SHAPE_RE.search(line)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _parse_matrix_after_shape(lines: Sequence[str], shape_idx: int) -> Tuple[Optional[np.ndarray], int]:
    shape = _parse_shape(lines[shape_idx])
    if not shape:
        return None, shape_idx + 1
    nrow, ncol = shape
    rows: List[List[float]] = []
    i = shape_idx + 1
    while i < len(lines) and len(rows) < nrow:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if _is_section_header_line(s):
            break
        vals = _extract_floats(lines[i])
        if len(vals) >= ncol:
            rows.append(vals[:ncol])
        i += 1
    if len(rows) != nrow:
        return None, i
    return np.asarray(rows, dtype=np.float64), i


def _section_to_method(line: str) -> Optional[str]:
    low = _normalize_header_text(line)
    if not _is_section_header_line(low):
        return None
    if "diff" in low:
        return None
    if "sdft" in low:
        return "sdft"
    if "sft" in low:
        return "sft"
    return None


def parse_method_matrices_from_analysis_txt(path: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        method = _section_to_method(lines[i])
        if not method:
            i += 1
            continue

        j = i + 1
        found_shape = None
        while j < len(lines):
            if lines[j].strip().startswith("==="):
                break
            if _parse_shape(lines[j]) is not None:
                found_shape = j
                break
            j += 1

        if found_shape is None:
            i += 1
            continue

        mat, next_idx = _parse_matrix_after_shape(lines, found_shape)
        if mat is not None:
            out[method] = mat
        i = max(next_idx, i + 1)
    return out


def find_analysis_corr_file(analysis_safe_dir: str) -> Optional[str]:
    base = os.path.join(analysis_safe_dir, "analysis_corr_safe.txt")
    if os.path.isfile(base):
        return base
    cands = sorted(glob.glob(os.path.join(analysis_safe_dir, "analysis_corr_safe_*.txt")))
    return cands[-1] if cands else None


def load_existing_ownh_from_analysis(result_roots: Sequence[str], train_dataset: str, epoch: str) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {
        "sft": {"T": None, "C": None, "analysis_log_path": None, "analysis_corr_path": None, "source_root": None},
        "sdft": {"T": None, "C": None, "analysis_log_path": None, "analysis_corr_path": None, "source_root": None},
    }

    for root in result_roots:
        base = os.path.join(root, train_dataset, epoch)
        analysis_log = os.path.join(base, "analysis", "analysis_log.txt")
        analysis_corr = find_analysis_corr_file(os.path.join(base, "analysis_safe"))

        mats_t = parse_method_matrices_from_analysis_txt(analysis_log) if os.path.isfile(analysis_log) else {}
        mats_c = parse_method_matrices_from_analysis_txt(analysis_corr) if analysis_corr else {}

        for m in ("sft", "sdft"):
            if out[m]["T"] is None and mats_t.get(m) is not None:
                out[m]["T"] = mats_t[m]
                out[m]["analysis_log_path"] = analysis_log
                out[m]["source_root"] = root
            if out[m]["C"] is None and mats_c.get(m) is not None:
                out[m]["C"] = mats_c[m]
                out[m]["analysis_corr_path"] = analysis_corr
                out[m]["source_root"] = root

        if all(out[m]["T"] is not None and out[m]["C"] is not None for m in ("sft", "sdft")):
            break

    return out


def _load_json_score(path: str) -> Optional[float]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        score = obj.get("score", None)
        if score is None:
            return None
        return float(score)
    except Exception:
        return None


def matrix_from_pairwise_json(pairwise_dir: str, names: Sequence[str]) -> Optional[np.ndarray]:
    if not os.path.isdir(pairwise_dir):
        return None
    n = len(names)
    K = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(i, n):
            candidates = [
                os.path.join(pairwise_dir, f"sim_{names[i]}_{names[j]}.json"),
                os.path.join(pairwise_dir, f"sim_{names[j]}_{names[i]}.json"),
                os.path.join(pairwise_dir, f"sim_{i}_{j}.json"),
                os.path.join(pairwise_dir, f"sim_{j}_{i}.json"),
            ]
            score = None
            for p in candidates:
                if os.path.isfile(p):
                    score = _load_json_score(p)
                    if score is not None:
                        break
            if score is not None:
                K[i, j] = score
                K[j, i] = score
    return None if np.all(np.isnan(K)) else K


def load_pairwise_matrix_any(
    datainf_root: str,
    model: str,
    epoch: str,
    method: str,
    names: Sequence[str],
    extra_result_roots: Optional[Sequence[str]] = None,
) -> Optional[np.ndarray]:
    roots = [os.path.join(datainf_root, "results"), os.path.join(datainf_root, "result")]
    if extra_result_roots:
        roots.extend(extra_result_roots)

    stem = f"pairwise_matrix_{model}_{epoch}_{method}"
    for root in roots:
        base = os.path.join(root, model, epoch, method)
        npy = os.path.join(base, f"{stem}.npy")
        csvp = os.path.join(base, f"{stem}.csv")
        if os.path.isfile(npy):
            try:
                return np.load(npy)
            except Exception:
                pass
        if os.path.isfile(csvp):
            try:
                return np.loadtxt(csvp, delimiter=",")
            except Exception:
                pass
        for pair_dir_name in ("pairwise_results", "pairwise_result"):
            pair_dir = os.path.join(base, pair_dir_name)
            m = matrix_from_pairwise_json(pair_dir, names)
            if m is not None:
                return m
    return None


def covariance_to_correlation(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    d = np.diag(K).copy()
    denom = np.sqrt(np.outer(d, d))
    with np.errstate(divide="ignore", invalid="ignore"):
        C = K / denom
    C[~np.isfinite(C)] = np.nan
    for i in range(C.shape[0]):
        if np.isfinite(d[i]) and d[i] != 0:
            C[i, i] = 1.0
    return C


def eigendecompose_raw(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = np.asarray(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Matrix must be square, got shape={M.shape}")
    if np.allclose(M, M.T, atol=1e-8, rtol=0.0):
        vals, vecs = np.linalg.eigh(M)
    else:
        vals, vecs = np.linalg.eig(M)
    idx = np.argsort(-np.real(vals))
    return vals[idx], vecs[:, idx]


def _safe_cond(M: np.ndarray) -> Optional[float]:
    try:
        c = float(np.linalg.cond(M))
        return c if np.isfinite(c) else None
    except Exception:
        return None


def spectral_diagnostics(M: np.ndarray) -> Dict[str, object]:
    vals, _ = eigendecompose_raw(M)
    vals_r = np.real(vals)
    return {
        "shape": list(M.shape),
        "is_symmetric": bool(np.allclose(M, M.T, atol=1e-8, rtol=0.0)),
        "eig_min_real": float(np.min(vals_r)) if vals_r.size else None,
        "eig_max_real": float(np.max(vals_r)) if vals_r.size else None,
        "eig_negative_count_real": int(np.sum(vals_r < 0)),
        "eig_near_zero_count_abs_lt_1e-12": int(np.sum(np.abs(vals_r) < 1e-12)),
        "condition_number": _safe_cond(M),
    }


def _offdiag_values(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    return np.asarray(M)[~np.eye(n, dtype=bool)]


def shared_mode_suite(C: np.ndarray) -> Dict[str, Optional[float]]:
    vals, _ = eigendecompose_raw(np.nan_to_num(C, nan=0.0))
    vals_r = np.real(vals)
    lam1 = float(vals_r[0]) if vals_r.size >= 1 else None
    lam2 = float(vals_r[1]) if vals_r.size >= 2 else None
    off = _offdiag_values(C)
    off = off[np.isfinite(off)]
    return {
        "lambda1_C": lam1,
        "lambda1_minus_lambda2_C": (lam1 - lam2) if (lam1 is not None and lam2 is not None) else None,
        "mean_offdiag_C": float(np.mean(off)) if off.size else None,
        "fro_offdiag_C": float(np.linalg.norm(off)) if off.size else None,
        "trace_C": float(np.trace(np.nan_to_num(C, nan=0.0))),
    }


def save_matrix_bundle(
    output_dir: str,
    tag: str,
    K: np.ndarray,
    object_names: Sequence[str],
    metadata: Optional[Dict[str, object]] = None,
    C_override: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    ensure_dir(output_dir)
    K = np.asarray(K)
    C = np.asarray(C_override) if C_override is not None else covariance_to_correlation(K)
    c_source = "provided_existing_corr" if C_override is not None else "derived_from_T"

    eig_K_vals, eig_K_vecs = eigendecompose_raw(K)
    eig_C_vals, eig_C_vecs = eigendecompose_raw(np.nan_to_num(C, nan=0.0))

    paths = {
        "T_npy": os.path.join(output_dir, f"T_{tag}.npy"),
        "T_csv": os.path.join(output_dir, f"T_{tag}.csv"),
        "C_npy": os.path.join(output_dir, f"C_{tag}.npy"),
        "C_csv": os.path.join(output_dir, f"C_{tag}.csv"),
        "eigvals_T_npy": os.path.join(output_dir, f"eigvals_T_{tag}.npy"),
        "eigvecs_T_npy": os.path.join(output_dir, f"eigvecs_T_{tag}.npy"),
        "eigvals_C_npy": os.path.join(output_dir, f"eigvals_C_{tag}.npy"),
        "eigvecs_C_npy": os.path.join(output_dir, f"eigvecs_C_{tag}.npy"),
        "shared_mode_suite_json": os.path.join(output_dir, f"shared_mode_suite_{tag}.json"),
    }
    summary_json = os.path.join(output_dir, f"summary_{tag}.json")

    np.save(paths["T_npy"], K)
    np.savetxt(paths["T_csv"], K, delimiter=",", fmt="%.18e")
    np.save(paths["C_npy"], C)
    np.savetxt(paths["C_csv"], C, delimiter=",", fmt="%.18e")
    np.save(paths["eigvals_T_npy"], eig_K_vals)
    np.save(paths["eigvecs_T_npy"], eig_K_vecs)
    np.save(paths["eigvals_C_npy"], eig_C_vals)
    np.save(paths["eigvecs_C_npy"], eig_C_vecs)

    suite = shared_mode_suite(C)
    with open(paths["shared_mode_suite_json"], "w", encoding="utf-8") as f:
        json.dump(suite, f, ensure_ascii=False, indent=2)

    summary: Dict[str, object] = {
        "tag": tag,
        "object_names": list(object_names),
        "c_source": c_source,
        "paths": {k: os.path.abspath(v) for k, v in paths.items()},
        "spectral_T": spectral_diagnostics(K),
        "spectral_C": spectral_diagnostics(np.nan_to_num(C, nan=0.0)),
        "shared_mode_suite": suite,
    }
    if metadata:
        summary["metadata"] = metadata
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    out = {k: os.path.abspath(v) for k, v in paths.items()}
    out["summary_json"] = os.path.abspath(summary_json)
    return out


def recover_coordinates_from_gram(K: np.ndarray) -> Dict[str, np.ndarray]:
    vals, vecs = eigendecompose_raw(K)
    sqrt_vals = np.lib.scimath.sqrt(vals)
    Z_hat = np.diag(sqrt_vals) @ vecs.T
    return {"eigvals": vals, "eigvecs": vecs, "sqrt_eigvals": sqrt_vals, "Z_hat": Z_hat}


def save_coordinate_bundle(
    output_dir: str,
    tag: str,
    K: np.ndarray,
    object_names: Sequence[str],
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, str]:
    ensure_dir(output_dir)
    rec = recover_coordinates_from_gram(K)
    z_npy = os.path.join(output_dir, f"Zhat_{tag}.npy")
    eigvals_npy = os.path.join(output_dir, f"Zhat_eigvals_{tag}.npy")
    eigvecs_npy = os.path.join(output_dir, f"Zhat_eigvecs_{tag}.npy")
    summary_json = os.path.join(output_dir, f"Zhat_summary_{tag}.json")

    np.save(z_npy, rec["Z_hat"])
    np.save(eigvals_npy, rec["eigvals"])
    np.save(eigvecs_npy, rec["eigvecs"])

    summary: Dict[str, object] = {
        "tag": tag,
        "note": (
            "Z_hat is a Gram-recovered coordinate realization (basis is not unique). "
            "Use only geometric invariants; do not interpret as original parameter coordinates."
        ),
        "object_names": list(object_names),
        "paths": {
            "Zhat_npy": os.path.abspath(z_npy),
            "eigvals_npy": os.path.abspath(eigvals_npy),
            "eigvecs_npy": os.path.abspath(eigvecs_npy),
        },
        "spectral_K": spectral_diagnostics(np.asarray(K)),
    }
    if metadata:
        summary["metadata"] = metadata
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "Zhat_npy": os.path.abspath(z_npy),
        "eigvals_npy": os.path.abspath(eigvals_npy),
        "eigvecs_npy": os.path.abspath(eigvecs_npy),
        "summary_json": os.path.abspath(summary_json),
    }

def _center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n), dtype=K.dtype) / float(n)
    return H @ K @ H


def linear_hsic_from_gram(Kx: np.ndarray, Ky: np.ndarray, centered: bool = True) -> float:
    Kx = np.asarray(Kx, dtype=np.float64)
    Ky = np.asarray(Ky, dtype=np.float64)
    if Kx.shape != Ky.shape:
        raise ValueError(f"shape mismatch: {Kx.shape} vs {Ky.shape}")
    if centered:
        Kx = _center_gram(Kx)
        Ky = _center_gram(Ky)
    n = Kx.shape[0]
    return float(np.trace(Kx @ Ky) / max(1.0, float((n - 1) ** 2)))


def linear_cka_from_gram(Kx: np.ndarray, Ky: np.ndarray) -> float:
    Kxc = _center_gram(np.asarray(Kx, dtype=np.float64))
    Kyc = _center_gram(np.asarray(Ky, dtype=np.float64))
    num = np.sum(Kxc * Kyc)
    den = math.sqrt(float(np.sum(Kxc * Kxc)) * float(np.sum(Kyc * Kyc)))
    if den == 0:
        return float("nan")
    return float(num / den)


def distance_matrix_from_gram(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    d = np.diag(K)
    return d[:, None] + d[None, :] - 2.0 * K


def gaussian_kernel_from_distance(D: np.ndarray, sigma: Optional[float] = None) -> Tuple[np.ndarray, float]:
    D = np.asarray(D, dtype=np.float64)
    if sigma is None or sigma <= 0:
        tri = D[np.triu_indices(D.shape[0], k=1)]
        tri = tri[np.isfinite(tri)]
        tri = tri[tri > 0]
        if tri.size == 0:
            sigma = 1.0
        else:
            sigma = float(np.sqrt(np.median(tri)))
            if sigma <= 0 or (not np.isfinite(sigma)):
                sigma = 1.0
    G = np.exp(-D / (2.0 * sigma * sigma))
    return G, float(sigma)


def gaussian_hsic_from_gram(
    Kx: np.ndarray,
    Ky: np.ndarray,
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None,
) -> Dict[str, float]:
    Dx = distance_matrix_from_gram(Kx)
    Dy = distance_matrix_from_gram(Ky)
    Gx, sx = gaussian_kernel_from_distance(Dx, sigma_x)
    Gy, sy = gaussian_kernel_from_distance(Dy, sigma_y)
    return {"gaussian_hsic": linear_hsic_from_gram(Gx, Gy, centered=True), "sigma_x": sx, "sigma_y": sy}


def run_calc_dataset_similarity_pair(
    python_exe: str,
    calc_script: str,
    base_model_path: str,
    train_dataset_path: str,
    grad1_path: str,
    grad2_path: str,
    out_path: str,
    lora_path: Optional[str] = None,
    damping: Optional[float] = None,
    env_overrides: Optional[Dict[str, str]] = None,
    timeout_sec: Optional[int] = None,
) -> Tuple[bool, str]:
    cmd = [
        python_exe,
        calc_script,
        "--base_model_path",
        base_model_path,
        "--train_dataset_path",
        train_dataset_path,
        "--grad1_path",
        grad1_path,
        "--grad2_path",
        grad2_path,
        "--out_path",
        out_path,
    ]
    if lora_path:
        cmd.extend(["--lora_path", lora_path])
    if damping is not None:
        cmd.extend(["--damping", str(damping)])
    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_sec)
        ok = proc.returncode == 0
        msg = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return ok, msg.strip()
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        err = (e.stderr or "") if isinstance(e.stderr, str) else ""
        msg = f"timeout after {timeout_sec}s\n{out}\n{err}".strip()
        return False, msg


@dataclass
class PairwiseRunResult:
    matrix: Optional[np.ndarray]
    pairwise_dir: str
    missing_grad_paths: List[str]
    failed_pairs: List[Dict[str, str]]
    submitted_pairs: int
    available_pairs: int
    total_pairs: int
    is_complete: bool


def _pair_score_from_candidates(pairwise_dir: str, names: Sequence[str], i: int, j: int) -> Optional[float]:
    candidates = [
        os.path.join(pairwise_dir, f"sim_{names[i]}_{names[j]}.json"),
        os.path.join(pairwise_dir, f"sim_{names[j]}_{names[i]}.json"),
        os.path.join(pairwise_dir, f"sim_{i}_{j}.json"),
        os.path.join(pairwise_dir, f"sim_{j}_{i}.json"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            score = _load_json_score(p)
            if score is not None:
                return score
    return None


def count_available_pairwise_scores(pairwise_dir: str, names: Sequence[str]) -> Tuple[int, int]:
    n = len(names)
    total = n * (n + 1) // 2
    if not os.path.isdir(pairwise_dir):
        return 0, total
    available = 0
    for i in range(n):
        for j in range(i, n):
            if _pair_score_from_candidates(pairwise_dir, names, i, j) is not None:
                available += 1
    return available, total


def compute_pairwise_scores_via_cli(
    datainf_root: str,
    output_dir: str,
    base_model_path: str,
    train_dataset_path: str,
    grad_paths: Dict[str, str],
    dataset_names: Sequence[str],
    lora_path: Optional[str] = None,
    damping: Optional[float] = None,
    python_exe: Optional[str] = None,
    max_workers: Optional[int] = None,
    gpu_ids: Optional[Sequence[str]] = None,
    pair_timeout_sec: Optional[int] = None,
    pair_shard_count: int = 1,
    pair_shard_index: int = 0,
    run_missing_pairs: bool = True,
) -> PairwiseRunResult:
    ensure_dir(output_dir)
    pairwise_dir = ensure_dir(os.path.join(output_dir, "pairwise_result"))
    python_exe = python_exe or sys.executable
    calc_script = os.path.join(datainf_root, "src", "calc_dataset_similarity.py")

    missing: List[str] = []
    for n in dataset_names:
        p = grad_paths.get(n)
        if not p or (not os.path.isfile(p)):
            missing.append(p or f"(missing mapping for {n})")

    all_pair_jobs: List[Tuple[str, str, str]] = []
    for i, a in enumerate(dataset_names):
        for j in range(i, len(dataset_names)):
            b = dataset_names[j]
            out_json = os.path.join(pairwise_dir, f"sim_{a}_{b}.json")
            if os.path.isfile(out_json):
                continue
            all_pair_jobs.append((a, b, out_json))

    shard_count = max(1, int(pair_shard_count))
    shard_index = int(pair_shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"pair_shard_index must be in [0, {shard_count - 1}], got {shard_index}")

    if shard_count > 1:
        pair_jobs = [job for idx, job in enumerate(all_pair_jobs) if (idx % shard_count) == shard_index]
    else:
        pair_jobs = list(all_pair_jobs)

    gpu_list: List[str] = []
    if gpu_ids:
        gpu_list = [str(x).strip() for x in gpu_ids if str(x).strip()]
    if not gpu_list:
        env_gpu = os.environ.get("SCHEMEA_GPU_IDS", "").strip()
        if not env_gpu:
            env_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if env_gpu:
            gpu_list = [x.strip() for x in env_gpu.split(",") if x.strip()]

    if max_workers is None or max_workers <= 0:
        max_workers_eff = len(gpu_list) if gpu_list else 1
    else:
        max_workers_eff = max_workers
    max_workers_eff = max(1, max_workers_eff)

    failed_pairs: List[Dict[str, str]] = []
    if run_missing_pairs and (not missing) and pair_jobs:
        if max_workers_eff <= 1:
            for idx, (a, b, out_json) in enumerate(pair_jobs):
                env_overrides = {"TOKENIZERS_PARALLELISM": "false"}
                if gpu_list:
                    env_overrides["CUDA_VISIBLE_DEVICES"] = gpu_list[idx % len(gpu_list)]
                ok, msg = run_calc_dataset_similarity_pair(
                    python_exe=python_exe,
                    calc_script=calc_script,
                    base_model_path=base_model_path,
                    train_dataset_path=train_dataset_path,
                    grad1_path=grad_paths[a],
                    grad2_path=grad_paths[b],
                    out_path=out_json,
                    lora_path=lora_path,
                    damping=damping,
                    env_overrides=env_overrides,
                    timeout_sec=pair_timeout_sec,
                )
                if not ok:
                    failed_pairs.append({"pair": f"{a}|{b}", "message": msg[:2000]})
        else:
            with ThreadPoolExecutor(max_workers=max_workers_eff) as ex:
                fut_map = {}
                for idx, (a, b, out_json) in enumerate(pair_jobs):
                    env_overrides = {"TOKENIZERS_PARALLELISM": "false"}
                    if gpu_list:
                        env_overrides["CUDA_VISIBLE_DEVICES"] = gpu_list[idx % len(gpu_list)]
                    fut = ex.submit(
                        run_calc_dataset_similarity_pair,
                        python_exe,
                        calc_script,
                        base_model_path,
                        train_dataset_path,
                        grad_paths[a],
                        grad_paths[b],
                        out_json,
                        lora_path,
                        damping,
                        env_overrides,
                        pair_timeout_sec,
                    )
                    fut_map[fut] = (a, b)
                for fut in as_completed(fut_map):
                    a, b = fut_map[fut]
                    try:
                        ok, msg = fut.result()
                    except Exception as e:
                        ok, msg = False, str(e)
                    if not ok:
                        failed_pairs.append({"pair": f"{a}|{b}", "message": str(msg)[:2000]})

    K = matrix_from_pairwise_json(pairwise_dir, dataset_names)
    available_pairs, total_pairs = count_available_pairwise_scores(pairwise_dir, dataset_names)
    return PairwiseRunResult(
        matrix=K,
        pairwise_dir=pairwise_dir,
        missing_grad_paths=missing,
        failed_pairs=failed_pairs,
        submitted_pairs=len(pair_jobs),
        available_pairs=available_pairs,
        total_pairs=total_pairs,
        is_complete=(available_pairs >= total_pairs),
    )


def write_unavailable_note(path: str, reason: str, context: Optional[Dict[str, object]] = None) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    obj: Dict[str, object] = {"status": "unavailable", "reason": reason}
    if context:
        obj["context"] = context
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def parse_metrics_from_log_file(path: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not os.path.isfile(path):
        return metrics
    current_ctx: Optional[str] = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            low = line.lower()
            if not line:
                continue
            if "evaluation on" in low:
                if "gsm8k" in low:
                    current_ctx = "gsm8k"
                elif "multiarith" in low:
                    current_ctx = "multiarith"
                elif "openfunction" in low:
                    current_ctx = "openfunction"
                elif "humaneval" in low:
                    current_ctx = "humaneval"
                elif "alpaca" in low:
                    current_ctx = "alpaca_eval"
                elif "safety" in low:
                    current_ctx = "safety"
                else:
                    current_ctx = None
                continue
            nums = _FLOAT_RE.findall(line)
            if not nums:
                continue
            val = float(nums[-1])
            if current_ctx:
                metrics[f"metric_{current_ctx}"] = val
            if "accuracy" in low:
                metrics["metric_accuracy_generic"] = val
            if "pass@" in low:
                metrics["metric_humaneval_passk"] = val
    return metrics


def discover_performance_rows(
    sdft_root: str,
    models: Sequence[str],
    methods: Sequence[str],
    epochs: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model in models:
        for method in methods:
            for epoch in epochs:
                logp = os.path.join(sdft_root, "epoch1_results", model, f"{method}.log") if epoch == "epoch_1" else os.path.join(sdft_root, "results", model, f"{method}.log")
                row: Dict[str, object] = {
                    "train_dataset": model,
                    "model": model,
                    "epoch": epoch,
                    "method": method,
                    "perf_log_path": logp,
                    "perf_log_exists": os.path.isfile(logp),
                }
                row.update(parse_metrics_from_log_file(logp))
                rows.append(row)
    return rows


def write_rows_csv(path: str, rows: Sequence[Dict[str, object]]) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return path
    cols: List[str] = []
    col_set = set()
    for r in rows:
        for k in r.keys():
            if k not in col_set:
                col_set.add(k)
                cols.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def write_rows_txt(path: str, rows: Sequence[Dict[str, object]], max_cols: int = 16) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("no rows\n")
        return path
    cols: List[str] = []
    col_set = set()
    for r in rows:
        for k in r.keys():
            if k not in col_set:
                col_set.add(k)
                cols.append(k)
    cols = cols[:max_cols]
    widths: Dict[str, int] = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}

    with open(path, "w", encoding="utf-8") as f:
        f.write(" | ".join(c.ljust(widths[c]) for c in cols) + "\n")
        f.write("-+-".join("-" * widths[c] for c in cols) + "\n")
        for r in rows:
            f.write(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols) + "\n")
    return path
