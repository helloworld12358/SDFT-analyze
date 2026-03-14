#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scheme A - Step 06
Raw/Rewrite 样本级 Gram-world 分解。

默认第一轮配置：
- 训练集：7个全跑
- epoch：epoch_5
- feature_method：sdft
- oracle_mode：mixed，若不可用自动回退 own_sdft
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAINF_ROOT_DEFAULT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(DATAINF_ROOT_DEFAULT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gram_scheme_a_utils import (  # type: ignore  # noqa: E402
    DEFAULT_EPOCHS,
    DEFAULT_TRAIN_DATASETS,
    compute_pairwise_scores_via_cli,
    detect_datainf_root,
    ensure_dir,
    load_records_any,
    merge_records_for_mixed_h,
    normalize_epoch_list,
    normalize_method_list,
    recover_coordinates_from_gram,
    resolve_checkpoint_path,
    resolve_result_root,
    resolve_sdft_root,
    resolve_train_dataset_path,
    save_matrix_bundle,
    save_coordinate_bundle,
    spectral_diagnostics,
    split_csv_arg,
    write_rows_csv,
    write_rows_txt,
    write_unavailable_note,
)

ID_KEYS = ["id", "sample_id", "idx", "index", "question_id", "task_id", "uuid"]


def _save_one_record_json(path: str, rec: dict) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump([rec], f, ensure_ascii=False, indent=2)
    return path


def _run_save_avg_grad(
    python_exe: str,
    script_path: str,
    base_model_path: str,
    dataset_path: str,
    output_path: str,
    lora_path: Optional[str],
    batch_size: int = 1,
) -> Tuple[bool, str]:
    cmd = [
        python_exe,
        script_path,
        "--base_model_path",
        base_model_path,
        "--dataset_path",
        dataset_path,
        "--output_path",
        output_path,
        "--batch_size",
        str(batch_size),
        "--max_samples",
        "1",
    ]
    if lora_path:
        cmd.extend(["--lora_path", lora_path])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0
    msg = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return ok, msg.strip()


def _oracle_config(
    oracle_mode: str,
    sdft_root: str,
    train_dataset: str,
    epoch: str,
    mixed_cache_path: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if oracle_mode == "mixed":
        sft_train = resolve_train_dataset_path(sdft_root, train_dataset, "sft")
        sdft_dist = resolve_train_dataset_path(sdft_root, train_dataset, "sdft")
        mixed_path, err = merge_records_for_mixed_h(sft_train, sdft_dist, mixed_cache_path)
        if err:
            return None, None, err
        return mixed_path, None, None
    if oracle_mode == "own_sft":
        train_path = resolve_train_dataset_path(sdft_root, train_dataset, "sft")
        lora = resolve_checkpoint_path(sdft_root, epoch, train_dataset, "sft")
        if not os.path.isfile(train_path):
            return None, None, f"missing train dataset for own_sft: {train_path}"
        return train_path, lora, None
    if oracle_mode == "own_sdft":
        train_path = resolve_train_dataset_path(sdft_root, train_dataset, "sdft")
        lora = resolve_checkpoint_path(sdft_root, epoch, train_dataset, "sdft")
        if not os.path.isfile(train_path):
            return None, None, f"missing train dataset for own_sdft: {train_path}"
        return train_path, lora, None
    return None, None, f"unsupported oracle_mode: {oracle_mode}"


def _orthonormal_basis(X: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    if X.size == 0:
        return np.zeros((X.shape[0], 0), dtype=X.dtype)
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    r = int(np.sum(s > tol))
    return U[:, :r]


def _stable_rank_from_eigs(eigs: np.ndarray) -> float:
    vals = np.abs(np.asarray(eigs))
    denom = float(np.sum(vals * vals))
    if denom == 0.0:
        return float("nan")
    return float((np.sum(vals) ** 2) / denom)


def _cov_stats(X_cols: np.ndarray) -> Dict[str, object]:
    if X_cols.size == 0 or X_cols.shape[1] == 0:
        return {"trace": None, "top_eig_real": None, "stable_rank_abs_eigs": None, "eigvals": []}
    n = X_cols.shape[1]
    S = (X_cols @ X_cols.conj().T) / float(n)
    eigs = np.linalg.eigvals(S)
    return {
        "trace": float(np.real(np.trace(S))),
        "top_eig_real": float(np.max(np.real(eigs))) if eigs.size else None,
        "stable_rank_abs_eigs": _stable_rank_from_eigs(eigs),
        "eigvals": [float(np.real(x)) if abs(np.imag(x)) < 1e-12 else {"re": float(np.real(x)), "im": float(np.imag(x))} for x in eigs],
    }


def _build_pairs(raw_records: List[dict], rw_records: List[dict]) -> Tuple[List[Tuple[int, int]], str, Optional[str]]:
    # 先尝试显式 id 映射
    for key in ID_KEYS:
        raw_map: Dict[object, int] = {}
        rw_map: Dict[object, int] = {}
        for i, r in enumerate(raw_records):
            if key in r:
                raw_map[r[key]] = i
        for i, r in enumerate(rw_records):
            if key in r:
                rw_map[r[key]] = i
        overlap = sorted(set(raw_map.keys()) & set(rw_map.keys()))
        if len(overlap) >= 4:
            return [(raw_map[k], rw_map[k]) for k in overlap], "id_mapping", key

    # 回退为同索引配对（与 gen_distilled_data.py 的 zip 逻辑一致）
    n = min(len(raw_records), len(rw_records))
    return [(i, i) for i in range(n)], "same_index", None


def _default_fallback_oracle(feature_method: str) -> str:
    return "own_sdft" if feature_method == "sdft" else "own_sft"


def run_one_combo(
    datainf_root: str,
    sdft_root: str,
    output_root: str,
    train_dataset: str,
    epoch: str,
    feature_method: str,
    oracle_mode_requested: str,
    fallback_when_mixed_unavailable: bool,
    sample_size: int,
    seed: int,
    batch_size: int,
    base_model_path: str,
    damping: float,
    python_exe: str,
    k_angles: int,
    max_workers: Optional[int],
    gpu_ids: List[str],
    pair_timeout_sec: Optional[int],
) -> Dict[str, object]:
    run_dir = os.path.join(
        output_root,
        train_dataset,
        epoch,
        f"feature_{feature_method}",
        f"oracle_{oracle_mode_requested}",
    )
    ensure_dir(run_dir)

    raw_path = resolve_train_dataset_path(sdft_root, train_dataset, "sft")
    rw_path = resolve_train_dataset_path(sdft_root, train_dataset, "sdft")
    raw_records = load_records_any(raw_path)
    rw_records = load_records_any(rw_path)
    if not raw_records or not rw_records:
        reason = write_unavailable_note(
            os.path.join(run_dir, "unavailable_raw_rewrite.json"),
            reason="raw/rewrite datasets are missing or empty",
            context={"raw_path": raw_path, "rewrite_path": rw_path},
        )
        return {"status": "unavailable", "reason_file": os.path.abspath(reason), "train_dataset": train_dataset, "epoch": epoch, "feature_method": feature_method}

    pairs, mapping_mode, mapping_key = _build_pairs(raw_records, rw_records)
    if not pairs:
        reason = write_unavailable_note(
            os.path.join(run_dir, "unavailable_raw_rewrite.json"),
            reason="no pair can be built",
            context={"raw_path": raw_path, "rewrite_path": rw_path},
        )
        return {"status": "unavailable", "reason_file": os.path.abspath(reason), "train_dataset": train_dataset, "epoch": epoch, "feature_method": feature_method}

    rng = np.random.default_rng(seed)
    take = min(sample_size, len(pairs))
    sampled_ids = sorted(rng.choice(np.arange(len(pairs)), size=take, replace=False).tolist())
    sampled_pairs = [pairs[i] for i in sampled_ids]

    with open(os.path.join(run_dir, "sample_pairs.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "sample_size_used": take,
                "mapping_mode": mapping_mode,
                "mapping_key": mapping_key,
                "pair_indices": sampled_pairs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    lora_feature = resolve_checkpoint_path(sdft_root, epoch, train_dataset, feature_method)
    grad_builder_script = os.path.join(datainf_root, "src", "save_avg_grad_with_integrated_templates.py")
    cache_grad_dir = ensure_dir(os.path.join(run_dir, "_cache_object_grads"))
    cache_data_dir = ensure_dir(os.path.join(run_dir, "_cache_object_data"))

    object_names: List[str] = []
    grad_paths: Dict[str, str] = {}
    grad_failures: List[Dict[str, str]] = []

    for raw_idx, rw_idx in sampled_pairs:
        pair_tag = f"r{raw_idx}_w{rw_idx}"
        for kind, rec in (("raw", raw_records[raw_idx]), ("rw", rw_records[rw_idx])):
            obj_name = f"{kind}_{pair_tag}"
            object_names.append(obj_name)
            one_json = os.path.join(cache_data_dir, f"{obj_name}.json")
            out_pt = os.path.join(cache_grad_dir, f"{obj_name}.pt")
            if not os.path.isfile(one_json):
                _save_one_record_json(one_json, rec)
            if not os.path.isfile(out_pt):
                ok, msg = _run_save_avg_grad(
                    python_exe=python_exe,
                    script_path=grad_builder_script,
                    base_model_path=base_model_path,
                    dataset_path=one_json,
                    output_path=out_pt,
                    lora_path=lora_feature,
                    batch_size=batch_size,
                )
                if not ok:
                    grad_failures.append({"object": obj_name, "message": msg[:2000]})
                    continue
            grad_paths[obj_name] = out_pt

    if grad_failures:
        with open(os.path.join(run_dir, "grad_build_failures.json"), "w", encoding="utf-8") as f:
            json.dump(grad_failures, f, ensure_ascii=False, indent=2)

    ordered_names = [n for n in object_names if n in grad_paths]
    if len(ordered_names) < 2:
        reason = write_unavailable_note(
            os.path.join(run_dir, "unavailable_raw_rewrite.json"),
            reason="insufficient per-object gradients after build/cache stage",
            context={"grad_failures": grad_failures},
        )
        return {"status": "unavailable", "reason_file": os.path.abspath(reason), "train_dataset": train_dataset, "epoch": epoch, "feature_method": feature_method}

    mixed_cache = os.path.join(run_dir, "_cache_mixed_train.json")
    oracle_mode_used = oracle_mode_requested
    train_data_path, oracle_lora_path, oracle_err = _oracle_config(
        oracle_mode=oracle_mode_used,
        sdft_root=sdft_root,
        train_dataset=train_dataset,
        epoch=epoch,
        mixed_cache_path=mixed_cache,
    )

    fallback_used = False
    if oracle_err and oracle_mode_requested == "mixed" and fallback_when_mixed_unavailable:
        oracle_mode_used = _default_fallback_oracle(feature_method)
        train_data_path, oracle_lora_path, oracle_err = _oracle_config(
            oracle_mode=oracle_mode_used,
            sdft_root=sdft_root,
            train_dataset=train_dataset,
            epoch=epoch,
            mixed_cache_path=mixed_cache,
        )
        fallback_used = True

    if oracle_err or (not train_data_path) or (not os.path.isfile(train_data_path)):
        reason = write_unavailable_note(
            os.path.join(run_dir, "unavailable_raw_rewrite.json"),
            reason=f"oracle config unavailable: {oracle_err or 'missing train_data_path'}",
            context={
                "oracle_mode_requested": oracle_mode_requested,
                "oracle_mode_used": oracle_mode_used,
                "fallback_used": fallback_used,
                "train_data_path": train_data_path,
                "oracle_lora_path": oracle_lora_path,
            },
        )
        return {"status": "unavailable", "reason_file": os.path.abspath(reason), "train_dataset": train_dataset, "epoch": epoch, "feature_method": feature_method}

    pairwise = compute_pairwise_scores_via_cli(
        datainf_root=datainf_root,
        output_dir=run_dir,
        base_model_path=base_model_path,
        train_dataset_path=train_data_path,
        grad_paths=grad_paths,
        dataset_names=ordered_names,
        lora_path=oracle_lora_path,
        damping=damping,
        python_exe=python_exe,
        max_workers=max_workers,
        gpu_ids=gpu_ids,
        pair_timeout_sec=pair_timeout_sec,
    )
    if pairwise.matrix is None:
        reason = write_unavailable_note(
            os.path.join(run_dir, "unavailable_raw_rewrite.json"),
            reason="pairwise oracle failed to produce Gram matrix",
            context={"failed_pairs": pairwise.failed_pairs, "pairwise_dir": pairwise.pairwise_dir},
        )
        return {"status": "failed", "reason_file": os.path.abspath(reason), "train_dataset": train_dataset, "epoch": epoch, "feature_method": feature_method}

    K = pairwise.matrix
    tag = f"{train_dataset}_{epoch}_{feature_method}_{oracle_mode_requested}_raw_rewrite"
    matrix_bundle = save_matrix_bundle(
        output_dir=run_dir,
        tag=tag,
        K=K,
        object_names=ordered_names,
        metadata={
            "mode": "raw_rewrite",
            "train_dataset": train_dataset,
            "epoch": epoch,
            "feature_method": feature_method,
            "oracle_mode_requested": oracle_mode_requested,
            "oracle_mode_used": oracle_mode_used,
            "fallback_used": fallback_used,
            "pair_mapping_mode": mapping_mode,
            "pair_mapping_key": mapping_key,
            "train_data_path": train_data_path,
            "oracle_lora_path": oracle_lora_path,
        },
    )
    coord_bundle = save_coordinate_bundle(
        output_dir=run_dir,
        tag=tag,
        K=K,
        object_names=ordered_names,
        metadata={"source": matrix_bundle["summary_json"]},
    )

    rec = recover_coordinates_from_gram(K)
    Z = rec["Z_hat"]
    name_to_idx = {n: i for i, n in enumerate(ordered_names)}

    m_list: List[np.ndarray] = []
    d_list: List[np.ndarray] = []
    pair_rows: List[Dict[str, object]] = []

    for raw_idx, rw_idx in sampled_pairs:
        pair_tag = f"r{raw_idx}_w{rw_idx}"
        rn = f"raw_{pair_tag}"
        wn = f"rw_{pair_tag}"
        if rn not in name_to_idx or wn not in name_to_idx:
            continue
        z_raw = Z[:, name_to_idx[rn]]
        z_rw = Z[:, name_to_idx[wn]]
        m = (z_raw + z_rw) / 2.0
        d = (z_raw - z_rw) / 2.0
        m_list.append(m)
        d_list.append(d)

        denom = np.linalg.norm(m) * np.linalg.norm(d)
        ai = float(abs(np.vdot(m, d)) / denom) if denom != 0 else float("nan")
        pair_rows.append(
            {
                "raw_idx": raw_idx,
                "rw_idx": rw_idx,
                "raw_name": rn,
                "rw_name": wn,
                "a_i_orthogonality": ai,
                "norm_m": float(np.linalg.norm(m)),
                "norm_d": float(np.linalg.norm(d)),
            }
        )

    if not m_list or not d_list:
        reason = write_unavailable_note(
            os.path.join(run_dir, "unavailable_raw_rewrite.json"),
            reason="no valid raw/rewrite pairs after gradient filtering",
            context={"ordered_names": ordered_names},
        )
        return {"status": "unavailable", "reason_file": os.path.abspath(reason), "train_dataset": train_dataset, "epoch": epoch, "feature_method": feature_method}

    M_cols = np.stack(m_list, axis=1)
    D_cols = np.stack(d_list, axis=1)

    num = np.linalg.norm(M_cols.conj().T @ D_cols, ord="fro")
    den = np.linalg.norm(M_cols, ord="fro") * np.linalg.norm(D_cols, ord="fro")
    A_leak = float(num / den) if den != 0 else float("nan")

    Qm = _orthonormal_basis(M_cols)
    Qd = _orthonormal_basis(D_cols)
    if Qm.shape[1] == 0 or Qd.shape[1] == 0:
        principal_angles_deg: List[float] = []
    else:
        svals = np.linalg.svd(Qm.conj().T @ Qd, compute_uv=False)
        svals = np.clip(np.real(svals), -1.0, 1.0)
        k = min(k_angles, len(svals))
        principal_angles_deg = [float(np.degrees(np.arccos(v))) for v in svals[:k]]

    style_stats = _cov_stats(D_cols)
    content_stats = _cov_stats(M_cols)

    pair_csv = os.path.join(run_dir, f"pair_metrics_{tag}.csv")
    write_rows_csv(pair_csv, pair_rows)

    summary = {
        "train_dataset": train_dataset,
        "epoch": epoch,
        "feature_method": feature_method,
        "oracle_mode_requested": oracle_mode_requested,
        "oracle_mode_used": oracle_mode_used,
        "fallback_used": fallback_used,
        "pair_mapping_mode": mapping_mode,
        "pair_mapping_key": mapping_key,
        "seed": seed,
        "sample_size_requested": sample_size,
        "sample_size_used": len(pair_rows),
        "A_leak": A_leak,
        "principal_angles_deg": principal_angles_deg,
        "style_covariance": style_stats,
        "content_covariance": content_stats,
        "paths": {
            "matrix_summary": matrix_bundle["summary_json"],
            "coords_summary": coord_bundle["summary_json"],
            "pair_metrics_csv": os.path.abspath(pair_csv),
            "pairwise_dir": pairwise.pairwise_dir,
        },
        "spectral_K": spectral_diagnostics(K),
        "notes": [
            "All computations are Gram-world only.",
            "Z_hat is one coordinate realization (not unique).",
            "No negative-eigenvalue clipping or spectrum cleaning is applied.",
        ],
    }
    summary_json = os.path.join(run_dir, f"raw_rewrite_summary_{tag}.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "status": "ok",
        "train_dataset": train_dataset,
        "epoch": epoch,
        "feature_method": feature_method,
        "oracle_mode_requested": oracle_mode_requested,
        "oracle_mode_used": oracle_mode_used,
        "fallback_used": fallback_used,
        "A_leak": A_leak,
        "style_trace": style_stats.get("trace"),
        "style_top_eig": style_stats.get("top_eig_real"),
        "summary_json": os.path.abspath(summary_json),
        "pair_metrics_csv": os.path.abspath(pair_csv),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Scheme A raw/rewrite decomposition")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--train_dataset", type=str, default="")
    p.add_argument("--all_train_datasets", action="store_true")
    p.add_argument("--epoch", type=str, default="epoch_5")
    p.add_argument("--all_epochs", action="store_true")
    p.add_argument("--feature_method", type=str, default="sdft", help="sft/sdft/both")
    p.add_argument("--oracle_mode", type=str, default="mixed", choices=["mixed", "own_sft", "own_sdft"])
    p.add_argument("--no_fallback_when_mixed_unavailable", action="store_true")
    p.add_argument("--sample_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--base_model_path", type=str, default=None)
    p.add_argument("--damping", type=float, default=0.001)
    p.add_argument("--python_exe", type=str, default=sys.executable)
    p.add_argument("--num_workers", type=int, default=0, help="<=0 表示自动按 GPU 数量并行")
    p.add_argument("--gpu_ids", type=str, default="", help="逗号分隔，如 0,1,2,3")
    p.add_argument("--pair_timeout_sec", type=int, default=0, help="单个 pair 超时秒数，<=0 表示不限")
    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--k_angles", type=int, default=8)
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    sdft_root = resolve_sdft_root(datainf_root)
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    output_root = args.output_root or os.path.join(result_root, "schemeA", "raw_rewrite")

    train_datasets = split_csv_arg(args.train_dataset, DEFAULT_TRAIN_DATASETS) if (args.train_dataset and not args.all_train_datasets) else list(DEFAULT_TRAIN_DATASETS)
    epochs = list(DEFAULT_EPOCHS) if args.all_epochs else normalize_epoch_list(split_csv_arg(args.epoch, ["epoch_5"]))
    feature_methods = normalize_method_list(split_csv_arg(args.feature_method, ["sdft"])) or ["sdft"]
    base_model_path = args.base_model_path or os.path.join(sdft_root, "model", "Llama-2-7b-chat-hf")
    fallback_when_mixed_unavailable = not args.no_fallback_when_mixed_unavailable
    gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    max_workers = None if args.num_workers <= 0 else args.num_workers
    pair_timeout_sec = None if args.pair_timeout_sec <= 0 else args.pair_timeout_sec

    rows: List[Dict[str, object]] = []
    for train_dataset in train_datasets:
        for epoch in epochs:
            for feature_method in feature_methods:
                row = run_one_combo(
                    datainf_root=datainf_root,
                    sdft_root=sdft_root,
                    output_root=output_root,
                    train_dataset=train_dataset,
                    epoch=epoch,
                    feature_method=feature_method,
                    oracle_mode_requested=args.oracle_mode,
                    fallback_when_mixed_unavailable=fallback_when_mixed_unavailable,
                    sample_size=args.sample_size,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    base_model_path=base_model_path,
                    damping=args.damping,
                    python_exe=args.python_exe,
                    k_angles=args.k_angles,
                    max_workers=max_workers,
                    gpu_ids=gpu_ids,
                    pair_timeout_sec=pair_timeout_sec,
                )
                rows.append(row)
                if row.get("summary_json"):
                    print(row["summary_json"])

    summary_csv = os.path.join(output_root, "raw_rewrite_summary_all.csv")
    summary_json = os.path.join(output_root, "raw_rewrite_summary_all.json")
    summary_txt = os.path.join(output_root, "raw_rewrite_summary_all.txt")
    write_rows_csv(summary_csv, rows)
    write_rows_txt(summary_txt, rows, max_cols=18)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(os.path.abspath(summary_csv))
    print(os.path.abspath(summary_json))
    print(os.path.abspath(summary_txt))


if __name__ == "__main__":
    main()
