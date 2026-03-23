#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scheme A - Step 10
Build train-vs-test rectangular matrices (7x5 style):
- T[A, B] = g_train(A)^T H_A^{-1} g_test(B)
- C[A, B] = T[A, B] / sqrt( S_train[A] * S_test[A, B] )

Where:
- rows A are train datasets
- cols B are test tasks
- H_A is induced by train dataset A (same setting as calc_dataset_similarity)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAINF_ROOT_DEFAULT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(DATAINF_ROOT_DEFAULT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gram_scheme_a_utils import (  # type: ignore  # noqa: E402
    DEFAULT_EPOCHS,
    DEFAULT_TASKS,
    DEFAULT_TRAIN_DATASETS,
    detect_datainf_root,
    ensure_dir,
    load_existing_ownh_from_analysis,
    load_pairwise_matrix_any,
    normalize_epoch_list,
    normalize_method_list,
    resolve_checkpoint_path,
    resolve_existing_result_roots,
    resolve_grad_path,
    resolve_result_root,
    resolve_sdft_root,
    resolve_train_dataset_path,
    run_calc_dataset_similarity_pair,
    split_csv_arg,
    write_rows_csv,
    write_rows_txt,
    write_unavailable_note,
)


def _to_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _load_json_score(path: str) -> Optional[float]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None
        return _to_float(obj.get("score"))
    except Exception:
        return None


def _load_multi_score_map(path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not os.path.isfile(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return out
    if not isinstance(obj, dict):
        return out
    score_map = obj.get("score_map", {})
    if isinstance(score_map, dict):
        for k, v in score_map.items():
            fv = _to_float(v)
            if fv is not None:
                out[str(k)] = fv
    return out


def _save_json(path: str, obj: object) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def _select_train_datasets(args) -> List[str]:
    if args.all_train_datasets:
        return list(DEFAULT_TRAIN_DATASETS)
    if args.train_dataset:
        return split_csv_arg(args.train_dataset, DEFAULT_TRAIN_DATASETS)
    return list(DEFAULT_TRAIN_DATASETS)


def _select_epochs(args) -> List[str]:
    if args.all_epochs:
        return list(DEFAULT_EPOCHS)
    if args.epoch:
        return normalize_epoch_list(split_csv_arg(args.epoch, DEFAULT_EPOCHS))
    return list(DEFAULT_EPOCHS)


def _select_methods(args) -> List[str]:
    methods = normalize_method_list(split_csv_arg(args.method, ["both"]))
    return methods if methods else ["sft", "sdft"]


def _run_save_avg_grad(
    python_exe: str,
    save_script: str,
    base_model_path: str,
    lora_path: Optional[str],
    dataset_path: str,
    output_path: str,
    batch_size: int,
    max_length: int,
    max_samples: Optional[int],
    env_overrides: Optional[Dict[str, str]] = None,
    timeout_sec: Optional[int] = None,
) -> Tuple[bool, str]:
    cmd = [
        python_exe,
        save_script,
        "--base_model_path",
        base_model_path,
        "--dataset_path",
        dataset_path,
        "--output_path",
        output_path,
        "--batch_size",
        str(batch_size),
        "--max_length",
        str(max_length),
        "--lora_target",
        "q_proj,v_proj",
    ]
    if lora_path:
        cmd.extend(["--lora_path", lora_path])
    if max_samples is not None and max_samples > 0:
        cmd.extend(["--max_samples", str(max_samples)])

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
        return False, f"timeout after {timeout_sec}s\n{out}\n{err}".strip()


def _run_calc_dataset_similarity_multi(
    python_exe: str,
    multi_script: str,
    base_model_path: str,
    train_dataset_path: str,
    grad1_path: str,
    grad2_paths: Sequence[str],
    grad2_names: Sequence[str],
    out_path: str,
    lora_path: Optional[str] = None,
    damping: Optional[float] = None,
    env_overrides: Optional[Dict[str, str]] = None,
    timeout_sec: Optional[int] = None,
) -> Tuple[bool, str]:
    cmd = [
        python_exe,
        multi_script,
        "--base_model_path",
        base_model_path,
        "--train_dataset_path",
        train_dataset_path,
        "--grad1_path",
        grad1_path,
        "--grad2_paths",
        ",".join(grad2_paths),
        "--grad2_names",
        ",".join(grad2_names),
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
        return False, f"timeout after {timeout_sec}s\n{out}\n{err}".strip()


def _diag_from_existing_ownh(
    result_roots: Sequence[str],
    datainf_root: str,
    train_dataset: str,
    epoch: str,
    method: str,
    task_names: Sequence[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    existing = load_existing_ownh_from_analysis(result_roots, train_dataset, epoch)
    t_mat = existing.get(method, {}).get("T")
    if isinstance(t_mat, np.ndarray) and t_mat.shape == (len(task_names), len(task_names)):
        for i, t in enumerate(task_names):
            v = _to_float(t_mat[i, i])
            if v is not None:
                out[t] = v
        if len(out) == len(task_names):
            return out

    t2 = load_pairwise_matrix_any(
        datainf_root=datainf_root,
        model=train_dataset,
        epoch=epoch,
        method=method,
        names=task_names,
        extra_result_roots=result_roots,
    )
    if isinstance(t2, np.ndarray) and t2.shape == (len(task_names), len(task_names)):
        for i, t in enumerate(task_names):
            v = _to_float(t2[i, i])
            if v is not None:
                out[t] = v
    return out


def _rect_summary_stats(T: np.ndarray, C: np.ndarray) -> Dict[str, object]:
    t_flat = T[np.isfinite(T)]
    c_flat = C[np.isfinite(C)]
    return {
        "shape_T": list(T.shape),
        "shape_C": list(C.shape),
        "T_mean": float(np.mean(t_flat)) if t_flat.size else None,
        "T_std": float(np.std(t_flat)) if t_flat.size else None,
        "C_mean": float(np.mean(c_flat)) if c_flat.size else None,
        "C_std": float(np.std(c_flat)) if c_flat.size else None,
        "T_nan_count": int(np.isnan(T).sum()),
        "C_nan_count": int(np.isnan(C).sum()),
    }


def _save_rect_bundle(
    out_dir: str,
    tag: str,
    T: np.ndarray,
    C: np.ndarray,
    row_names: Sequence[str],
    col_names: Sequence[str],
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, str]:
    ensure_dir(out_dir)
    t_npy = os.path.join(out_dir, f"T_{tag}.npy")
    t_csv = os.path.join(out_dir, f"T_{tag}.csv")
    c_npy = os.path.join(out_dir, f"C_{tag}.npy")
    c_csv = os.path.join(out_dir, f"C_{tag}.csv")
    summary_json = os.path.join(out_dir, f"summary_{tag}.json")

    np.save(t_npy, T)
    np.savetxt(t_csv, T, delimiter=",", fmt="%.18e")
    np.save(c_npy, C)
    np.savetxt(c_csv, C, delimiter=",", fmt="%.18e")

    payload: Dict[str, object] = {
        "tag": tag,
        "row_names": list(row_names),
        "col_names": list(col_names),
        "paths": {
            "T_npy": os.path.abspath(t_npy),
            "T_csv": os.path.abspath(t_csv),
            "C_npy": os.path.abspath(c_npy),
            "C_csv": os.path.abspath(c_csv),
        },
        "stats": _rect_summary_stats(T, C),
        "normalization_formula": "C[A,B] = T[A,B] / sqrt(S_train[A] * S_test[A,B])",
    }
    if metadata:
        payload["metadata"] = metadata
    _save_json(summary_json, payload)
    return {
        "T_npy": os.path.abspath(t_npy),
        "T_csv": os.path.abspath(t_csv),
        "C_npy": os.path.abspath(c_npy),
        "C_csv": os.path.abspath(c_csv),
        "summary_json": os.path.abspath(summary_json),
    }


@dataclass
class RowComputeResult:
    train_dataset: str
    epoch: str
    method: str
    row_index: int
    t_values: List[float]
    c_values: List[float]
    s_train: Optional[float]
    s_test_map: Dict[str, Optional[float]]
    status: str
    message: str
    pairwise_dir: str
    train_grad_path: str
    unavailable_file: Optional[str]


def _compute_one_row(
    datainf_root: str,
    sdft_root: str,
    result_roots: Sequence[str],
    output_root: str,
    train_dataset: str,
    epoch: str,
    method: str,
    row_index: int,
    task_names: Sequence[str],
    base_model_path: str,
    damping: float,
    python_exe: str,
    gpu_id: Optional[str],
    pair_timeout_sec: Optional[int],
    compute_missing_train_grads: bool,
    train_grad_batch_size: int,
    train_grad_max_length: int,
    train_grad_max_samples: Optional[int],
) -> RowComputeResult:
    cache_dir = ensure_dir(os.path.join(output_root, "_cache", train_dataset, epoch, method))
    pairwise_dir = ensure_dir(os.path.join(cache_dir, "pairwise_result"))
    unavailable_file: Optional[str] = None

    train_dataset_path = resolve_train_dataset_path(sdft_root, train_dataset, method)
    lora_path = resolve_checkpoint_path(sdft_root, epoch, train_dataset, method)
    train_grad_path = os.path.join(cache_dir, "train_self.pt")

    env_overrides: Dict[str, str] = {"TOKENIZERS_PARALLELISM": "false"}
    if gpu_id is not None:
        env_overrides["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if not os.path.isfile(train_dataset_path):
        reason_file = write_unavailable_note(
            os.path.join(cache_dir, f"unavailable_{train_dataset}_{epoch}_{method}_train_data.json"),
            reason="train dataset file missing",
            context={"train_dataset_path": train_dataset_path},
        )
        return RowComputeResult(
            train_dataset=train_dataset,
            epoch=epoch,
            method=method,
            row_index=row_index,
            t_values=[float("nan")] * len(task_names),
            c_values=[float("nan")] * len(task_names),
            s_train=None,
            s_test_map={t: None for t in task_names},
            status="unavailable",
            message="train dataset missing",
            pairwise_dir=pairwise_dir,
            train_grad_path=train_grad_path,
            unavailable_file=os.path.abspath(reason_file),
        )

    if not os.path.isfile(train_grad_path):
        if not compute_missing_train_grads:
            reason_file = write_unavailable_note(
                os.path.join(cache_dir, f"unavailable_{train_dataset}_{epoch}_{method}_train_grad.json"),
                reason="train gradient missing and compute_missing_train_grads=0",
                context={"train_grad_path": train_grad_path},
            )
            return RowComputeResult(
                train_dataset=train_dataset,
                epoch=epoch,
                method=method,
                row_index=row_index,
                t_values=[float("nan")] * len(task_names),
                c_values=[float("nan")] * len(task_names),
                s_train=None,
                s_test_map={t: None for t in task_names},
                status="unavailable",
                message="train gradient missing",
                pairwise_dir=pairwise_dir,
                train_grad_path=train_grad_path,
                unavailable_file=os.path.abspath(reason_file),
            )

        save_script = os.path.join(datainf_root, "src", "save_avg_grad_with_integrated_templates.py")
        ok, msg = _run_save_avg_grad(
            python_exe=python_exe,
            save_script=save_script,
            base_model_path=base_model_path,
            lora_path=lora_path,
            dataset_path=train_dataset_path,
            output_path=train_grad_path,
            batch_size=train_grad_batch_size,
            max_length=train_grad_max_length,
            max_samples=train_grad_max_samples,
            env_overrides=env_overrides,
            timeout_sec=pair_timeout_sec,
        )
        if (not ok) or (not os.path.isfile(train_grad_path)):
            reason_file = write_unavailable_note(
                os.path.join(cache_dir, f"unavailable_{train_dataset}_{epoch}_{method}_train_grad_build.json"),
                reason="failed to build train gradient",
                context={
                    "train_dataset_path": train_dataset_path,
                    "train_grad_path": train_grad_path,
                    "message": msg[:4000],
                },
            )
            return RowComputeResult(
                train_dataset=train_dataset,
                epoch=epoch,
                method=method,
                row_index=row_index,
                t_values=[float("nan")] * len(task_names),
                c_values=[float("nan")] * len(task_names),
                s_train=None,
                s_test_map={t: None for t in task_names},
                status="unavailable",
                message="train gradient build failed",
                pairwise_dir=pairwise_dir,
                train_grad_path=train_grad_path,
                unavailable_file=os.path.abspath(reason_file),
            )

    test_grad_paths = {t: resolve_grad_path(datainf_root, epoch, method, train_dataset, t) for t in task_names}
    missing_test = {t: p for t, p in test_grad_paths.items() if not os.path.isfile(p)}
    if missing_test:
        reason_file = write_unavailable_note(
            os.path.join(cache_dir, f"unavailable_{train_dataset}_{epoch}_{method}_test_grads.json"),
            reason="missing test gradient files",
            context={"missing_test_grad_paths": missing_test},
        )
        return RowComputeResult(
            train_dataset=train_dataset,
            epoch=epoch,
            method=method,
            row_index=row_index,
            t_values=[float("nan")] * len(task_names),
            c_values=[float("nan")] * len(task_names),
            s_train=None,
            s_test_map={t: None for t in task_names},
            status="unavailable",
            message="missing test gradients",
            pairwise_dir=pairwise_dir,
            train_grad_path=train_grad_path,
            unavailable_file=os.path.abspath(reason_file),
        )

    # Multi-target run: one pass for train_self + all test tasks
    multi_json = os.path.join(pairwise_dir, "sim_trainself_multi.json")
    multi_score_map = _load_multi_score_map(multi_json)
    need_keys = ["train_self"] + [f"task::{t}" for t in task_names]
    if not all(k in multi_score_map for k in need_keys):
        multi_script = os.path.join(datainf_root, "src", "calc_dataset_similarity_multi.py")
        grad2_names = ["train_self"] + [f"task::{t}" for t in task_names]
        grad2_paths = [train_grad_path] + [test_grad_paths[t] for t in task_names]
        ok, _msg = _run_calc_dataset_similarity_multi(
            python_exe=python_exe,
            multi_script=multi_script,
            base_model_path=base_model_path,
            train_dataset_path=train_dataset_path,
            grad1_path=train_grad_path,
            grad2_paths=grad2_paths,
            grad2_names=grad2_names,
            out_path=multi_json,
            lora_path=lora_path,
            damping=damping,
            env_overrides=env_overrides,
            timeout_sec=pair_timeout_sec,
        )
        if ok:
            multi_score_map = _load_multi_score_map(multi_json)

    s_train = multi_score_map.get("train_self")

    s_test_map: Dict[str, Optional[float]] = {t: None for t in task_names}
    diag_existing = _diag_from_existing_ownh(
        result_roots=result_roots,
        datainf_root=datainf_root,
        train_dataset=train_dataset,
        epoch=epoch,
        method=method,
        task_names=task_names,
    )
    for t in task_names:
        if t in diag_existing:
            s_test_map[t] = _to_float(diag_existing[t])

    t_values: List[float] = []
    c_values: List[float] = []
    fail_count = 0

    for t in task_names:
        score = multi_score_map.get(f"task::{t}")
        # fallback to legacy single-pair json if multi output missed this task
        if score is None:
            out_json = os.path.join(pairwise_dir, f"sim_trainself_{t}.json")
            score = _load_json_score(out_json)
        # final fallback: run single pair once
        if score is None:
            out_json = os.path.join(pairwise_dir, f"sim_trainself_{t}.json")
            ok, _msg = run_calc_dataset_similarity_pair(
                python_exe=python_exe,
                calc_script=os.path.join(datainf_root, "src", "calc_dataset_similarity.py"),
                base_model_path=base_model_path,
                train_dataset_path=train_dataset_path,
                grad1_path=train_grad_path,
                grad2_path=test_grad_paths[t],
                out_path=out_json,
                lora_path=lora_path,
                damping=damping,
                env_overrides=env_overrides,
                timeout_sec=pair_timeout_sec,
            )
            if ok:
                score = _load_json_score(out_json)
        if score is None:
            t_values.append(float("nan"))
            c_values.append(float("nan"))
            fail_count += 1
            continue

        # fallback S_test via explicit self pair if ownH diag unavailable
        if s_test_map[t] is None:
            self_json = os.path.join(pairwise_dir, f"sim_{t}_{t}.json")
            self_score = _load_json_score(self_json)
            if self_score is None:
                ok, _ = run_calc_dataset_similarity_pair(
                    python_exe=python_exe,
                    calc_script=os.path.join(datainf_root, "src", "calc_dataset_similarity.py"),
                    base_model_path=base_model_path,
                    train_dataset_path=train_dataset_path,
                    grad1_path=test_grad_paths[t],
                    grad2_path=test_grad_paths[t],
                    out_path=self_json,
                    lora_path=lora_path,
                    damping=damping,
                    env_overrides=env_overrides,
                    timeout_sec=pair_timeout_sec,
                )
                if ok:
                    self_score = _load_json_score(self_json)
            s_test_map[t] = self_score

        t_values.append(float(score))
        s_t = s_test_map[t]
        if s_train is None or s_t is None or (not math.isfinite(s_train)) or (not math.isfinite(s_t)) or s_train <= 0 or s_t <= 0:
            c_values.append(float("nan"))
        else:
            c_values.append(float(score) / math.sqrt(float(s_train) * float(s_t)))

    status = "ok"
    message = "complete"
    if fail_count > 0:
        status = "partial" if fail_count < len(task_names) else "unavailable"
        message = f"{fail_count}/{len(task_names)} task scores missing"
        unavailable_file = os.path.abspath(
            write_unavailable_note(
                os.path.join(cache_dir, f"unavailable_{train_dataset}_{epoch}_{method}_row.json"),
                reason=message,
                context={
                    "train_dataset": train_dataset,
                    "epoch": epoch,
                    "method": method,
                    "pairwise_dir": pairwise_dir,
                    "train_grad_path": train_grad_path,
                },
            )
        )

    return RowComputeResult(
        train_dataset=train_dataset,
        epoch=epoch,
        method=method,
        row_index=row_index,
        t_values=t_values,
        c_values=c_values,
        s_train=s_train,
        s_test_map=s_test_map,
        status=status,
        message=message,
        pairwise_dir=pairwise_dir,
        train_grad_path=train_grad_path,
        unavailable_file=unavailable_file,
    )


def _build_txt_report(path: str, payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("SchemeA Train-Test Rectangular Matrix Summary")
    lines.append("=" * 88)
    meta = payload.get("meta", {})
    if isinstance(meta, dict):
        for k in sorted(meta.keys()):
            lines.append(f"{k}: {meta[k]}")
    lines.append("")
    rows = payload.get("rows", [])
    if isinstance(rows, list):
        lines.append(f"rows_count: {len(rows)}")
        lines.append("")
        for i, r in enumerate(rows):
            if not isinstance(r, dict):
                continue
            lines.append("-" * 88)
            lines.append(f"[{i}] train_dataset={r.get('train_dataset')} epoch={r.get('epoch')} method={r.get('method')} status={r.get('status')}")
            lines.append(f"message: {r.get('message')}")
            lines.append(f"s_train: {r.get('s_train')}")
            lines.append(f"summary_json: {r.get('summary_json')}")
            lines.append(f"T_row: {r.get('T_row')}")
            lines.append(f"C_row: {r.get('C_row')}")
            lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Scheme A step10: build train-vs-test rectangular T/C matrices.")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default=None, help="Default: <result_root>/schemeA/train_test_rect")
    p.add_argument("--train_dataset", type=str, default="")
    p.add_argument("--all_train_datasets", action="store_true")
    p.add_argument("--epoch", type=str, default="")
    p.add_argument("--all_epochs", action="store_true")
    p.add_argument("--method", type=str, default="both", help="sft/sdft/both")
    p.add_argument("--task_names", type=str, default=",".join(DEFAULT_TASKS))
    p.add_argument("--existing_result_roots", type=str, default="", help="comma-separated roots to recover ownH diag")
    p.add_argument("--base_model_path", type=str, default=None)
    p.add_argument("--damping", type=float, default=0.001)
    p.add_argument("--python_exe", type=str, default=sys.executable)
    p.add_argument("--gpu_ids", type=str, default="", help="comma-separated ids, e.g. 0,1,2,3")
    p.add_argument("--num_workers", type=int, default=0, help="<=0 means auto by gpu_ids count")
    p.add_argument("--pair_timeout_sec", type=int, default=0, help="<=0 means no timeout")
    p.add_argument("--compute_missing_train_grads", action="store_true")
    p.add_argument("--train_grad_batch_size", type=int, default=8)
    p.add_argument("--train_grad_max_length", type=int, default=1024)
    p.add_argument("--train_grad_max_samples", type=int, default=0)
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    sdft_root = resolve_sdft_root(datainf_root)
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    output_root = args.output_root or os.path.join(result_root, "schemeA", "train_test_rect")
    ensure_dir(output_root)

    train_datasets = _select_train_datasets(args)
    epochs = _select_epochs(args)
    methods = _select_methods(args)
    task_names = split_csv_arg(args.task_names, DEFAULT_TASKS)

    base_model_path = args.base_model_path or os.path.join(sdft_root, "model", "Llama-2-7b-chat-hf")
    extra_roots = [x.strip() for x in args.existing_result_roots.split(",") if x.strip()]
    result_roots = resolve_existing_result_roots(datainf_root, explicit_roots=extra_roots)

    gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        env_gpu = os.environ.get("SCHEMEA_GPU_IDS", "").strip() or os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if env_gpu:
            gpu_ids = [x.strip() for x in env_gpu.split(",") if x.strip()]
    workers_default = len(gpu_ids) if gpu_ids else 1
    num_workers = args.num_workers if args.num_workers > 0 else workers_default
    num_workers = max(1, num_workers)
    pair_timeout_sec = None if args.pair_timeout_sec <= 0 else int(args.pair_timeout_sec)
    train_grad_max_samples = args.train_grad_max_samples if args.train_grad_max_samples > 0 else None

    row_index_map = {d: i for i, d in enumerate(train_datasets)}
    t_mats: Dict[Tuple[str, str], np.ndarray] = {}
    c_mats: Dict[Tuple[str, str], np.ndarray] = {}
    for e in epochs:
        for m in methods:
            t_mats[(e, m)] = np.full((len(train_datasets), len(task_names)), np.nan, dtype=np.float64)
            c_mats[(e, m)] = np.full((len(train_datasets), len(task_names)), np.nan, dtype=np.float64)

    jobs: List[Tuple[int, str, str, str]] = []
    for e in epochs:
        for m in methods:
            for d in train_datasets:
                jobs.append((len(jobs), d, e, m))

    rows_summary: List[Dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        fut_map = {}
        for job_idx, d, e, m in jobs:
            gpu_id = gpu_ids[job_idx % len(gpu_ids)] if gpu_ids else None
            fut = ex.submit(
                _compute_one_row,
                datainf_root,
                sdft_root,
                result_roots,
                output_root,
                d,
                e,
                m,
                row_index_map[d],
                task_names,
                base_model_path,
                float(args.damping),
                args.python_exe,
                gpu_id,
                pair_timeout_sec,
                bool(args.compute_missing_train_grads),
                int(args.train_grad_batch_size),
                int(args.train_grad_max_length),
                train_grad_max_samples,
            )
            fut_map[fut] = (job_idx, d, e, m)

        for fut in as_completed(fut_map):
            _, d, e, m = fut_map[fut]
            try:
                rr = fut.result()
            except Exception as err:
                cache_dir = ensure_dir(os.path.join(output_root, "_cache", d, e, m))
                uf = write_unavailable_note(
                    os.path.join(cache_dir, f"unavailable_{d}_{e}_{m}_exception.json"),
                    reason=f"row compute exception: {err}",
                )
                rows_summary.append(
                    {
                        "train_dataset": d,
                        "epoch": e,
                        "method": m,
                        "status": "unavailable",
                        "message": str(err),
                        "unavailable_file": os.path.abspath(uf),
                    }
                )
                continue

            t_mats[(e, m)][rr.row_index, :] = np.asarray(rr.t_values, dtype=np.float64)
            c_mats[(e, m)][rr.row_index, :] = np.asarray(rr.c_values, dtype=np.float64)
            rows_summary.append(
                {
                    "train_dataset": rr.train_dataset,
                    "epoch": rr.epoch,
                    "method": rr.method,
                    "status": rr.status,
                    "message": rr.message,
                    "s_train": rr.s_train,
                    "pairwise_dir": os.path.abspath(rr.pairwise_dir),
                    "train_grad_path": os.path.abspath(rr.train_grad_path),
                    "unavailable_file": rr.unavailable_file,
                    "T_row": rr.t_values,
                    "C_row": rr.c_values,
                }
            )

    # Save per epoch-method full 7x5 matrices
    matrix_rows: List[Dict[str, object]] = []
    for e in epochs:
        for m in methods:
            out_dir = ensure_dir(os.path.join(output_root, e, m))
            tag = f"train_test_{e}_{m}_7x5"
            bundle = _save_rect_bundle(
                out_dir=out_dir,
                tag=tag,
                T=t_mats[(e, m)],
                C=c_mats[(e, m)],
                row_names=train_datasets,
                col_names=task_names,
                metadata={
                    "mode": "train_test_rect",
                    "epoch": e,
                    "method": m,
                    "train_datasets": train_datasets,
                    "task_names": task_names,
                },
            )
            matrix_rows.append(
                {
                    "mode": "train_test_rect",
                    "epoch": e,
                    "method": m,
                    "status": "ok",
                    "summary_json": bundle["summary_json"],
                    "T_npy": bundle["T_npy"],
                    "C_npy": bundle["C_npy"],
                }
            )
            print(bundle["summary_json"])

        # optional diff if both methods available
        if "sft" in methods and "sdft" in methods:
            out_dir = ensure_dir(os.path.join(output_root, e, "sft_minus_sdft"))
            tag = f"train_test_{e}_sft_minus_sdft_7x5"
            t_diff = t_mats[(e, "sft")] - t_mats[(e, "sdft")]
            c_diff = c_mats[(e, "sft")] - c_mats[(e, "sdft")]
            bundle = _save_rect_bundle(
                out_dir=out_dir,
                tag=tag,
                T=t_diff,
                C=c_diff,
                row_names=train_datasets,
                col_names=task_names,
                metadata={
                    "mode": "train_test_rect_diff",
                    "epoch": e,
                    "diff_definition": "sft_minus_sdft",
                    "train_datasets": train_datasets,
                    "task_names": task_names,
                },
            )
            matrix_rows.append(
                {
                    "mode": "train_test_rect_diff",
                    "epoch": e,
                    "method": "sft_minus_sdft",
                    "status": "ok",
                    "summary_json": bundle["summary_json"],
                    "T_npy": bundle["T_npy"],
                    "C_npy": bundle["C_npy"],
                }
            )
            print(bundle["summary_json"])

    rows_summary.sort(key=lambda x: (str(x.get("train_dataset")), str(x.get("epoch")), str(x.get("method"))))
    matrix_rows.sort(key=lambda x: (str(x.get("epoch")), str(x.get("method"))))

    summary_json = os.path.join(output_root, "train_test_rect_row_summary.json")
    _save_json(summary_json, rows_summary)
    write_rows_csv(os.path.join(output_root, "train_test_rect_row_summary.csv"), rows_summary)
    write_rows_txt(os.path.join(output_root, "train_test_rect_row_summary.txt"), rows_summary)

    final_payload: Dict[str, object] = {
        "meta": {
            "datainf_root": os.path.abspath(datainf_root),
            "output_root": os.path.abspath(output_root),
            "train_datasets": train_datasets,
            "task_names": task_names,
            "epochs": epochs,
            "methods": methods,
            "formula_T": "T[A,B] = g_train(A)^T H_A^{-1} g_test(B)",
            "formula_C": "C[A,B] = T[A,B] / sqrt(S_train[A] * S_test[A,B])",
            "S_train": "g_train(A)^T H_A^{-1} g_train(A)",
            "S_test": "g_test(B)^T H_A^{-1} g_test(B)",
            "compute_missing_train_grads": bool(args.compute_missing_train_grads),
        },
        "matrix_rows": matrix_rows,
        "rows": rows_summary,
    }

    final_json = os.path.join(output_root, "train_test_rect_summary.json")
    final_txt = os.path.join(output_root, "train_test_rect_summary.txt")
    _save_json(final_json, final_payload)
    _build_txt_report(final_txt, final_payload)
    print(os.path.abspath(final_json))
    print(os.path.abspath(final_txt))


if __name__ == "__main__":
    main()
