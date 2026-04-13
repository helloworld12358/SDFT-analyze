#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect LoRA norm statistics for (train_dataset, method, epoch).

Main metric:
- BA Frobenius norm with LoRA scaling: || (alpha / r) * B @ A ||_F

Also collect:
- ||A||_F
- ||B||_F

Epoch behavior:
- epoch_0: base model only, no adapter => all norms are 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAINF_ROOT_DEFAULT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(DATAINF_ROOT_DEFAULT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gram_scheme_a_utils import (  # type: ignore  # noqa: E402
    DEFAULT_EPOCHS,
    DEFAULT_METHODS,
    DEFAULT_TRAIN_DATASETS,
    detect_datainf_root,
    resolve_checkpoint_path,
    resolve_result_root,
    resolve_sdft_root,
    write_rows_csv,
)

try:
    from safetensors.torch import load_file as safetensors_load_file  # type: ignore
except Exception:
    safetensors_load_file = None


def split_csv_arg(s: str, default: Sequence[str]) -> List[str]:
    if not s.strip():
        return list(default)
    out = [x.strip() for x in s.split(",") if x.strip()]
    return out if out else list(default)


def resolve_output_root(datainf_root: str, output_root: str) -> str:
    if output_root.strip():
        return os.path.abspath(output_root.strip())
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    return os.path.join(result_root, "lora_norm")


def adapter_weight_file(checkpoint_path: str) -> Optional[str]:
    candidates = [
        os.path.join(checkpoint_path, "adapter_model.safetensors"),
        os.path.join(checkpoint_path, "adapter_model.bin"),
        os.path.join(checkpoint_path, "adapter_model.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def load_adapter_config(checkpoint_path: str) -> Dict[str, object]:
    path = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        if safetensors_load_file is None:
            raise RuntimeError("safetensors is not available but adapter_model.safetensors was found")
        data = safetensors_load_file(path, device="cpu")
        return dict(data)

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return dict(obj)
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
            if all(isinstance(v, torch.Tensor) for v in sd.values()):
                return dict(sd)
    raise RuntimeError(f"unsupported adapter checkpoint format: {path}")


def strip_prefix_repeated(name: str, prefix: str) -> str:
    out = name
    while out.startswith(prefix):
        out = out[len(prefix) :]
    return out


def canonical_module_name(base_key: str) -> str:
    m = base_key
    m = strip_prefix_repeated(m, "base_model.model.")
    m = strip_prefix_repeated(m, "model.")
    return m


def pattern_lookup(mapping: Dict[str, object], module_name: str) -> Optional[float]:
    if not mapping:
        return None
    if module_name in mapping:
        try:
            return float(mapping[module_name])  # type: ignore[arg-type]
        except Exception:
            return None

    best_key = None
    for k in mapping.keys():
        if module_name.endswith(k):
            if best_key is None or len(k) > len(best_key):
                best_key = k
    if best_key is None:
        return None
    try:
        return float(mapping[best_key])  # type: ignore[arg-type]
    except Exception:
        return None


def lora_scale(cfg: Dict[str, object], module_name: str, rank_from_weight: int) -> float:
    rank_pattern = cfg.get("rank_pattern", {})
    alpha_pattern = cfg.get("alpha_pattern", {})

    if not isinstance(rank_pattern, dict):
        rank_pattern = {}
    if not isinstance(alpha_pattern, dict):
        alpha_pattern = {}

    default_r = cfg.get("r", rank_from_weight)
    default_alpha = cfg.get("lora_alpha", rank_from_weight)

    try:
        r_val = float(default_r)  # type: ignore[arg-type]
    except Exception:
        r_val = float(rank_from_weight)
    try:
        alpha_val = float(default_alpha)  # type: ignore[arg-type]
    except Exception:
        alpha_val = float(rank_from_weight)

    r_hit = pattern_lookup(rank_pattern, module_name)
    a_hit = pattern_lookup(alpha_pattern, module_name)
    if r_hit is not None:
        r_val = r_hit
    if a_hit is not None:
        alpha_val = a_hit

    if not math.isfinite(r_val) or r_val <= 0:
        r_val = float(rank_from_weight)
    if not math.isfinite(alpha_val):
        alpha_val = float(rank_from_weight)
    return float(alpha_val / r_val)


def fro_norm_ba_scaled(a: torch.Tensor, b: torch.Tensor, scale: float) -> float:
    # ||scale * B A||_F^2 = scale^2 * tr((B^T B)(A A^T))
    a2 = a.detach().to(dtype=torch.float64, device="cpu")
    b2 = b.detach().to(dtype=torch.float64, device="cpu")
    bt_b = b2.transpose(0, 1).matmul(b2)
    aa_t = a2.matmul(a2.transpose(0, 1))
    sq = float(torch.trace(bt_b.matmul(aa_t)).item())
    sq = max(sq, 0.0)
    return float(abs(scale) * math.sqrt(sq))


def collect_one_checkpoint(checkpoint_path: str) -> Tuple[Optional[Dict[str, float]], Optional[str], Optional[str]]:
    weight_file = adapter_weight_file(checkpoint_path)
    if not weight_file:
        return None, None, "missing adapter_model.safetensors/bin/pt"

    cfg = load_adapter_config(checkpoint_path)
    try:
        sd = load_state_dict(weight_file)
    except Exception as e:
        return None, os.path.abspath(weight_file), f"failed to load adapter weights: {e}"

    modules = 0
    sum_ba_sq = 0.0
    sum_a_sq = 0.0
    sum_b_sq = 0.0

    suffix_a = ".lora_A.weight"
    suffix_b = ".lora_B.weight"

    for k, a in sd.items():
        if not isinstance(k, str) or not k.endswith(suffix_a):
            continue
        base = k[: -len(suffix_a)]
        b_key = base + suffix_b
        b = sd.get(b_key, None)
        if b is None or not isinstance(b, torch.Tensor):
            continue
        if not isinstance(a, torch.Tensor):
            continue
        if a.ndim != 2 or b.ndim != 2:
            continue
        if b.shape[1] != a.shape[0]:
            continue

        module = canonical_module_name(base)
        scale = lora_scale(cfg, module, int(a.shape[0]))
        ba = fro_norm_ba_scaled(a, b, scale)
        an = float(torch.linalg.norm(a.detach().to(dtype=torch.float64, device="cpu"), ord="fro").item())
        bn = float(torch.linalg.norm(b.detach().to(dtype=torch.float64, device="cpu"), ord="fro").item())

        sum_ba_sq += ba * ba
        sum_a_sq += an * an
        sum_b_sq += bn * bn
        modules += 1

    if modules == 0:
        return None, os.path.abspath(weight_file), "no valid lora_A/lora_B pairs found"

    return (
        {
            "modules_found": float(modules),
            "ba_norm_global": float(math.sqrt(sum_ba_sq)),
            "a_norm_global": float(math.sqrt(sum_a_sq)),
            "b_norm_global": float(math.sqrt(sum_b_sq)),
        },
        os.path.abspath(weight_file),
        None,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Collect LoRA BA/A/B norm stats for epoch_0/1/5.")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default="")
    p.add_argument("--train_datasets", type=str, default=",".join(DEFAULT_TRAIN_DATASETS))
    p.add_argument("--methods", type=str, default="sft,sdft")
    p.add_argument("--epochs", type=str, default="epoch_0,epoch_1,epoch_5")
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    sdft_root = resolve_sdft_root(datainf_root)
    output_root = resolve_output_root(datainf_root, args.output_root)
    os.makedirs(output_root, exist_ok=True)

    train_datasets = split_csv_arg(args.train_datasets, DEFAULT_TRAIN_DATASETS)
    methods = split_csv_arg(args.methods, DEFAULT_METHODS)
    epochs = split_csv_arg(args.epochs, DEFAULT_EPOCHS)

    rows: List[Dict[str, object]] = []
    unavailable: List[Dict[str, object]] = []

    for train_dataset in train_datasets:
        for method in methods:
            for epoch in epochs:
                row: Dict[str, object] = {
                    "train_dataset": train_dataset,
                    "method": method,
                    "epoch": epoch,
                    "checkpoint_path": "",
                    "adapter_weight_path": "",
                    "modules_found": None,
                    "ba_norm_global": None,
                    "a_norm_global": None,
                    "b_norm_global": None,
                    "status": "ok",
                    "reason": "",
                }

                if epoch == "epoch_0":
                    row.update(
                        {
                            "modules_found": 0,
                            "ba_norm_global": 0.0,
                            "a_norm_global": 0.0,
                            "b_norm_global": 0.0,
                            "status": "ok_epoch0_base_only",
                            "reason": "epoch_0 has no adapter; use zero shift baseline",
                        }
                    )
                    rows.append(row)
                    continue

                ckpt = resolve_checkpoint_path(sdft_root, epoch, train_dataset, method)
                if not ckpt or not os.path.isdir(ckpt):
                    row["status"] = "missing_checkpoint"
                    row["reason"] = f"checkpoint missing for {train_dataset}/{method}/{epoch}"
                    rows.append(row)
                    unavailable.append(
                        {
                            "train_dataset": train_dataset,
                            "method": method,
                            "epoch": epoch,
                            "reason": str(row["reason"]),
                        }
                    )
                    continue

                row["checkpoint_path"] = os.path.abspath(ckpt)
                stats, adapter_path, err = collect_one_checkpoint(ckpt)
                if adapter_path:
                    row["adapter_weight_path"] = adapter_path
                if err:
                    row["status"] = "error_collect"
                    row["reason"] = err
                    rows.append(row)
                    unavailable.append(
                        {
                            "train_dataset": train_dataset,
                            "method": method,
                            "epoch": epoch,
                            "checkpoint_path": os.path.abspath(ckpt),
                            "reason": err,
                        }
                    )
                    continue

                assert stats is not None
                row.update(
                    {
                        "modules_found": int(stats["modules_found"]),
                        "ba_norm_global": float(stats["ba_norm_global"]),
                        "a_norm_global": float(stats["a_norm_global"]),
                        "b_norm_global": float(stats["b_norm_global"]),
                        "status": "ok",
                        "reason": "",
                    }
                )
                rows.append(row)

    out_json = os.path.join(output_root, "lora_norm_rows.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    out_csv = os.path.join(output_root, "lora_norm_rows.csv")
    write_rows_csv(out_csv, rows)

    unavailable_json = os.path.join(output_root, "unavailable_lora_norm.json")
    payload = {
        "datainf_root": os.path.abspath(datainf_root),
        "sdft_root": os.path.abspath(sdft_root),
        "row_count": len(rows),
        "unavailable_count": len(unavailable),
        "unavailable": unavailable,
    }
    with open(unavailable_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(os.path.abspath(out_json))
    print(os.path.abspath(out_csv))
    print(os.path.abspath(unavailable_json))


if __name__ == "__main__":
    main()

