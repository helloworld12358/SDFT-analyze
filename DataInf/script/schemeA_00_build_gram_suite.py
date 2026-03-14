#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scheme A - Step 00
Generic Gram/compatibility builder for arbitrary object set under one H-oracle source.

Input:
- object names + corresponding gradient vector paths
- oracle source represented by (base_model_path, train_dataset_path, optional lora_path)

Output:
- T (Gram), C (correlation), eigvals/eigvecs, shared-mode suite, spectral diagnostics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAINF_ROOT_DEFAULT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(DATAINF_ROOT_DEFAULT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gram_scheme_a_utils import (  # type: ignore  # noqa: E402
    compute_pairwise_scores_via_cli,
    detect_datainf_root,
    save_matrix_bundle,
    write_unavailable_note,
)


def parse_name_grad_mapping(names_csv: str, grads_csv: str) -> Dict[str, str]:
    names = [x.strip() for x in names_csv.split(",") if x.strip()]
    grads = [x.strip() for x in grads_csv.split(",") if x.strip()]
    if len(names) != len(grads):
        raise ValueError(f"name/grad count mismatch: {len(names)} vs {len(grads)}")
    return {n: g for n, g in zip(names, grads)}


def main() -> None:
    p = argparse.ArgumentParser(description="Build Gram suite for one object set and one oracle source.")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--train_dataset_path", type=str, required=True)
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--names_csv", type=str, default=None, help="Comma-separated object names")
    p.add_argument("--grads_csv", type=str, default=None, help="Comma-separated grad .pt paths")
    p.add_argument("--mapping_json", type=str, default=None, help="Optional JSON map {name: grad_path}")
    p.add_argument("--damping", type=float, default=0.001)
    p.add_argument("--python_exe", type=str, default=sys.executable)
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mapping_json:
        with open(args.mapping_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("mapping_json must be a JSON object {name: path}")
        grad_paths = {str(k): str(v) for k, v in obj.items()}
    else:
        if not args.names_csv or not args.grads_csv:
            raise ValueError("Provide either --mapping_json or both --names_csv and --grads_csv")
        grad_paths = parse_name_grad_mapping(args.names_csv, args.grads_csv)

    object_names: List[str] = list(grad_paths.keys())

    run = compute_pairwise_scores_via_cli(
        datainf_root=datainf_root,
        output_dir=args.output_dir,
        base_model_path=args.base_model_path,
        train_dataset_path=args.train_dataset_path,
        grad_paths=grad_paths,
        dataset_names=object_names,
        lora_path=args.lora_path,
        damping=args.damping,
        python_exe=args.python_exe,
    )

    if run.matrix is None:
        unavailable = write_unavailable_note(
            os.path.join(args.output_dir, f"unavailable_{args.tag}.json"),
            reason="pairwise oracle produced no matrix",
            context={"failed_pairs": run.failed_pairs, "pairwise_dir": run.pairwise_dir},
        )
        print(os.path.abspath(unavailable))
        return

    bundle = save_matrix_bundle(
        output_dir=args.output_dir,
        tag=args.tag,
        K=run.matrix,
        object_names=object_names,
        metadata={
            "base_model_path": args.base_model_path,
            "train_dataset_path": args.train_dataset_path,
            "lora_path": args.lora_path,
            "pairwise_dir": run.pairwise_dir,
        },
    )
    print(bundle["summary_json"])


if __name__ == "__main__":
    main()

