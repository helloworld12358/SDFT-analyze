#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_gradients.py
合并多个 rank 的 avg_grad / stats / norms 输出。

用法示例：
    python merge_gradients.py --output_dir analysis/alpaca/gradient_analysis
或显式给 avg_grad 文件：
    python merge_gradients.py --avg_paths analysis/alpaca/gradient_analysis/rank0/avg_grad.pt analysis/.../rank1/avg_grad.pt --output_dir analysis/alpaca/gradient_analysis

输出：
    <output_dir>/merged/
      - avg_grad.pt            # 合并后的加权平均梯度（torch tensor）
      - norms.npy              # 拼接后的每-step norms
      - merged_stats.json      # 合并后的统计（包含近似 trace_covariance）
      - merged_step_meta.json  # 拼接的 step level meta（如果存在）
"""
from pathlib import Path
import argparse
import json
import torch
import numpy as np
from typing import List

def find_rank_dirs(output_dir: Path) -> List[Path]:
    # 找到 output_dir 下的 rank* 子目录（如 rank0, rank1 ...）
    ranks = []
    if not output_dir.exists():
        return []
    for p in sorted(output_dir.iterdir()):
        if p.is_dir() and p.name.lower().startswith("rank"):
            ranks.append(p)
    return ranks

def find_avg_in_dir(d: Path):
    candidates = [
        d / "avg_grad.pt",
        d / "avg_grad.pkl",
        d / "avg_grad.npy"
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def load_stats(d: Path):
    sfile = d / "stats.json"
    if sfile.exists():
        try:
            return json.loads(sfile.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def load_norms(d: Path):
    nfile = d / "norms.npy"
    if nfile.exists():
        try:
            return np.load(nfile)
        except Exception:
            return None
    return None

def collect_step_meta(d: Path):
    # collect step_*.json sorted
    metas = []
    for p in sorted(d.glob("step_*.json")):
        try:
            metas.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return metas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, required=True, help="主输出目录（包含 rank0/rank1/...）")
    ap.add_argument("--avg_paths", type=str, nargs="*", default=[], help="可选：显式的 avg_grad.pt 路径，优先使用")
    ap.add_argument("--save_name", type=str, default="merged", help="保存子目录名（默认 merged）")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    merged_dir = outdir / args.save_name
    merged_dir.mkdir(parents=True, exist_ok=True)

    # 找到要合并的 rank 目录或 avg 文件
    avg_entries = []  # list of dicts: {avg_path, stats, norms, step_meta, rank_dir}
    if args.avg_paths:
        for p in args.avg_paths:
            pth = Path(p)
            if not pth.exists():
                print(f"[Warning] provided avg path not found: {p}")
                continue
            # locate parent dir for stats/norms
            parent = pth.parent
            stats = load_stats(parent)
            norms = load_norms(parent)
            step_meta = collect_step_meta(parent)
            avg_entries.append({"avg_path": pth, "stats": stats, "norms": norms, "meta": step_meta, "dir": parent})
    else:
        # 自动发现 rank dirs
        rank_dirs = find_rank_dirs(outdir)
        if not rank_dirs:
            # maybe single-run outputs directly in outdir (no rank subdirs)
            avg_file = find_avg_in_dir(outdir)
            if avg_file is not None:
                stats = load_stats(outdir)
                norms = load_norms(outdir)
                step_meta = collect_step_meta(outdir)
                avg_entries.append({"avg_path": avg_file, "stats": stats, "norms": norms, "meta": step_meta, "dir": outdir})
        else:
            for rd in rank_dirs:
                avg_file = find_avg_in_dir(rd)
                if avg_file is None:
                    print(f"[Info] No avg_grad found in {rd}, skipping.")
                    continue
                stats = load_stats(rd)
                norms = load_norms(rd)
                step_meta = collect_step_meta(rd)
                avg_entries.append({"avg_path": avg_file, "stats": stats, "norms": norms, "meta": step_meta, "dir": rd})

    if len(avg_entries) == 0:
        print("[Error] No avg_grad files found to merge. Exiting.")
        return 1

    # Load avg tensors and aggregate
    total_steps = 0
    weighted_sum = None
    concatenated_norms = []
    merged_step_meta = []
    trace_cov_list = []
    trace_cov_weights = []
    loaded_count = 0

    for e in avg_entries:
        avg_path = e["avg_path"]
        try:
            avg_vec = torch.load(avg_path, map_location="cpu").float()
        except Exception as ex:
            print(f"[Warning] Failed to load {avg_path}: {ex}. Skipping.")
            continue
        stats = e.get("stats")
        steps = None
        if stats and "processed_steps" in stats:
            try:
                steps = int(stats["processed_steps"])
            except Exception:
                steps = None
        # If no processed_steps, fallback to 1 (equal weight)
        if steps is None or steps <= 0:
            steps = 1

        if weighted_sum is None:
            weighted_sum = avg_vec * float(steps)
        else:
            # ensure same shape
            if avg_vec.numel() != weighted_sum.numel():
                raise RuntimeError(f"Shape mismatch: {avg_path} has shape {avg_vec.shape} but others differ.")
            weighted_sum += avg_vec * float(steps)

        total_steps += steps
        loaded_count += 1

        # norms
        norms = e.get("norms")
        if norms is not None:
            try:
                concatenated_norms.append(np.asarray(norms, dtype=np.float32))
            except Exception:
                pass

        # step meta
        meta = e.get("meta")
        if meta:
            merged_step_meta.extend(meta)

        # trace_cov approximate
        if stats and ("trace_covariance" in stats):
            try:
                trace_cov_list.append(float(stats["trace_covariance"]))
                trace_cov_weights.append(float(steps))
            except Exception:
                pass

    if loaded_count == 0:
        print("[Error] No valid avg_grad loaded. Exiting.")
        return 2

    # compute merged average
    merged_avg = (weighted_sum / float(total_steps)).to(torch.float32)

    # save merged avg
    merged_avg_path = merged_dir / "avg_grad.pt"
    torch.save(merged_avg, merged_avg_path)
    print(f"[Info] Saved merged avg_grad to {merged_avg_path}")

    # concat norms and save
    if len(concatenated_norms) > 0:
        all_norms = np.concatenate(concatenated_norms, axis=0)
        np.save(merged_dir / "norms.npy", all_norms)
    else:
        all_norms = np.array([], dtype=np.float32)

    # compute merged stats
    merged_stats = {
        "merged_from": [str(e["dir"]) for e in avg_entries],
        "num_sources": len(avg_entries),
        "total_steps": int(total_steps),
        "avg_grad_shape": list(merged_avg.shape),
        "avg_grad_norm": float(torch.norm(merged_avg).item()),
    }

    if all_norms.size > 0:
        merged_stats["grad_norm_mean"] = float(np.mean(all_norms).item())
        merged_stats["grad_norm_variance"] = float(np.var(all_norms).item())
    else:
        merged_stats["grad_norm_mean"] = None
        merged_stats["grad_norm_variance"] = None

    # approximate global trace_cov by weighted average of per-rank trace_cov (if available)
    if len(trace_cov_list) > 0:
        try:
            weighted_trace_cov = float(sum(w * v for w, v in zip(trace_cov_weights, trace_cov_list)) / sum(trace_cov_weights))
            merged_stats["trace_covariance_approx"] = weighted_trace_cov
            merged_stats["trace_covariance_approx_note"] = (
                "Approximate: per-rank trace_covariance were averaged weighted by processed_steps. "
                "Exact global trace covariance requires per-dimension second moment data which was not saved."
            )
        except Exception:
            merged_stats["trace_covariance_approx"] = None
    else:
        merged_stats["trace_covariance_approx"] = None

    # Save merged stats and merged step meta
    (merged_dir / "merged_stats.json").write_text(json.dumps(merged_stats, indent=2), encoding="utf-8")
    if merged_step_meta:
        (merged_dir / "merged_step_meta.json").write_text(json.dumps(merged_step_meta, indent=2), encoding="utf-8")

    print(f"[Info] Merged stats saved to {merged_dir / 'merged_stats.json'}")
    if all_norms.size > 0:
        print(f"[Info] Merged norms saved to {merged_dir / 'norms.npy'} (length {all_norms.shape[0]})")
    print("[Done] merge_gradients.py finished.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
