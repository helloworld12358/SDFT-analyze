# File: DataInf/src/assemble_matrix.py
# Minor robustness: if out_csv not provided, default to result_dir/pairwise_matrix.csv

import argparse
import os
import json
import numpy as np
import csv

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grads_list", type=str, required=True, help="包含 grad 路径的文本文件（每行一个）")
    p.add_argument("--result_dir", type=str, required=True, help="保存 sim_i_j.json 的目录")
    p.add_argument("--out_csv", type=str, default=None, help="输出 CSV 路径（可选）")
    p.add_argument("--out_npy", type=str, default=None, help="输出 NPY 路径（可选）")
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.grads_list, "r", encoding="utf-8") as f:
        grads = [line.strip() for line in f if line.strip()]
    n = len(grads)
    M = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        for j in range(i, n):
            fname = os.path.join(args.result_dir, f"sim_{i}_{j}.json")
            if not os.path.exists(fname):
                fname = os.path.join(args.result_dir, f"sim_{j}_{i}.json")
            if os.path.exists(fname):
                with open(fname, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    score = obj.get("score", None)
                    if score is not None:
                        M[i, j] = float(score)
                        M[j, i] = float(score)

    if args.out_csv:
        out_csv = args.out_csv
    else:
        out_csv = os.path.join(args.result_dir, "pairwise_matrix.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    header = [""] + [os.path.basename(p) for p in grads]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n):
            row = [os.path.basename(grads[i])] + [("" if np.isnan(x) else f"{x:.8e}") for x in M[i]]
            writer.writerow(row)

    if args.out_npy:
        np.save(args.out_npy, M)

    print("Matrix assembled. CSV:", out_csv)
    if args.out_npy:
        print("Numpy saved:", args.out_npy)

if __name__ == "__main__":
    main()
