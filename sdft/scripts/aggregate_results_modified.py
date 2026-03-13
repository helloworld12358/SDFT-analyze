#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import re
import json
from typing import List, Any, Optional, Tuple

# =======================
# 仅修改这里即可
# =======================
C_START_WEIGHT = 40000  # ← 改成 100 或其他值
C_DOT_WEIGHT = 40    # ← 新增：dot_avggrad_delta_before 的权重（可调整）
# =======================


TRAIN_DOMAINS: List[str] = [
    "gsm8k", "openfunction", "magicoder", "alpaca", "dolly", "lima", "openhermes"
]

TEST_KEYS: List[str] = [
    "gsm8k", "openfunction", "humaneval", "multiarith", "alpaca_eval"
]

FILENAME_RE = re.compile(
    r"^(?P<domain>[^_]+)_(?P<ckpt>[^_]+)__(?P<test>.+)\.json$",
    flags=re.IGNORECASE
)

def default_results_dir() -> str:
    current_file = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, "experiment_results")

def load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def build_empty_matrix(rows: int, cols: int, fill=None):
    return [[fill for _ in range(cols)] for _ in range(rows)]

def build_tables(results_dir: str):
    rows = len(TEST_KEYS)
    cols = len(TRAIN_DOMAINS)

    sdft_mat = build_empty_matrix(rows, cols, None)
    sft_mat = build_empty_matrix(rows, cols, None)

    if not os.path.isdir(results_dir):
        raise RuntimeError(f"results_dir not found: {results_dir}")

    for fname in sorted(os.listdir(results_dir)):
        m = FILENAME_RE.match(fname)
        if not m:
            continue

        domain = m.group("domain").lower()
        ckpt = m.group("ckpt").lower()
        test_key = m.group("test").lower()

        if domain not in TRAIN_DOMAINS:
            continue
        if test_key not in TEST_KEYS:
            continue

        col_idx = TRAIN_DOMAINS.index(domain)
        row_idx = TEST_KEYS.index(test_key)

        fullpath = os.path.join(results_dir, fname)
        obj = load_json(fullpath)
        if obj is None:
            continue

        # old fields
        v_align = safe_float(obj.get("V_align"))
        c_start = safe_float(obj.get("C_start"))
        c_end = safe_float(obj.get("C_end"))

        # new/required dot fields
        dot_before = safe_float(obj.get("dot_avggrad_delta_before"))
        dot_after = safe_float(obj.get("dot_avggrad_delta"))

        # if none of the key indicators exist, skip
        if v_align is None and c_start is None and c_end is None and dot_before is None and dot_after is None:
            continue

        s_db = 0.0 if dot_before is None else dot_before
        s_da = 0.0 if dot_after is None else dot_after
        s_s = 0.0 if c_start is None else c_start
        s_e = 0.0 if c_end is None else c_end
        # V_align not used in the new total, but keep available if needed elsewhere
        s_v = 0.0 if v_align is None else v_align

        # new total: C_DOT_WEIGHT*dot_avggrad_delta_before - dot_avggrad_delta + C_START_WEIGHT * C_start - C_end
        total = (C_DOT_WEIGHT * s_db) - s_da + (C_START_WEIGHT * s_s) + s_e

        if ckpt == "sdft":
            sdft_mat[row_idx][col_idx] = total
        elif ckpt == "sft":
            sft_mat[row_idx][col_idx] = total

    diff_mat = build_empty_matrix(rows, cols, None)
    for i in range(rows):
        for j in range(cols):
            a = sft_mat[i][j]
            b = sdft_mat[i][j]
            if a is None and b is None:
                diff = None
            else:
                aa = 0.0 if a is None else a
                bb = 0.0 if b is None else b
                diff = aa - bb
            diff_mat[i][j] = diff

    return sdft_mat, sft_mat, diff_mat

def compute_col_widths(matrices, labels_rows, labels_cols, prec=6):
    first_col_width = max(max(len(lbl) for lbl in labels_rows), len("test_key"))
    col_widths = [first_col_width]

    for col in range(len(labels_cols)):
        maxw = len(labels_cols[col])
        for r in range(len(labels_rows)):
            for mat in matrices:
                v = mat[r][col]
                disp = "<missing>" if v is None else f"{v:.{prec}f}"
                maxw = max(maxw, len(disp))
        col_widths.append(maxw)

    return col_widths

def render_table(mat, title, rows_labels, cols_labels, col_widths, prec=6):
    parts = []
    parts.append(title)

    header = (
        f"{'test_key'.ljust(col_widths[0])} "
        + " ".join(cols_labels[c].rjust(col_widths[c+1]) for c in range(len(cols_labels)))
    )
    parts.append(header)
    parts.append("-" * (sum(col_widths) + len(col_widths)))

    for i, rlabel in enumerate(rows_labels):
        cells = [rlabel.ljust(col_widths[0])]
        for j in range(len(cols_labels)):
            v = mat[i][j]
            cell = ("<missing>" if v is None else f"{v:.{prec}f}").rjust(col_widths[j+1])
            cells.append(cell)
        parts.append(" ".join(cells))

    return "\n".join(parts)

def save_single_txt(results_dir: str, prec: int = 6) -> str:
    sdft_mat, sft_mat, diff_mat = build_tables(results_dir)

    col_widths = compute_col_widths(
        [sdft_mat, sft_mat, diff_mat],
        TEST_KEYS,
        TRAIN_DOMAINS,
        prec=prec
    )

    formula_str = f"{C_DOT_WEIGHT}*dot_avggrad_delta_before - dot_avggrad_delta + {C_START_WEIGHT}*C_start - C_end"

    sdft_txt = render_table(
        sdft_mat,
        f"sdft ({formula_str})",
        TEST_KEYS,
        TRAIN_DOMAINS,
        col_widths,
        prec=prec
    )

    sft_txt = render_table(
        sft_mat,
        f"sft  ({formula_str})",
        TEST_KEYS,
        TRAIN_DOMAINS,
        col_widths,
        prec=prec
    )

    diff_txt = render_table(
        diff_mat,
        "sft - sdft (difference)",
        TEST_KEYS,
        TRAIN_DOMAINS,
        col_widths,
        prec=prec
    )

    # 文件名包含两个权重参数，便于区分
    out_name = f"aggregate_single_txt_w{C_START_WEIGHT}_d{C_DOT_WEIGHT}.txt"
    out_path = os.path.join(results_dir, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(sdft_txt)
        f.write("\n\n")
        f.write(sft_txt)
        f.write("\n\n")
        f.write(diff_txt)
        f.write("\n")

    return out_path

def main():
    results_dir = default_results_dir()
    if not os.path.isdir(results_dir):
        raise RuntimeError(f"results_dir not found: {results_dir}")

    out_path = save_single_txt(results_dir, prec=6)
    print(out_path)

if __name__ == "__main__":
    main()