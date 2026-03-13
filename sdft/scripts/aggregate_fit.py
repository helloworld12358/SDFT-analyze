#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_and_fit.py (minimal change: add numeric diff table)

修改点（严格最小修改）：
- 在输出中额外添加一张表：数值差值表 (sft - sdft)
- 其余逻辑、文件名规则和行为保持不变
"""
from __future__ import annotations
import os
import re
import json
import math
import time
from typing import List, Any, Optional, Tuple, Dict

# =======================
# 可调整的全局参数
# =======================
DEFAULT_W_START = 100
DEFAULT_W_DOT = 100

# 直接在此指定你要使用的拟合权重（在程序内一开始指定）
SPECIFIED_W_START = 40000.0  # <-- 在这里改为你想要的 w_start
SPECIFIED_W_DOT = 5.0    # <-- 在这里改为你想要的 w_dot

# 拟合约束：w_start >= ratio * w_dot
RATIO_WSTART_OVER_WDOT = 10.0

# 搜索配置（保留未用到的常量以最小改动保留原结构）
SEARCH_MAX_EXP = 3.0
LOG_STEP_SIZE = 0.0001

# 使用的训练域（列）与测试键（行集合）
TRAIN_DOMAINS: List[str] = [
    "gsm8k", "openfunction", "magicoder", "alpaca", "dolly", "lima", "openhermes"
]

TEST_KEYS: List[str] = [
    "gsm8k", "openfunction", "humaneval", "multiarith", "alpaca_eval"
]

FIT_TEST_ORDER = ["gsm8k", "multiarith", "openfunction", "humaneval"]

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


def build_tables_with_components(results_dir: str):
    rows = len(TEST_KEYS)
    cols = len(TRAIN_DOMAINS)

    metric_names = ["dot_before", "dot_after", "c_start", "c_end", "v_align"]
    def empty_metrics():
        return {m: build_empty_matrix(rows, cols, None) for m in metric_names}

    sft = empty_metrics()
    sdft = empty_metrics()
    sdft_total = build_empty_matrix(rows, cols, None)
    sft_total = build_empty_matrix(rows, cols, None)

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

        v_align = safe_float(obj.get("V_align"))
        c_start = safe_float(obj.get("C_start"))
        c_end = safe_float(obj.get("C_end"))
        dot_before = safe_float(obj.get("dot_avggrad_delta_before"))
        dot_after = safe_float(obj.get("dot_avggrad_delta"))

        s_v = 0.0 if v_align is None else v_align
        s_s = 0.0 if c_start is None else c_start
        s_e = 0.0 if c_end is None else c_end
        s_db = 0.0 if dot_before is None else dot_before
        s_da = 0.0 if dot_after is None else dot_after

        # 默认值展示，稍后会被拟合值覆盖
        total_default = (DEFAULT_W_DOT * s_db) - s_da + (DEFAULT_W_START * s_s) + s_e

        target = sdft if ckpt == "sdft" else sft
        target["dot_before"][row_idx][col_idx] = s_db
        target["dot_after"][row_idx][col_idx] = s_da
        target["c_start"][row_idx][col_idx] = s_s
        target["c_end"][row_idx][col_idx] = s_e
        target["v_align"][row_idx][col_idx] = s_v

        if ckpt == "sdft":
            sdft_total[row_idx][col_idx] = total_default
        elif ckpt == "sft":
            sft_total[row_idx][col_idx] = total_default

    diff_mat = build_empty_matrix(rows, cols, None)
    for i in range(rows):
        for j in range(cols):
            a = sft_total[i][j]
            b = sdft_total[i][j]
            if a is None and b is None:
                diff = None
            else:
                aa = 0.0 if a is None else a
                bb = 0.0 if b is None else b
                diff = aa - bb
            diff_mat[i][j] = diff

    return sdft, sft, diff_mat


USER_BRACKET_TABLE = {
    "gsm8k":     [2.80, 3.89, -0.90, 0.61],
    "openfunction":[-0.40, 8.89, 4.47, 4.26],
    "magicoder": [0.68, 4.44, -3.57, -0.61],
    "alpaca":    [9.69, 35.00, -0.89, -1.83],
    "dolly":     [6.58, 34.44, 11.6, 1.22],
    "lima":      [0.83, -0.55, 7.14, 0.61],
    "openhermes":[1.21, 13.33, -3.57, 0.00],
}

USER_LABELS_SIGN: Dict[Tuple[str, str], int] = {}
for train_dom, col_vals in USER_BRACKET_TABLE.items():
    for idx, val in enumerate(col_vals):
        test_key = FIT_TEST_ORDER[idx]
        sign = 1 if (val >= 0.0) else -1
        USER_LABELS_SIGN[(train_dom, test_key)] = sign


def prepare_fitting_dataset(sdft_metrics: dict, sft_metrics: dict):
    # 将数据展平为简单的列表，以提高遍历速度
    # A list of (dx1, dx2, dbias, label)
    dataset = []
    
    for test_key in FIT_TEST_ORDER:
        if test_key not in TEST_KEYS:
            continue
        row_idx = TEST_KEYS.index(test_key)
        for col_idx, train_dom in enumerate(TRAIN_DOMAINS):
            db_before_sft = sft_metrics["dot_before"][row_idx][col_idx] or 0.0
            db_before_sdft = sdft_metrics["dot_before"][row_idx][col_idx] or 0.0
            da_sft = sft_metrics["dot_after"][row_idx][col_idx] or 0.0
            da_sdft = sdft_metrics["dot_after"][row_idx][col_idx] or 0.0
            cs_sft = sft_metrics["c_start"][row_idx][col_idx] or 0.0
            cs_sdft = sdft_metrics["c_start"][row_idx][col_idx] or 0.0
            ce_sft = sft_metrics["c_end"][row_idx][col_idx] or 0.0
            ce_sdft = sdft_metrics["c_end"][row_idx][col_idx] or 0.0

            dx1 = db_before_sft - db_before_sdft
            dx2 = cs_sft - cs_sdft
            dbias = ((-da_sft + ce_sft) - (-da_sdft + ce_sdft))

            label_key = (train_dom, test_key)
            if label_key not in USER_LABELS_SIGN:
                continue
            label = USER_LABELS_SIGN[label_key]
            
            dataset.append((dx1, dx2, dbias, label))
            
    return dataset


def compute_accuracy_from_dataset(dataset, w_start, w_dot):
    n = len(dataset)
    if n == 0:
        return 0.0
    correct = 0
    for (dx1, dx2, dbias, label) in dataset:
        score = w_dot * dx1 + w_start * dx2 + dbias
        pred_sign = 1 if score >= 0.0 else -1
        if pred_sign == label:
            correct += 1
    return correct / n


def optimize_weights_massive_grid(dataset, 
                                  ratio: float = RATIO_WSTART_OVER_WDOT,
                                  max_exp: float = SEARCH_MAX_EXP,
                                  step: float = LOG_STEP_SIZE):
    """
    原有的网格搜索函数（在此保留但不再被调用，保持代码结构最小修改）
    """
    
    print(f"Starting Massive Grid Search...")
    print(f"  Range: 10^0 to 10^{max_exp}")
    print(f"  Step Size (log10): {step}")
    print(f"  Ratio constraint: w_start >= {ratio} * w_dot")
    
    log_ratio = math.log10(ratio)
    
    best_acc = -1.0
    best_margin = -1.0
    best_w_start = DEFAULT_W_START
    best_w_dot = DEFAULT_W_DOT
    
    num_steps = int(max_exp / step)
    
    start_time = time.time()
    last_print = start_time
    total_checks = 0
    
    n_samples = len(dataset)
    d_x1 = [d[0] for d in dataset]
    d_x2 = [d[1] for d in dataset]
    d_bias = [d[2] for d in dataset]
    d_lbl = [d[3] for d in dataset]
    
    for i in range(num_steps + 1):
        exp_dot = i * step
        w_dot = math.pow(10, exp_dot)
        start_exp_start = exp_dot + log_ratio
        if start_exp_start > max_exp:
            continue
        start_j = int(math.ceil(start_exp_start / step))
        for j in range(start_j, num_steps + 1):
            exp_start = j * step
            w_start = math.pow(10, exp_start)
            correct = 0
            for k in range(n_samples):
                score = w_dot * d_x1[k] + w_start * d_x2[k] + d_bias[k]
                pred_sign = 1 if score >= 0.0 else -1
                if pred_sign == d_lbl[k]:
                    correct += 1
            acc = correct / n_samples
            if acc > best_acc:
                best_acc = acc
                best_w_start = w_start
                best_w_dot = w_dot
                min_abs_score = float('inf')
                for k in range(n_samples):
                    score = w_dot * d_x1[k] + w_start * d_x2[k] + d_bias[k]
                    min_abs_score = min(min_abs_score, abs(score))
                best_margin = min_abs_score
            elif acc == best_acc:
                min_abs_score = float('inf')
                for k in range(n_samples):
                    score = w_dot * d_x1[k] + w_start * d_x2[k] + d_bias[k]
                    min_abs_score = min(min_abs_score, abs(score))
                if min_abs_score > best_margin:
                    best_margin = min_abs_score
                    best_w_start = w_start
                    best_w_dot = w_dot
            total_checks += 1
        curr_time = time.time()
        if curr_time - last_print > 5.0:
            progress = (i / num_steps) * 100
            elapsed = curr_time - start_time
            print(f"  Progress: {progress:.1f}% | Elapsed: {elapsed:.0f}s | Checks: {total_checks} | Best Acc: {best_acc:.4f}")
            last_print = curr_time

    elapsed = time.time() - start_time
    print(f"Grid Search Finished in {elapsed:.1f}s. Total checks: {total_checks}")
    return best_w_start, best_w_dot, best_acc


def compute_and_save_outputs(results_dir: str, out_basename_prefix: str = "aggregate_fitted"):
    sdft_metrics, sft_metrics, diff_mat = build_tables_with_components(results_dir)
    dataset = prepare_fitting_dataset(sdft_metrics, sft_metrics)

    if len(dataset) == 0:
        raise RuntimeError("没有可用于拟合的样本（samples 为空）。请检查 results_dir 与用户标签映射。")

    # 直接使用用户在文件开头指定的参数（最小改动：不再搜索）
    best_w_start = float(SPECIFIED_W_START)
    best_w_dot = float(SPECIFIED_W_DOT)
    best_acc = compute_accuracy_from_dataset(dataset, best_w_start, best_w_dot)

    # 计算最终表格
    rows = len(TEST_KEYS)
    cols = len(TRAIN_DOMAINS)
    sdft_table = build_empty_matrix(rows, cols, None)
    sft_table = build_empty_matrix(rows, cols, None)
    pred_sign_table = build_empty_matrix(rows, cols, "<missing>")
    diff_value_table = build_empty_matrix(rows, cols, None)  # 新增：数值差值表 (sft - sdft)

    for i in range(rows):
        for j in range(cols):
            s_db = sdft_metrics["dot_before"][i][j] or 0.0
            s_da = sdft_metrics["dot_after"][i][j] or 0.0
            s_cs = sdft_metrics["c_start"][i][j] or 0.0
            s_ce = sdft_metrics["c_end"][i][j] or 0.0

            t_db = sft_metrics["dot_before"][i][j] or 0.0
            t_da = sft_metrics["dot_after"][i][j] or 0.0
            t_cs = sft_metrics["c_start"][i][j] or 0.0
            t_ce = sft_metrics["c_end"][i][j] or 0.0

            sdft_val = (best_w_dot * s_db) - s_da + (best_w_start * s_cs) + s_ce
            sft_val = (best_w_dot * t_db) - t_da + (best_w_start * t_cs) + t_ce

            sdft_table[i][j] = sdft_val
            sft_table[i][j] = sft_val

            diff_val = sft_val - sdft_val
            pred_sign_table[i][j] = "+" if diff_val >= 0.0 else "-"
            diff_value_table[i][j] = diff_val  # 填充数值差值表

    # 保存
    w1_str = f"{best_w_start:.6f}".replace(".", "p")
    w2_str = f"{best_w_dot:.6f}".replace(".", "p")
    out_name = f"{out_basename_prefix}_wstart_{w1_str}_wdot_{w2_str}.txt"
    out_path = os.path.join(results_dir, out_name)

    prec = 6
    def compute_col_widths_for_tables(matrices, rows_labels, cols_labels, prec=6):
        first_col_width = max(max(len(lbl) for lbl in rows_labels), len("test_key"))
        col_widths = [first_col_width]
        for col in range(len(cols_labels)):
            maxw = len(cols_labels[col])
            for r in range(len(rows_labels)):
                for mat in matrices:
                    v = mat[r][col]
                    disp = "<missing>" if v is None else (v if isinstance(v, str) else f"{v:.{prec}f}")
                    maxw = max(maxw, len(disp))
            col_widths.append(maxw)
        return col_widths

    # 把 diff_value_table 一并加入以确定列宽
    matrices_for_width = [sdft_table, sft_table, pred_sign_table, diff_value_table]
    col_widths = compute_col_widths_for_tables(matrices_for_width, TEST_KEYS, TRAIN_DOMAINS, prec=prec)

    def render_table_generic(mat, title, rows_labels, cols_labels, col_widths, prec=6, is_symbol_table=False):
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
                if v is None:
                    cell = "<missing>".rjust(col_widths[j+1])
                else:
                    if is_symbol_table:
                        cell = str(v).rjust(col_widths[j+1])
                    else:
                        cell = f"{v:.{prec}f}".rjust(col_widths[j+1])
                cells.append(cell)
            parts.append(" ".join(cells))
        return "\n".join(parts)

    sdft_txt = render_table_generic(sdft_table, f"sdft (fitted w_start={best_w_start:.6f}, w_dot={best_w_dot:.6f})", TEST_KEYS, TRAIN_DOMAINS, col_widths, prec=prec, is_symbol_table=False)
    sft_txt = render_table_generic(sft_table, f"sft  (fitted w_start={best_w_start:.6f}, w_dot={best_w_dot:.6f})", TEST_KEYS, TRAIN_DOMAINS, col_widths, prec=prec, is_symbol_table=False)
    pred_txt = render_table_generic(pred_sign_table, f"predicted_signs (sft - sdft)", TEST_KEYS, TRAIN_DOMAINS, col_widths, prec=prec, is_symbol_table=True)
    diff_txt = render_table_generic(diff_value_table, f"diff_values (sft - sdft)  [numeric]", TEST_KEYS, TRAIN_DOMAINS, col_widths, prec=prec, is_symbol_table=False)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Fitted weights (User-specified): w_start={best_w_start:.6f}, w_dot={best_w_dot:.6f}\n")
        f.write(f"# Fitting accuracy: {best_acc:.6f}\n")
        f.write(f"# Parameters were taken from SPECIFIED_W_START / SPECIFIED_W_DOT in source (no grid search performed)\n\n")
        f.write(sdft_txt)
        f.write("\n\n")
        f.write(sft_txt)
        f.write("\n\n")
        f.write(pred_txt)
        f.write("\n\n")
        f.write(diff_txt)
        f.write("\n")

    return out_path, best_w_start, best_w_dot, best_acc


def main():
    results_dir = default_results_dir()
    if not os.path.isdir(results_dir):
        raise RuntimeError(f"results_dir not found: {results_dir}")

    out_path, w_start, w_dot, acc = compute_and_save_outputs(results_dir)
    print(out_path)


if __name__ == "__main__":
    main()