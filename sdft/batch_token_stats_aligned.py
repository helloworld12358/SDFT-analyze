#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
per_file_token_stats.py

每个文件单独统计 token，并只在该文件内部显示 top-N token（降序）。
输出（每个文件）:
 - <result_dir>/<dataset>/<basename>_log.txt   (log，包含 top-N_log)
 - <result_dir>/<dataset>/<basename>_token_topM.png  (图，只显示该文件内部 top-M_plot token，按该文件内部频率降序)

用法示例：
python3 per_file_token_stats.py \
  --data-file /path/to/alpaca_train.json \
  --model-path /path/to/Llama-2-7b-chat-hf \
  --result-dir /path/to/results/alpaca \
  --top-n-log 100 \
  --top-m-plot 100

注意：
 - 请确保已安装依赖：transformers matplotlib tqdm
 - 如服务器无 CJK 字体，可将 FORCE_FONT_PATH 指向 ttf/ttc 字体文件（Noto 等）
"""
from pathlib import Path
import argparse
import json
import csv
from collections import Counter
from typing import List, Any, Dict
import math
import sys
import traceback

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

# matplotlib headless backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# 如果你有字体文件 (.ttf/.ttc) 想强制使用以保证 CJK 显示，请填路径；否则留空。
FORCE_FONT_PATH = ""  # e.g. "fonts/NotoSansCJK-Regular.ttc"

# 尝试加载强制字体（优先）
if FORCE_FONT_PATH:
    try:
        fm.fontManager.addfont(FORCE_FONT_PATH)
        prop = fm.FontProperties(fname=FORCE_FONT_PATH)
        font_name = prop.get_name()
        matplotlib.rcParams['font.sans-serif'] = [font_name]
        matplotlib.rcParams['axes.unicode_minus'] = False
        print(f"[INFO] 强制加载字体文件: {FORCE_FONT_PATH} -> 字体名: {font_name}")
    except Exception as e:
        print(f"[WARN] 无法加载 FORCE_FONT_PATH 指定的字体：{e}. 将尝试自动查找系统字体.")

# 自动选择系统字体（如果未强制加载）
if not FORCE_FONT_PATH:
    preferred_fonts = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK KR",
        "Noto Sans CJK",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans"
    ]
    chosen_font = None
    for fname in preferred_fonts:
        try:
            path = fm.findfont(fname)
            prop = fm.FontProperties(fname=path)
            found_name = prop.get_name()
            if fname.replace(" ", "").lower() in found_name.replace(" ", "").lower() or \
               "noto" in found_name.lower() or "simhei" in found_name.lower() or \
               "microsoft" in found_name.lower() or "arial" in found_name.lower():
                chosen_font = found_name
                break
        except Exception:
            continue
    if chosen_font:
        matplotlib.rcParams['font.sans-serif'] = [chosen_font]
        matplotlib.rcParams['axes.unicode_minus'] = False
        print(f"[INFO] 已设置 matplotlib 字体为：{chosen_font}")
    else:
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        print("[WARN] 未找到首选 CJK 字体。将使用 DejaVu Sans（可能缺少 CJK 字符）。")

# transformers tokenizer
try:
    from transformers import AutoTokenizer
except Exception as e:
    print("ERROR: transformers 未安装或加载失败。请安装：pip install transformers")
    raise e

# ---------------- loaders & extractors ----------------
def load_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            for rec in data:
                yield rec
        elif isinstance(data, dict):
            for k in ('data', 'records', 'examples', 'items'):
                if k in data and isinstance(data[k], list):
                    for rec in data[k]:
                        yield rec
                    return
            yield data
        else:
            yield data

def load_csv(path: Path):
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

def detect_loader(path: Path):
    s = path.suffix.lower()
    if s == '.jsonl':
        return load_jsonl
    if s == '.json':
        return load_json
    if s in ('.csv', '.tsv'):
        return load_csv
    return load_jsonl

def extract_texts_from_record(rec: Any) -> List[str]:
    texts = []
    if rec is None:
        return texts
    if isinstance(rec, str):
        if rec.strip():
            return [rec.strip()]
        return []
    if isinstance(rec, list):
        for r in rec:
            texts.extend(extract_texts_from_record(r))
        return texts
    if isinstance(rec, dict):
        for key in ('output', 'answer', 'response', 'completion', 'text', 'content', 'final', 'target'):
            if key in rec and rec[key]:
                v = rec[key]
                if isinstance(v, str):
                    if v.strip():
                        texts.append(v.strip())
                else:
                    texts.extend(extract_texts_from_record(v))
        if 'choices' in rec and rec['choices']:
            for ch in rec['choices']:
                if isinstance(ch, dict):
                    if 'text' in ch and ch['text']:
                        texts.extend(extract_texts_from_record(ch['text']))
                    if 'message' in ch and isinstance(ch['message'], dict):
                        msg = ch['message']
                        if 'content' in msg and msg['content']:
                            texts.extend(extract_texts_from_record(msg['content']))
                    if 'delta' in ch and isinstance(ch['delta'], dict) and ch['delta'].get('content'):
                        texts.extend(extract_texts_from_record(ch['delta']['content']))
                else:
                    texts.extend(extract_texts_from_record(ch))
        for key in ('conversations', 'conversation', 'messages', 'dialog', 'chat'):
            if key in rec and rec[key]:
                texts.extend(extract_texts_from_record(rec[key]))
        for k, v in rec.items():
            if isinstance(v, dict):
                for subk in ('answer', 'output', 'text', 'content'):
                    if subk in v and v[subk]:
                        texts.extend(extract_texts_from_record(v[subk]))
        if not texts:
            for k, v in rec.items():
                if isinstance(v, str) and len(v.strip()) > 20:
                    texts.append(v.strip())
    return [t for t in (s.strip() for s in texts) if t]

# ---------------- per-file stats ----------------
def stats_for_file(filepath: Path, tokenizer) -> Dict[str, Any]:
    counter = Counter()
    total_texts = 0
    total_tokens = 0
    loader = detect_loader(filepath)
    for rec in tqdm(loader(filepath), desc=f"Parsing {filepath.name}"):
        texts = extract_texts_from_record(rec)
        for text in texts:
            total_texts += 1
            try:
                ids = tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                try:
                    enc = tokenizer(text, add_special_tokens=False)
                    ids = enc.get('input_ids', [])
                except Exception:
                    continue
            try:
                toks = tokenizer.convert_ids_to_tokens(ids)
            except Exception:
                toks = [str(i) for i in ids]
            toks = [t for t in toks if t is not None and str(t).strip() != '']
            counter.update(toks)
            total_tokens += len(toks)
    avg_tokens = (total_tokens / total_texts) if total_texts > 0 else 0.0
    return {
        'counter': counter,
        'total_texts': total_texts,
        'total_tokens': total_tokens,
        'avg_tokens_per_text': avg_tokens
    }

# ---------------- label sanitize & plotting ----------------
def sanitize_label(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace('\\', r'\\')
    s = s.replace('$', r'\$')
    return s

def save_log(result_dir: Path, data_file: Path, stats: Dict[str, Any], top_n_log: int, top_m_plot: int, warnings: List[str]):
    result_dir.mkdir(parents=True, exist_ok=True)
    basename = data_file.stem
    log_path = result_dir / f"{basename}_log.txt"
    with log_path.open('w', encoding='utf-8') as f:
        f.write("Token 统计（按文件内部排序）日志\n")
        f.write("================================\n\n")
        f.write(f"源文件: {data_file}\n")
        f.write(f"总提取到的文本段（句子/段落）数量: {stats['total_texts']}\n")
        f.write(f"所有句子的平均 token 长度: {stats['avg_tokens_per_text']:.4f}\n")
        f.write(f"所有文本的 token 总数: {stats['total_tokens']}\n\n")
        top_tokens = stats['counter'].most_common(top_n_log)
        f.write(f"出现次数最多的前 {top_n_log} 个 token（按该文件内部频率降序）：\n")
        for i, (tok, cnt) in enumerate(top_tokens, start=1):
            f.write(f"{i:3d}. {tok}    {cnt}\n")
        f.write("\n")
        f.write(f"绘图将展示该文件内部前 {top_m_plot} 个 token（按该文件内部频率降序）。\n")
        if warnings:
            f.write("\n警告/说明：\n")
            for w in warnings:
                f.write(f"- {w}\n")
    return str(log_path)

def plot_topm(result_dir: Path, data_file: Path, stats: Dict[str, Any], top_m_plot: int, outname_suffix: str = 'token_topM.png', warnings: List[str] = None, keep_title: bool = False):
    result_dir.mkdir(parents=True, exist_ok=True)
    basename = data_file.stem
    img_path = result_dir / f"{basename}_{outname_suffix}"
    counter = stats['counter']
    top_items = counter.most_common(top_m_plot)
    if not top_items:
        print(f"[WARN] 文件 {data_file} 无可绘制 token。")
        return None
    toks, counts = zip(*top_items)
    labels = [sanitize_label(t) for t in toks]
    n = len(labels)

    # figure size
    width_per_bar = 0.18
    fig_width = max(8, min(300, int(math.ceil(n * width_per_bar))))
    fig_height = 6

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.bar(range(n), counts)
    ax.set_xticks(range(n))
    # Label 横向显示（rotation=0），水平对齐为右侧减小错位
    ax.set_xticklabels(labels, rotation=0, fontsize=7, ha='right')
    ax.set_xlabel('token (该文件内部 top tokens)')
    ax.set_ylabel('出现次数')
    if keep_title:
        ax.set_title(f'Token freq for {basename} (top {n})')

    try:
        plt.tight_layout()
    except Exception as e:
        warn_msg = f"tight_layout 失败（文件 {data_file.name}），跳过 tight_layout 保存图片。错误: {e}"
        print("[WARN] " + warn_msg)
        if warnings is not None:
            warnings.append(warn_msg)
        traceback.print_exc()
    try:
        fig.savefig(str(img_path), dpi=300)
    except Exception as e:
        warn_msg = f"保存图片失败: {e}"
        print("[ERROR] " + warn_msg)
        if warnings is not None:
            warnings.append(warn_msg)
        raise
    finally:
        plt.close(fig)
    return str(img_path)


# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description="按文件统计 token 并只显示该文件内部 top tokens（降序）")
    parser.add_argument('--data-file', type=str, required=True, help='要处理的数据文件（jsonl/json/csv）')
    parser.add_argument('--model-path', type=str, required=True, help='本地 HF tokenizer/模型 目录')
    parser.add_argument('--result-dir', type=str, required=True, help='结果保存目录（会在此目录下写入 <basename>_log.txt 与 <basename>_token_topM.png）')
    parser.add_argument('--top-n-log', type=int, default=100, help='log 中列出的 top-N token（默认 100）')
    parser.add_argument('--top-m-plot', type=int, default=100, help='绘图中展示的 top-M token（默认 100）')
    args = parser.parse_args()

    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"[ERROR] 指定的数据文件不存在: {data_file}")
        sys.exit(2)

    result_dir = Path(args.result_dir)
    print(f"[INFO] 正在加载 tokenizer：{args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    warnings = []
    stats = stats_for_file(data_file, tokenizer)
    logpath = save_log(result_dir, data_file, stats, args.top_n_log, args.top_m_plot, warnings)
    imgpath = plot_topm(result_dir, data_file, stats, args.top_m_plot, warnings=warnings, keep_title=False)
    print(f"[INFO] 已保存 log: {logpath}")
    print(f"[INFO] 已保存图像: {imgpath}")
    if warnings:
        print("[WARN] 以下警告已写入 log：")
        for w in warnings:
            print(" -", w)
    print("[DONE]")

if __name__ == '__main__':
    main()
