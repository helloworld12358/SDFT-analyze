#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两个分布微分熵差估计脚本（UMAP + KDE）：
1. 从两个 JSON/JSONL 文件中读取 'answer'、'output' 或 'predict' 文本
2. 用 Hugging Face Transformers 做 Mean Pooling 得到句子向量
3. 用 UMAP 将向量降维到固定维度列表 [10,15,20,25,30,50]
4. 对降维后的数据进行标准化
5. 用 KDE（默认 Scott 带宽）估计微分熵
6. 计算两组熵的差值 ΔH = H(P) - H(Q)
7. 将 (dim, H(P), H(Q), ΔH) 写入日志，保存在与输入文件路径对应的独立目录下
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import umap
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

# 默认模型路径
DEFAULT_MODEL_PATH = "/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/model/Llama-2-7b-chat-hf"
# 结果基础目录
RESULTS_BASE      = "/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/Mutual-Information/results/entropy_diff"
os.makedirs(RESULTS_BASE, exist_ok=True)


def load_json_data(fp):
    """读取 JSON or JSONL 文件，提取 'answer'、'output' 或 'predict' 字段文本列表"""
    texts = []
    with open(fp, 'r', encoding='utf-8') as f:
        first = f.read(1); f.seek(0)
        # JSON 数组格式
        if first.strip().startswith('['):
            arr = json.load(f)
            for obj in arr:
                txt = obj.get('answer') or obj.get('output') or obj.get('predict')
                if txt:
                    texts.append(txt)
        else:
            # JSONL 格式
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                txt = obj.get('answer') or obj.get('output') or obj.get('predict')
                if txt:
                    texts.append(txt)
    return texts


def embed_sentences(sentences, model_path, batch_size=16, max_length=4096):
    """Mean-pooling 句子列表，返回 numpy([N, D])"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(
        model_path, cache_dir=model_path, local_files_only=True, use_fast=True
    )
    if tok.pad_token is None:
        tok.pad_token, tok.pad_token_id = tok.eos_token, tok.eos_token_id
    model = AutoModel.from_pretrained(
        model_path, cache_dir=model_path, local_files_only=True
    )
    model.to(device).eval()

    embeds = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            ids   = enc['input_ids'].to(device)
            mask  = enc['attention_mask'].to(device).unsqueeze(-1).float()
            with autocast():
                out = model(input_ids=ids, attention_mask=mask.squeeze(-1))
            h      = out.last_hidden_state
            summed = (h * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            embeds.append((summed / counts).cpu())
            torch.cuda.empty_cache()
    return torch.cat(embeds, dim=0).numpy()


def make_output_dir(fp):
    """根据输入文件路径，生成唯一的输出目录名"""
    parts = fp.split(os.sep)
    if 'predictions' in parts:
        i = parts.index('predictions')
        subset = parts[i+1:i+4]  # e.g. ['alpaca','sdft','advbench-jailbreak']
        name = '_'.join(subset)
    else:
        name = os.path.splitext(os.path.basename(fp))[0]
    outdir = os.path.join(RESULTS_BASE, name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def main():
    parser = argparse.ArgumentParser(description="计算两分布的微分熵差 ΔH")
    parser.add_argument('--file_p', required=True, help="分布 P 的 JSON/JSONL 路径")
    parser.add_argument('--file_q', required=True, help="分布 Q 的 JSON/JSONL 路径")
    parser.add_argument('--batch_size', type=int, default=16, help="嵌入批大小")
    parser.add_argument('--max_length', type=int, default=4096, help="Tokenizer 最大截断长度")
    args = parser.parse_args()

    texts_p = load_json_data(args.file_p)
    texts_q = load_json_data(args.file_q)
    if not texts_p or not texts_q:
        print("任一文件未加载到有效文本，退出。", file=sys.stderr)
        sys.exit(1)

    emb_p = embed_sentences(texts_p, DEFAULT_MODEL_PATH, args.batch_size, args.max_length)
    emb_q = embed_sentences(texts_q, DEFAULT_MODEL_PATH, args.batch_size, args.max_length)

    dims = [10, 15, 20, 25, 30, 50]
    results = []
    for d in dims:
        reducer = umap.UMAP(n_components=d, metric='euclidean', random_state=42)
        red_p = reducer.fit_transform(emb_p)
        red_q = reducer.transform(emb_q)
        scaler = StandardScaler().fit(red_p)
        rp = scaler.transform(red_p)
        rq = scaler.transform(red_q)
        kde_p = gaussian_kde(rp.T)
        kde_q = gaussian_kde(rq.T)
        H_p = -np.mean(np.log(kde_p(rp.T) + 1e-12))
        H_q = -np.mean(np.log(kde_q(rq.T) + 1e-12))
        results.append((d, H_p, H_q, H_p - H_q))

    outdir  = make_output_dir(args.file_p)
    logfile = os.path.join(outdir, os.path.basename(outdir) + '_entropy_diff.log')
    with open(logfile, 'w', encoding='utf-8') as f:
        f.write(f"P 样本数: {len(texts_p)}, Q 样本数: {len(texts_q)}\n")
        f.write(f"{'dim':>4} | {'H(P)':>10} | {'H(Q)':>10} | {'ΔH':>10}\n")
        f.write("-"*44 + "\n")
        for d, Hp, Hq, dH in results:
            f.write(f"{d:4d} | {Hp:10.6f} | {Hq:10.6f} | {dH:10.6f}\n")

    print(f"结果已保存到 {logfile}")

if __name__ == "__main__":
    main()


