#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两个分布微分熵差估计脚本（UMAP + KDE）——仅兼容 predictions/.../*.jsonl：
1. 从两个 JSONL 文件中读取 'predict' 文本
2. 用 Hugging Face Transformers 做 Mean Pooling 得到句子向量
3. 用 UMAP 将向量降维到 [10,15,20,25,30,50]
4. 标准化后用 KDE（Scott 带宽）估计微分熵
5. 计算 ΔH = H(P) - H(Q)
6. 将结果写入：
     RESULTS_BASE/<dataset>/<testset>/<testset>_entropy_diff.log
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModel
import umap
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

# 模型 & 结果基础路径
DEFAULT_MODEL_PATH = "/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/model/Llama-2-7b-chat-hf"
RESULTS_BASE      = "/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/Mutual-Information/results/entropy_diff"
os.makedirs(RESULTS_BASE, exist_ok=True)

def load_predicts(fp):
    """只从 JSONL 里提取 'predict' 字段文本"""
    texts = []
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except:
                continue
            p = obj.get('predict')
            if p:
                texts.append(p)
    return texts

def embed_sentences(sentences, model_path, batch_size=16, max_length=1024):
    """Mean-pooling，返回 numpy([N,D])"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()

    embeds = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            enc   = tok(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt')
            ids   = enc.input_ids.to(device)
            mask  = enc.attention_mask.to(device).unsqueeze(-1).float()
            with autocast():
                out = model(input_ids=ids, attention_mask=mask.squeeze(-1))
            h      = out.last_hidden_state
            summed = (h * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            embeds.append((summed / counts).cpu())
            torch.cuda.empty_cache()
    return torch.cat(embeds, 0).numpy()

def make_output_dir(fp):
    """
    解析 predictions/.../<dataset>/<mode>/<testset>/... 路径，
    返回 RESULTS_BASE/<dataset>/<testset>/
    """
    parts   = fp.split(os.sep)
    i_pred  = parts.index('predictions')
    dataset = parts[i_pred+1]
    testset = parts[i_pred+3] if len(parts) > i_pred+3 else 'unknown'
    outdir  = os.path.join(RESULTS_BASE, dataset, testset)
    os.makedirs(outdir, exist_ok=True)
    return outdir, testset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_p', required=True, help="P 分布 JSONL")
    parser.add_argument('--file_q', required=True, help="Q 分布 JSONL")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=4096)
    args = parser.parse_args()

    P_texts = load_predicts(args.file_p)
    Q_texts = load_predicts(args.file_q)
    if not P_texts or not Q_texts:
        print("ERROR: P/Q 文本为空，退出", file=sys.stderr)
        sys.exit(1)

    emb_p = embed_sentences(P_texts, DEFAULT_MODEL_PATH,
                            args.batch_size, args.max_length)
    emb_q = embed_sentences(Q_texts, DEFAULT_MODEL_PATH,
                            args.batch_size, args.max_length)

    dims = [10,15,20,25,30,50]
    stats = []
    for d in dims:
        reducer = umap.UMAP(n_components=d, random_state=42)
        rp = reducer.fit_transform(emb_p)
        rq = reducer.transform(emb_q)
        sc = StandardScaler().fit(rp)
        rp, rq = sc.transform(rp), sc.transform(rq)
        kde_p = gaussian_kde(rp.T); kde_q = gaussian_kde(rq.T)
        H_p = -np.mean(np.log(kde_p(rp.T)+1e-12))
        H_q = -np.mean(np.log(kde_q(rq.T)+1e-12))
        stats.append((d, H_p, H_q, H_p-H_q))

    outdir, testset = make_output_dir(args.file_p)
    logfile = os.path.join(outdir, f"{testset}_entropy_diff.log")
    with open(logfile, 'w', encoding='utf-8') as f:
        f.write(f"P 样本数: {len(P_texts)}, Q 样本数: {len(Q_texts)}\n")
        f.write(" dim |     H(P) |     H(Q) |      ΔH\n")
        f.write("-"*40 + "\n")
        for d, Hp, Hq, dH in stats:
            f.write(f"{d:4d} | {Hp:8.4f} | {Hq:8.4f} | {dH:8.4f}\n")

    print("结果保存到:", logfile)

if __name__ == "__main__":
    main()
