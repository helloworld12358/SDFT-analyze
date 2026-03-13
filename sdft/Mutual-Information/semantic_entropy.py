#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
语义熵计算脚本（统一欧几里得度量）：
- UMAP: metric='euclidean', n_neighbors=10, min_dist=0.1
- HDBSCAN: metric='euclidean', min_cluster_size=2
- 支持多卡并行（HuggingFace `device_map='auto'`），兼容单卡和 CPU
- 分别对 dims = [10,20,30,50,64,128,256,512] 做聚类，输出 entropy 与 noise ratio
兼容 Python 3.12
"""

import os
import json
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import umap
import hdbscan

# 路径配置
DEFAULT_MODEL_PATH = "/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/model/Llama-2-7b-chat-hf"
RESULTS_BASE = "/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/Mutual-Information/results/semantic_entropy"

os.makedirs(RESULTS_BASE, exist_ok=True)


def load_json_data(fp):
    seq = []
    with open(fp, 'r', encoding='utf-8') as f:
        first = f.read(1);
        f.seek(0)
        if first.strip().startswith('['):
            data = json.load(f)
            for obj in data:
                if 'answer' in obj:
                    seq.append(obj['answer'])
                elif 'output' in obj:
                    seq.append(obj['output'])
        else:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                if 'answer' in obj:
                    seq.append(obj['answer'])
                elif 'output' in obj:
                    seq.append(obj['output'])
    return seq


def embed_sentences(sentences, model_path, batch_size=16):
    # 设备映射，HuggingFace 自动在多 GPU/单 GPU/CPU 上分配
    model = AutoModel.from_pretrained(
        model_path,
        cache_dir=model_path,
        local_files_only=True,
        device_map='auto',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    # 获取 tokenizer
    tok = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=model_path,
        local_files_only=True,
        use_fast=True
    )
    if tok.pad_token is None:
        tok.add_special_tokens({'pad_token': tok.eos_token or '<|pad|>'})
    
    # 确定合理的 max_length
    raw_max = getattr(tok, 'model_max_length', None)
    max_length = 2048 if not isinstance(raw_max, int) or raw_max <= 0 or raw_max > 10000 else raw_max

    embeds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i: i + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            # 令 inputs 放到各自设备
            for k, v in enc.items():
                enc[k] = v.to(model.device)
            out = model(**enc).last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1).float()
            summed = (out * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            embeds.append((summed / counts).cpu())

    return torch.cat(embeds, dim=0).numpy()


def compute_metrics(emb, dim):
    reducer = umap.UMAP(
        n_components=dim,
        n_neighbors=10,
        min_dist=0.1,
        metric='euclidean',
        random_state=42
    )
    red = reducer.fit_transform(emb)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        metric='euclidean'
    )
    labels = clusterer.fit_predict(red)
    vals, cnts = np.unique(labels.astype(int), return_counts=True)
    tot = cnts.sum()
    noise = cnts[vals == -1].sum() if -1 in vals else 0
    noise_ratio = noise / tot
    probs = cnts / tot
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return entropy, noise_ratio


def parse_output_path(fp):
    ds = os.path.basename(os.path.dirname(fp))
    fn = os.path.splitext(os.path.basename(fp))[0]
    if '_train' in fn:
        split = 'train'
    elif fn.startswith('distilled_'):
        split = 'distilled'
    else:
        parts = fp.split(os.sep)
        split = parts[parts.index('predictions')+2] if 'predictions' in parts and len(parts) > parts.index('predictions')+2 else 'other'
    od = os.path.join(RESULTS_BASE, ds, split)
    os.makedirs(od, exist_ok=True)
    return os.path.join(od, f"{fn}.log")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    seq = load_json_data(args.json_file)
    if not seq:
        raise RuntimeError("在输入文件中未找到 'answer' 或 'output' 字段。")
    emb = embed_sentences(seq, DEFAULT_MODEL_PATH, batch_size=args.batch_size)

    dims = [10,20,30,50,64,128,256,512]
    results = {}
    for d in dims:
        ent, nr = compute_metrics(emb, d)
        results[d] = (ent, nr)

    lf = parse_output_path(args.json_file)
    with open(lf, 'w', encoding='utf-8') as f:
        f.write(f"样本总数: {len(seq)}\n")
        f.write("UMAP 参数: n_neighbors=10, min_dist=0.1, metric='euclidean'\n")
        f.write("HDBSCAN 参数: min_cluster_size=2, metric='euclidean'\n\n")
        for d, (ent, nr) in results.items():
            f.write(f"UMAP dim = {d}\n")
            f.write(f"  语义熵: {ent:.4f}\n")
            f.write(f"  噪声比例: {nr:.2%}\n\n")

    print(f"结果已保存到 {lf}")
    print("UMAP(euclidean), HDBSCAN(euclidean), min_cluster_size=2, n_neighbors=10")
    for d, (ent, nr) in results.items():
        print(f"dim={d}: entropy={ent:.4f}, noise={nr:.2%}")


if __name__ == '__main__':
    main()




