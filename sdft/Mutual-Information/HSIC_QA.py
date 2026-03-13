#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨互信息计算脚本（HSIC）：
- A：从 data/<ds>/distilled_<ds>.json 提取 “output” 或 “answer”
- B：从 predictions/<ds>/sdft/inference_sdft/generated_predictions.jsonl 提取 “predict”
- C：从 predictions/<ds>/sft/inference_sft/generated_predictions.jsonl 提取 “predict”
计算 HSIC(A, B) 和 HSIC(A, C)，并将结果写入与原脚本相同的日志目录。
"""

import os
import sys
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from mi_estimators import hsic_gaussian

# —— 固定配置 ——
MODEL_PATH = "/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/model/Llama-2-7b-chat-hf"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

def ensure_results_dir(ds):
    d = os.path.join(os.path.dirname(__file__), "results", "hsic", ds)
    os.makedirs(d, exist_ok=True)
    return d

def load_distilled_answers(fp):
    """从 JSON 文件（数组）中提取 'output' 或 'answer' 文本列表"""
    seq = []
    with open(fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for obj in data:
            if 'output' in obj:
                seq.append(obj['output'])
            elif 'answer' in obj:
                seq.append(obj['answer'])
    if not seq:
        raise RuntimeError(f"{fp} 中未找到 'output' 或 'answer' 字段")
    return seq

def load_prediction_texts(fp):
    """从 JSONL 文件中提取 'predict' 文本列表"""
    seq = []
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if 'predict' not in obj:
                raise KeyError(f"{fp} 缺少 'predict' 字段: {obj}")
            seq.append(obj['predict'])
    if not seq:
        raise RuntimeError(f"{fp} 中未加载到任何 'predict'")
    return seq

def embed_sentences(sentences):
    """Mean‐pooling 句子列表 → 返回 numpy([N, D])"""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, cache_dir=MODEL_PATH, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModel.from_pretrained(
        MODEL_PATH, cache_dir=MODEL_PATH, local_files_only=True
    )
    if DEVICE.startswith("cuda") and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(DEVICE).eval()

    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i : i + BATCH_SIZE]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            ids  = enc["input_ids"].to(DEVICE)
            mask = enc["attention_mask"].to(DEVICE).unsqueeze(-1).float()

            out = model(input_ids=ids, attention_mask=mask.squeeze(-1))
            hidden = out.last_hidden_state

            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            emb = summed / counts
            all_embeds.append(emb.cpu())

    return torch.cat(all_embeds, dim=0).numpy()

def compute_hsic(X, Y):
    """调用高斯核 HSIC 估计器"""
    return hsic_gaussian(X, Y, ktype="gaussian").item()

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dataset_name>", file=sys.stderr)
        sys.exit(1)
    ds = sys.argv[1]

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    # A: distilled answers
    file_a = os.path.join(base, "data", ds, f"distilled_{ds}.json")
    # B: SD-FT predictions
    file_b = os.path.join(base, "predictions", ds, "sdft", "inference_sdft", "generated_predictions.jsonl")
    # C: SFT predictions
    file_c = os.path.join(base, "predictions", ds, "sft",  "inference_sft",  "generated_predictions.jsonl")

    # 加载三组文本
    A = load_distilled_answers(file_a)
    B = load_prediction_texts(file_b)
    C = load_prediction_texts(file_c)
    print(f"[{ds}] Loaded A={len(A)} distilled answers, B={len(B)} sdft preds, C={len(C)} sft preds.")

    # 嵌入
    emb_A = embed_sentences(A)
    emb_B = embed_sentences(B)
    emb_C = embed_sentences(C)

    # 计算 HSIC
    hsic_AB = compute_hsic(emb_A, emb_B)
    hsic_AC = compute_hsic(emb_A, emb_C)
    print(f"[{ds}] HSIC(A,B) = {hsic_AB:.6f}")
    print(f"[{ds}] HSIC(A,C) = {hsic_AC:.6f}")

    # 写日志
    results_dir = ensure_results_dir(ds)
    log_path = os.path.join(results_dir, f"{ds}_hsic_ab_ac.log")
    with open(log_path, "w", encoding="utf-8") as fout:
        fout.write(f"Dataset: {ds}\n")
        fout.write(f"A = distilled answers ({file_a})\n")
        fout.write(f"B = sdft predictions ({file_b})\n")
        fout.write(f"C = sft predictions  ({file_c})\n\n")
        fout.write(f"HSIC(A, B) = {hsic_AB:.6f}\n")
        fout.write(f"HSIC(A, C) = {hsic_AC:.6f}\n")

    print(f"Results saved to {log_path}")

if __name__ == "__main__":
    main()

