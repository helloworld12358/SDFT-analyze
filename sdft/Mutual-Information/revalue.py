#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

def load_json_data(file_path):
    """从 JSONL 文件中读取 'label' 与 'predict' 文本列表"""
    labels, predicts = [], []
    with open(file_path, 'r') as fin:
        for line in fin:
            obj = json.loads(line)
            if 'label' not in obj or 'predict' not in obj:
                raise KeyError(f"{file_path} 缺少 'label' 或 'predict': {obj}")
            labels.append(obj['label'])
            predicts.append(obj['predict'])
    return labels, predicts

def embed_sentences(sentences):
    """Mean‑pooling 句子列表 → 返回 numpy([N,D])"""
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
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            ids  = enc["input_ids"].to(DEVICE)
            mask = enc["attention_mask"].to(DEVICE).unsqueeze(-1).float()

            out = model(input_ids=ids, attention_mask=enc["attention_mask"].to(DEVICE))
            hidden = out.last_hidden_state

            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            emb = summed / counts
            all_embeds.append(emb.cpu())

    return torch.cat(all_embeds, dim=0).numpy()

def compute_hsic_for_file(path, tag):
    labels, preds = load_json_data(path)
    print(f"[{tag}] {len(labels)} samples → embedding …")
    X = embed_sentences(labels)
    Y = embed_sentences(preds)
    val = hsic_gaussian(X, Y, ktype="gaussian").item()
    print(f"[{tag}] HSIC = {val:.6f}\n")
    return val

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dataset_name>")
        sys.exit(1)
    ds = sys.argv[1]

    # base 指向 sdft/ 目录，而非 Mutual-Information
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    file_sd = os.path.join(base, "predictions", ds, "sdft", "inference_sdft", "generated_predictions.jsonl")
    file_sf = os.path.join(base, "predictions", ds, "sft",  "inference_sft",  "generated_predictions.jsonl")

    hsic_sd = compute_hsic_for_file(file_sd,  f"{ds} Distilled↔SD‑FT")
    hsic_sf = compute_hsic_for_file(file_sf,  f"{ds} Train↔SFT")

    results_dir = ensure_results_dir(ds)
    log_path = os.path.join(results_dir, f"{ds}_hsic.log")
    with open(log_path, "w") as fout:
        fout.write(f"Dataset: {ds}\n")
        fout.write(f"Distilled_vs_SD-FT_HSIC: {hsic_sd:.6f}\n")
        fout.write(f"Train_vs_SFT_HSIC:       {hsic_sf:.6f}\n")

    print(f"Results saved to {log_path}")

if __name__ == "__main__":
    main()
