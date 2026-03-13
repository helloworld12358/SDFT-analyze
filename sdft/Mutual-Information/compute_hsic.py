#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from mi_estimators import hsic_gaussian

# 请确认此目录下已有从 Hugging Face 下载好的模型权重和 tokenizer 文件
MODEL_PATH = "/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/model/Llama-2-7b-chat-hf"


def load_json_data(file_path):
    """从 JSONL 文件中读取 label 与 predict 原始句子"""
    labels, predicts = [], []
    with open(file_path, 'r', encoding='utf-8') as fin:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'label' in obj and 'predict' in obj:
                labels.append(obj['label'])
                predicts.append(obj['predict'])
    return labels, predicts


def embed_sentences(sentences, device='cpu', batch_size=16):
    """
    使用 Hugging Face Transformers 对句子列表做 Mean Pooling，
    支持多 GPU 并行：
      1) AutoTokenizer + AutoModel 加载本地模型
      2) 包装 model = DataParallel(model)（若有多卡）
      3) forward ⇒ last_hidden_state
      4) 按 attention_mask 加权求和并平均
    返回 torch.Tensor([N, D])
    """
    # 1) 加载 tokenizer 和模型（仅从本地读取）
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, cache_dir=MODEL_PATH,
        local_files_only=True, use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModel.from_pretrained(
        MODEL_PATH, cache_dir=MODEL_PATH,
        local_files_only=True
    )
    # 2) 多卡并行包装
    if device.startswith('cuda') and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    # 将模型置于指定设备
    model.to(device)
    model.eval()

    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i: i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)

            # 3) 模型前向
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            last_hidden = out.last_hidden_state  # [B, L, D]

            # 4) mean pooling（仅对真实 token 求平均）
            mask = attention_mask.unsqueeze(-1).float()       # [B, L, 1]
            summed = (last_hidden * mask).sum(dim=1)          # [B, D]
            counts = mask.sum(dim=1).clamp(min=1e-9)          # [B, 1]
            embeds = (summed / counts)                       # [B, D]
            all_embeds.append(embeds.cpu())

    return torch.cat(all_embeds, dim=0)


def calculate_hsic_from_json(
    json_file_path,
    device='cpu',
    batch_size=16,
    ktype='gaussian',
    verbose=False
):
    """
    对单个 JSONL 文件：
      1) 读取 label/predict 原始句子
      2) embed_sentences → 得到 X, Y 两个 (N, D) 矩阵
      3) 调用 hsic_gaussian 计算 HSIC 标量并返回
    """
    labels, predicts = load_json_data(json_file_path)
    if not labels:
        raise RuntimeError(f"[ERROR] 在 {json_file_path} 未加载到任何数据。")

    if verbose:
        print(f"[compute_hsic] 句子数量: {len(labels)}，开始 embedding → HSIC")

    X = embed_sentences(labels,   device=device, batch_size=batch_size).numpy()
    Y = embed_sentences(predicts, device=device, batch_size=batch_size).numpy()

    if verbose:
        print(f"[compute_hsic] X.shape={X.shape}, Y.shape={Y.shape}")

    hsic_val = hsic_gaussian(X, Y, ktype=ktype)
    return float(hsic_val)



