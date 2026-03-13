#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打印用于 sentence embedding 的模型隐藏层维度和相关配置参数
（与 semantic_entropy.py 中使用的 embed_sentences 保持一致）
"""
from transformers import AutoConfig, AutoModel

# 与 semantic_entropy.py 中一致的模型路径
MODEL_PATH = "/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/model/Llama-2-7b-chat-hf"

def main():
    # 只加载配置，不下载权重
    config = AutoConfig.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
    print("模型名称或路径:", MODEL_PATH)
    print("hidden_size (输出维度):", config.hidden_size)
    print("num_hidden_layers:", config.num_hidden_layers)
    print("num_attention_heads:", config.num_attention_heads)
    print("intermediate_size (Feed-Forward 层维度):", getattr(config, "intermediate_size", "N/A"))

    print("hidden_act (激活函数):", config.hidden_act)
    # 安全地获取 LayerNorm epsilon，不同模型可能用不同字段名
    ln_eps = getattr(config, "layer_norm_eps",
             getattr(config, "rms_norm_eps", "N/A"))
    print("layer_norm_eps / rms_norm_eps:", ln_eps)

    # 验证最后一层输出维度
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    sample = tokenizer(["Test embedding"], return_tensors="pt")
    outputs = model(**sample)
    bs, sl, hs = outputs.last_hidden_state.shape
    print(f"验证输出形状: batch_size={bs}, seq_len={sl}, hidden_size={hs}")

if __name__ == "__main__":
    main()
