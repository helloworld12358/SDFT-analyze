#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保存 dataset 的平均 LoRA 梯度（内置模板：Alpaca & GSM8K），
默认输出位置改为：<script_dir>/result/output_grad/<model_tag>/<dataset_basename>.pt
如果显式提供 --output_path，则使用该路径。
"""
import argparse
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional


def parse_args():
    p = argparse.ArgumentParser(description="保存 dataset 的平均 LoRA 梯度（内置模板）")
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target", type=str, default="q_proj,v_proj")
    p.add_argument("--dataset_path", type=str, required=True, help="json 文件路径或包含关键词的数据集名")
    p.add_argument("--output_path", type=str, default=None, help="可选：显式输出路径（.pt），若不提供则保存到本地 result 目录")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--template_override", type=str, default=None, help="可选: 'alpaca' or 'gsm8k'")
    return p.parse_args()


# ---------------------------
# Templates (内置，不依赖外部库)
# ---------------------------
def alpaca_format(user_content: str, assistant_response: str) -> str:
    return f"### Instruction:\n{user_content}\n\n### Response:\n{assistant_response}"


def gsm8k_format(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


# ---------------------------
# 辅助：从 dataset 示例中抽取 prompt/response
# ---------------------------
def smart_parse_example(example: Dict) -> Tuple[str, str]:
    keys = set(example.keys())
    if "instruction" in keys and "output" in keys:
        instr = example.get("instruction", "")
        extra_input = example.get("input", "")
        if extra_input:
            instr = instr + "\n" + extra_input
        return instr, example.get("output", "")
    if "instruction" in keys and "response" in keys:
        instr = example.get("instruction", "")
        extra_input = example.get("input", "")
        if extra_input:
            instr = instr + "\n" + extra_input
        return instr, example.get("response", "")
    if "question" in keys and "answer" in keys:
        return example.get("question", ""), example.get("answer", "")
    if "goal" in keys and "target" in keys:
        return example.get("goal", ""), example.get("target", "")
    if "prompt" in keys and ("canonical_solution" in keys or "buggy_solution" in keys):
        instruction = example.get("instruction", "")
        full_prompt = f"{instruction}\n{example['prompt']}" if instruction else example["prompt"]
        solution = example.get("canonical_solution", example.get("output", ""))
        return full_prompt, solution
    if "input" in keys and "output" in keys:
        return example.get("input", ""), example.get("output", "")
    return example.get("text", example.get("input", "")), example.get("label", example.get("response", ""))


# ---------------------------
# 选择模板规则
# ---------------------------
def choose_template_by_dataset_path(dataset_path: str, explicit: Optional[str] = None) -> str:
    if explicit is not None:
        explicit = explicit.lower()
        if explicit in ("alpaca", "gsm8k"):
            return explicit
    name = os.path.basename(dataset_path).lower()
    if "gsm8k" in name or "multiarith" in name or ("multi" in name and "arith" in name):
        return "gsm8k"
    return "alpaca"


# ---------------------------
# LoRA 梯度提取
# ---------------------------
def get_lora_grad_vector(model: torch.nn.Module) -> Optional[torch.Tensor]:
    grad_list = []
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad:
            if param.grad is not None:
                grad_list.append(param.grad.detach().view(-1).cpu())
            else:
                grad_list.append(torch.zeros_like(param).view(-1).cpu())
    if not grad_list:
        return None
    return torch.cat(grad_list)


# ---------------------------
# 自定义 collate（对 causal LM 我们需要将 labels 的 padding token 设置为 -100）
# ---------------------------
def collate_fn(batch: List[Dict], pad_token_id: int, label_pad_token_id: int = -100) -> Dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for x in batch:
        ids = list(x["input_ids"])
        att = [1] * len(ids)
        lab = list(x["labels"])
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = ids + [pad_token_id] * pad_len
            att = att + [0] * pad_len
            lab = lab + [label_pad_token_id] * pad_len
        input_ids.append(ids)
        attention_mask.append(att)
        labels.append(lab)
    batch_tensor = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    return batch_tensor


# ---------------------------
# 主流程
# ---------------------------
def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 设备选择（绑定到单个可见 GPU：若 CUDA_VISIBLE_DEVICES 已被设置为单个设备，则将模型映射到 cuda:0）
    device_available = torch.cuda.is_available()
    device = "cuda" if device_available else "cpu"
    torch_dtype = (
        torch.bfloat16 if (device_available and torch.cuda.is_bf16_supported()) else
        (torch.float16 if device_available else torch.float32)
    )

    # device_map 策略（避免多进程下 transformers 的 device_map="auto" 导致跨进程冲突）
    device_map = None
    if device_available:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if vis != "":
            device_map = {"": "cuda:0"}
        else:
            device_map = "auto"

    print(f"Loading Model: {args.base_model_path} (dtype={torch_dtype}, device={device})")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    base_model.config.use_cache = False

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    template_name = choose_template_by_dataset_path(args.dataset_path, args.template_override)
    print(f"[Info] Using integrated template: {template_name}")

    # load or build PEFT model (LoRA)
    if args.lora_path and os.path.exists(args.lora_path):
        model = PeftModel.from_pretrained(base_model, args.lora_path, is_trainable=True)
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target.split(",")
        )
        model = get_peft_model(base_model, peft_config)

    model.train()

    # --- 2. 数据处理与 DataLoader ---
    dataset = load_dataset("json", data_files={"train": args.dataset_path}, split="train")
    if args.max_samples:
        dataset = dataset.select(range(min(len(dataset), args.max_samples)))

    def preprocess_function(examples):
        inputs_texts = []
        for i in range(len(examples[list(examples.keys())[0]])):
            ex = {k: examples[k][i] for k in examples}
            prompt, response = smart_parse_example(ex)
            if not prompt and not response:
                inputs_texts.append("")
                continue
            if template_name == "gsm8k":
                text = gsm8k_format(prompt, response)
            else:
                text = alpaca_format(prompt, response)
            inputs_texts.append(text)

        model_inputs = tokenizer(inputs_texts, max_length=args.max_length, truncation=True, padding=False)
        model_inputs["labels"] = [ids.copy() for ids in model_inputs["input_ids"]]
        return model_inputs

    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing with integrated templates"
    )
    processed_dataset = processed_dataset.filter(lambda x: len(x["input_ids"]) > 0)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    dataloader = DataLoader(
        processed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=pad_id, label_pad_token_id=-100)
    )

    # --- 3. 批量计算梯度 ---
    print(f"Computing Gradients (Batch Size: {args.batch_size})...")
    sum_grad_vector = None
    total_samples = 0

    for batch in tqdm(dataloader):
        batch = {k: v.to("cuda" if device == "cuda" else "cpu") for k, v in batch.items()}
        current_batch_size = batch["input_ids"].shape[0]

        model.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        grad_vec = get_lora_grad_vector(model)

        if grad_vec is not None:
            weighted_grad = grad_vec * current_batch_size
            if sum_grad_vector is None:
                sum_grad_vector = torch.zeros_like(grad_vec)
            sum_grad_vector += weighted_grad
            total_samples += current_batch_size

        del batch, outputs, loss, grad_vec

    if sum_grad_vector is None or total_samples == 0:
        raise RuntimeError(f"[Error] No gradients computed. total_samples={total_samples}")

    avg_grad_vector = sum_grad_vector / total_samples

    # 决定输出路径：若用户指定 --output_path 则使用，否则保存到本地 result 目录
    if args.output_path:
        out_path = args.output_path
    else:
        model_tag = os.path.basename(args.base_model_path).replace("/", "_")
        ds_tag = os.path.splitext(os.path.basename(args.dataset_path))[0]
        local_root = os.path.join(script_dir, "result", "output_grad", model_tag)
        os.makedirs(local_root, exist_ok=True)
        out_path = os.path.join(local_root, f"{ds_tag}.pt")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(avg_grad_vector, out_path)
    print(f"Saved to: {out_path} (Total Samples: {total_samples})")


if __name__ == "__main__":
    main()