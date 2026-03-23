#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute DataInf similarity for one grad1 against multiple grad2 vectors
in a single pass over the train dataset.

This is a throughput-oriented variant of calc_dataset_similarity.py:
- load model once
- traverse train data once
- produce multiple scores: score_i = <r_final(v1), v2_i>
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute one-to-many DataInf similarity scores in one pass.")
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target", type=str, default="q_proj,v_proj")
    p.add_argument("--train_dataset_path", type=str, required=True)
    p.add_argument("--grad1_path", type=str, required=True)
    p.add_argument("--grad2_paths", type=str, required=True, help="comma-separated grad2 paths")
    p.add_argument("--grad2_names", type=str, default="", help="comma-separated names for grad2; optional")
    p.add_argument("--damping", type=float, default=0.001)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--out_path", type=str, default=None)
    return p.parse_args()


def smart_parse_example(example: Dict) -> Tuple[str, str]:
    keys = example.keys()
    if "question" in keys and "answer" in keys:
        return example["question"], example["answer"]
    if "goal" in keys and "target" in keys:
        return example["goal"], example["target"]
    if "prompt" in keys and ("canonical_solution" in keys or "buggy_solution" in keys):
        instruction = example.get("instruction", "")
        full_prompt = f"{instruction}\n{example['prompt']}" if instruction else example["prompt"]
        solution = example.get("canonical_solution", example.get("output", ""))
        return full_prompt, solution
    if "instruction" in keys and "output" in keys:
        prompt = example["instruction"]
        if "input" in keys and example["input"]:
            prompt += "\n" + example["input"]
        return prompt, example["output"]
    input_text = example.get("text", example.get("input", ""))
    output_text = example.get("label", example.get("response", ""))
    return input_text, output_text


def get_lora_grad_vector(model: torch.nn.Module, device: str) -> Optional[torch.Tensor]:
    grad_list: List[torch.Tensor] = []
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad:
            if param.grad is not None:
                grad_list.append(param.grad.view(-1))
            else:
                grad_list.append(torch.zeros_like(param).view(-1))
    if not grad_list:
        return None
    return torch.cat(grad_list).to(device)


def split_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def load_model_and_tokenizer(args: argparse.Namespace, device_available: bool, device: str):
    torch_dtype = (
        torch.bfloat16
        if (device_available and torch.cuda.is_bf16_supported())
        else (torch.float16 if device_available else torch.float32)
    )
    device_map: object = "auto"
    if device_available:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if vis:
            device_map = {"": "cuda:0"}

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.lora_path and os.path.exists(args.lora_path):
        model = PeftModel.from_pretrained(base_model, args.lora_path, is_trainable=True)
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target.split(","),
        )
        model = get_peft_model(base_model, peft_config)

    model.train()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    device_available = torch.cuda.is_available()
    device = "cuda" if device_available else "cpu"

    grad2_paths = split_csv(args.grad2_paths)
    if not grad2_paths:
        raise ValueError("grad2_paths is empty")
    grad2_names = split_csv(args.grad2_names) if args.grad2_names else []
    if grad2_names and len(grad2_names) != len(grad2_paths):
        raise ValueError(f"grad2_names count ({len(grad2_names)}) != grad2_paths count ({len(grad2_paths)})")
    if not grad2_names:
        grad2_names = [os.path.basename(p) for p in grad2_paths]

    v1 = torch.load(args.grad1_path, map_location=device).float()
    v2_list: List[torch.Tensor] = []
    for p in grad2_paths:
        v2_list.append(torch.load(p, map_location=device).float())

    model, tokenizer = load_model_and_tokenizer(args, device_available=device_available, device=device)

    train_dataset = load_dataset("json", data_files=args.train_dataset_path, split="train")
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(len(train_dataset), args.max_train_samples)))

    r = torch.zeros_like(v1, device=device)
    lambda_const = float(args.damping)
    n_train = len(train_dataset)

    for example in tqdm(train_dataset, desc="DataInf loop (multi-grad2)"):
        input_text, output_text = smart_parse_example(example)
        if not input_text or not output_text:
            continue

        full_text = f"{input_text}\n{output_text}"
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs["input_ids"].to(device)

        model.zero_grad()
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        g_train = get_lora_grad_vector(model, device)
        if g_train is not None:
            dot_product = torch.dot(v1, g_train)
            norm_sq = torch.norm(g_train) ** 2
            c = dot_product / (lambda_const + norm_sq)
            r += (v1 - c * g_train)

        del input_ids, outputs, loss, g_train

    if n_train == 0:
        r_final = r
    else:
        if lambda_const > 0:
            r_final = r / (n_train * lambda_const)
        else:
            r_final = r / n_train

    score_map: Dict[str, float] = {}
    items: List[Dict[str, object]] = []
    for name, path, v2 in zip(grad2_names, grad2_paths, v2_list):
        score = torch.dot(r_final, v2).item()
        score_map[name] = float(score)
        items.append({"name": name, "grad2": os.path.basename(path), "score": float(score)})

    result: Dict[str, object] = {
        "grad1": os.path.basename(args.grad1_path),
        "n_train": n_train,
        "damping": lambda_const,
        "scores": items,
        "score_map": score_map,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

