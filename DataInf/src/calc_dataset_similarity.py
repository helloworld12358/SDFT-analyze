#!/usr/bin/env python3
import argparse
import torch
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="计算两个梯度的 Hessian 相似度")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target", type=str, default="q_proj,v_proj")
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--grad1_path", type=str, required=True)
    parser.add_argument("--grad2_path", type=str, required=True)
    parser.add_argument("--damping", type=float, default=0.001)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--out_path", type=str, default=None, help="写入 JSON 结果的路径")
    return parser.parse_args()

def get_lora_grad_vector(model, device):
    grad_list = []
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad:
            if param.grad is not None:
                grad_list.append(param.grad.view(-1))
            else:
                grad_list.append(torch.zeros_like(param).view(-1))
    if not grad_list:
        return None
    return torch.cat(grad_list).to(device)

def smart_parse_example(example):
    keys = example.keys()
    if "question" in keys and "answer" in keys: return example["question"], example["answer"]
    if "goal" in keys and "target" in keys: return example["goal"], example["target"]
    if "prompt" in keys and ("canonical_solution" in keys or "buggy_solution" in keys):
        instruction = example.get("instruction", "")
        full_prompt = f"{instruction}\n{example['prompt']}" if instruction else example['prompt']
        solution = example.get("canonical_solution", example.get("output", ""))
        return full_prompt, solution
    if "instruction" in keys and "output" in keys:
        prompt = example["instruction"]
        if "input" in keys and example["input"]: prompt += "\n" + example["input"]
        return prompt, example["output"]
    input_text = example.get("text", example.get("input", ""))
    output_text = example.get("label", example.get("response", ""))
    return input_text, output_text

def main():
    args = parse_args()
    device_available = torch.cuda.is_available()
    device = "cuda" if device_available else "cpu"
    torch_dtype = torch.bfloat16 if (device_available and torch.cuda.is_bf16_supported()) else (torch.float16 if device_available else torch.float32)

    v1 = torch.load(args.grad1_path, map_location=device).float()
    v2 = torch.load(args.grad2_path, map_location=device).float()

    device_map = "auto"
    if device_available:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if vis != "":
            device_map = {"": "cuda:0"}

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    base_model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
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
            target_modules=args.lora_target.split(",")
        )
        model = get_peft_model(base_model, peft_config)

    model.train()

    train_dataset = load_dataset("json", data_files=args.train_dataset_path, split="train")
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(len(train_dataset), args.max_train_samples)))

    r = torch.zeros_like(v1, device=device)
    lambda_const = args.damping
    n_train = len(train_dataset)

    for example in tqdm(train_dataset, desc="DataInf loop"):
        input_text, output_text = smart_parse_example(example)
        if not input_text or not output_text: continue

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

    similarity_score = torch.dot(r_final, v2).item()

    result = {
        "grad1": os.path.basename(args.grad1_path),
        "grad2": os.path.basename(args.grad2_path),
        "score": float(similarity_score),
        "n_train": n_train,
        "damping": lambda_const
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()