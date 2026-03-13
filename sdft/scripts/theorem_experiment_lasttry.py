#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theorem_experiment_lasttry.py

修改（按用户要求）：
- 强制要求传入 --output_path（必须是文件路径），否则直接报错。
- 在写入前进行保护性检查（绝对路径化、父目录存在性检查、若目标文件存在且非空则报错；若存在且为空则允许写入）。
- 不再使用原先的“没有output_path则写入 base_dir/experiment_results”的 fallback。
- 其他逻辑保持不变（最小入侵）。
"""
import os
import sys
import json
import argparse
import time
import gc
import random
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import parameters_to_vector

# transformers / peft
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType
except Exception as e:
    print("Error importing transformers/peft:", e, file=sys.stderr)
    raise

# ---------------------------
# Configurable warm-up hyperparams (保留常量以便不改变外部接口，但不再使用)
# ---------------------------
WARMUP_BATCH_SIZE = 8      # 原先用于 warm-up 抽样（现在不使用）
WARMUP_LR = 1e-4           # 原先用于 warm-up 学习率（现在不使用）
WARMUP_UPDATES = 1         # 原先 warm-up 更新步数（现在不使用）

# ---------------------------
# Utilities & data handling
# ---------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def validate_and_prepare_output_path(output_path: Optional[str]) -> str:
    """
    Enforce that output_path is provided and is a file path (not a directory).
    Resolve relative paths to absolute. Create parent directory if needed.
    If file exists and is non-empty -> raise RuntimeError.
    If file exists and size == 0 -> allow overwrite.
    """
    if not output_path:
        raise RuntimeError("必须提供 --output_path（完整的目标文件路径），否则退出。")
    out_path = os.path.abspath(output_path)
    if os.path.isdir(out_path):
        raise RuntimeError(f"output_path 指向一个目录（{out_path}）；现在要求传入完整文件路径而非目录。")
    parent = os.path.dirname(out_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(out_path):
        try:
            size = os.path.getsize(out_path)
        except OSError:
            raise RuntimeError(f"无法访问已存在文件: {out_path}")
        if size > 0:
            raise RuntimeError(f"目标输出文件已存在且非空，拒绝覆盖：{out_path}")
        # 如果是空文件（size == 0），允许写入（按用户要求）
    return out_path

def smart_parse_example(example: Dict) -> Tuple[str, str]:
    keys = set(example.keys()) if isinstance(example, dict) else set()
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
    if "text" in keys:
        return example.get("text", ""), ""
    if isinstance(example, str):
        return example, ""
    return (json.dumps(example, ensure_ascii=False), "")

def alpaca_format(user_content: str, assistant_response: str) -> str:
    return f"### Instruction:\n{user_content}\n\n### Response:\n{assistant_response}"

def gsm8k_format(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"

def choose_template_by_dataset_path(dataset_path: str, explicit: Optional[str] = None) -> str:
    if explicit is not None:
        explicit = explicit.lower()
        if explicit in ("alpaca", "gsm8k"):
            return explicit
    name = os.path.basename(dataset_path).lower()
    if "gsm8k" in name or "multiarith" in name or ("multi" in name and "arith" in name):
        return "gsm8k"
    return "alpaca"

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

class TextDatasetFromJson(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length:int):
        enc = tokenizer(texts, truncation=True, padding=False, max_length=max_length)
        self.input_ids = [ids for ids in enc["input_ids"]]
        self.labels = [ids.copy() for ids in enc["input_ids"]]
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}

# ---------------------------
# Helper utils for adapters & grads
# ---------------------------

def collect_adapter_grads_flattened(model: torch.nn.Module, adapter_names: List[str]) -> torch.Tensor:
    name_to_param = dict(model.named_parameters())
    parts = []
    for n in adapter_names:
        p = name_to_param.get(n, None)
        if p is None:
            parts.append(torch.zeros(0, dtype=torch.float32))
            continue
        if p.grad is None:
            parts.append(torch.zeros(p.numel(), dtype=torch.float32))
        else:
            parts.append(p.grad.detach().reshape(-1).cpu().to(torch.float32))
    if len(parts) == 0:
        return torch.tensor([], dtype=torch.float32)
    return torch.cat(parts)

def avg_of_vectors(list_of_vecs: List[torch.Tensor]) -> torch.Tensor:
    if len(list_of_vecs) == 0:
        return torch.tensor([], dtype=torch.float32)
    return torch.mean(torch.stack(list_of_vecs, dim=0), dim=0)

def quadratic_form_from_grads_list(grads_list: List[torch.Tensor], v_vec: torch.Tensor) -> float:
    """
    v^T H v = (1/n) * sum_i (v^T g_i)^2
    grads_list: list of per-sample flattened grads (CPU tensors)
    v_vec: flattened vector (CPU)
    """
    if not grads_list:
        return 0.0
    v = v_vec.to(torch.float32).cpu()
    s = 0.0
    for g in grads_list:
        c = float(torch.dot(g, v))
        s += c * c
    return float(s / len(grads_list))

def flatten_adapter_from_state_dict(m: torch.nn.Module, adapter_names: List[str]) -> torch.Tensor:
    sd = m.state_dict()
    parts = []
    for name in adapter_names:
        if name in sd:
            parts.append(sd[name].detach().reshape(-1).cpu().float())
        else:
            # fallback to named_parameters
            found = False
            for n, p in m.named_parameters():
                if n == name:
                    parts.append(p.detach().reshape(-1).cpu().float())
                    found = True
                    break
            if not found:
                parts.append(torch.zeros(0, dtype=torch.float32))
    return torch.cat(parts) if parts else torch.tensor([], dtype=torch.float32)

# ---------------------------
# Model loading helpers (small wrapper)
# ---------------------------

def load_tokenizer_and_base(base_model_path: str, device_str: Optional[str], device_map_auto: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device_map_arg = None
    if device_str is not None:
        device_map_arg = {"": device_str}
    else:
        device_map_arg = "auto" if device_map_auto else None

    use_bf16 = torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map_arg,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    base_model.config.use_cache = False
    return base_model, tokenizer

# ---------------------------
# Main runner (按您指定的行为顺序)
# ---------------------------

def run_experiment(
    base_model_path: str,
    lora_checkpoint_path: Optional[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target: str,
    dataset_path: str,
    output_path: Optional[str],
    max_samples: Optional[int],
    batch_size: int,
    max_length: int,
    template_override: Optional[str],
    device: Optional[str],
):
    # header info
    print(json.dumps({
        "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "base_model_path": base_model_path,
        "lora_checkpoint_path": lora_checkpoint_path,
        "dataset_path": dataset_path,
        "device": device,
        "batch_size": batch_size,
        "max_samples": max_samples
    }, ensure_ascii=False))

    # validate and prepare output_path early (按用户要求：必须提供完整文件路径)
    out_path = validate_and_prepare_output_path(output_path)

    device_str = device
    device_for_runs = torch.device(device_str) if device_str else None

    # -----------------------
    # Phase A: load checkpoint PEFT model (if exists) and compute grads_after_list & final_adapter_vec
    # -----------------------
    print("[Phase A] load checkpoint PEFT model (用于 C_end)")
    # Load tokenizer + base model with device_map chosen to put model on GPU if device_str provided
    base_model_ckpt, tokenizer = load_tokenizer_and_base(base_model_path, device_str, device_map_auto=True)
    if lora_checkpoint_path and os.path.exists(lora_checkpoint_path):
        peft_ckpt_model = PeftModel.from_pretrained(base_model_ckpt, lora_checkpoint_path, is_trainable=False)
    else:
        # If no checkpoint is provided, keep the base_model as the checkpoint model (no adapter)
        # but user said checkpoint will be provided; keep this fallback minimal
        peft_ckpt_model = get_peft_model(base_model_ckpt, LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=lora_r,
            lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=[t.strip() for t in lora_target.split(",") if t.strip()]
        ))
    peft_ckpt_model.eval()

    # Prepare dataset tokenization (shared)
    texts = []
    if dataset_path.endswith(".jsonl"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                p, r = smart_parse_example(obj)
                template = choose_template_by_dataset_path(dataset_path, template_override)
                texts.append(gsm8k_format(p, r) if template=="gsm8k" else alpaca_format(p, r))
    elif dataset_path.endswith(".json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            for obj in raw:
                p, r = smart_parse_example(obj)
                template = choose_template_by_dataset_path(dataset_path, template_override)
                texts.append(gsm8k_format(p, r) if template=="gsm8k" else alpaca_format(p, r))
        elif isinstance(raw, dict):
            p, r = smart_parse_example(raw)
            template = choose_template_by_dataset_path(dataset_path, template_override)
            texts.append(gsm8k_format(p, r) if template=="gsm8k" else alpaca_format(p, r))
        else:
            raise RuntimeError("Unsupported json content in dataset")
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    texts.append(s)

    if max_samples:
        texts = texts[:max_samples]

    # tokenization (do on CPU)
    tokenized = tokenizer(texts, truncation=True, padding=False, max_length=max_length)
    ds = TextDatasetFromJson(texts, tokenizer, max_length=max_length)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda batch: collate_fn(batch, pad_id, label_pad_token_id=-100))
    # note: batch_size fixed to 1 for per-sample grads (matches previous code). Keep minimal change.

    # identify adapter names for checkpoint model
    adapter_names_ckpt = []
    sd_ckpt = peft_ckpt_model.state_dict()
    name_set_ckpt = set([n for n, _ in peft_ckpt_model.named_parameters()])
    for k in sd_ckpt.keys():
        kl = k.lower()
        if ("lora" in kl) or ("peft" in kl) or ("adapter" in kl):
            if k in name_set_ckpt:
                adapter_names_ckpt.append(k)
    if len(adapter_names_ckpt) == 0:
        for n, _ in peft_ckpt_model.named_parameters():
            nl = n.lower()
            if "lora" in nl or "peft" in nl or "adapter" in nl:
                adapter_names_ckpt.append(n)
    if len(adapter_names_ckpt) == 0:
        # no adapter in checkpoint model -> still continue but raise (user insisted checkpoints are present normally)
        raise RuntimeError("Checkpoint PEFT model contains no adapter parameters; 请检查 checkpoint 路径是否正确。")

    # ensure adapter params require_grad True (for gradient extraction)
    for n, p in peft_ckpt_model.named_parameters():
        if n in adapter_names_ckpt:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # move model to device (if not already)
    if device_for_runs is not None:
        peft_ckpt_model.to(device_for_runs)

    # compute grads_after_list on checkpoint model (逐样本)
    print("[Phase A] computing grads_after_list on checkpoint model (用于 C_end 和 avg_grad_after)")
    grads_after_list = []
    peft_ckpt_model.train()
    for batch in dataloader:
        batch = {k: v.to(next(peft_ckpt_model.parameters()).device) for k, v in batch.items()}
        peft_ckpt_model.zero_grad()
        outputs = peft_ckpt_model(**batch)
        loss = outputs.loss
        loss.backward()
        gvec = collect_adapter_grads_flattened(peft_ckpt_model, adapter_names_ckpt)
        grads_after_list.append(gvec)
        peft_ckpt_model.zero_grad()
        del outputs, loss, batch
        torch.cuda.empty_cache()
        gc.collect()

    # extract final adapter flattened vector (CPU)
    final_adapter_vec = flatten_adapter_from_state_dict(peft_ckpt_model, adapter_names_ckpt)

    # free checkpoint model to reduce memory pressure
    del peft_ckpt_model
    torch.cuda.empty_cache()
    gc.collect()

    # -----------------------
    # Phase B: load fresh base_model and construct initial LoRA adapter on it (用于 C_start)
    # -----------------------
    print("[Phase B] load fresh base_model and construct init LoRA adapter (用于 C_start)")
    # load a fresh base model (place on device if requested)
    base_device_map = {"": device_str} if device_str else "auto"
    use_bf16 = torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    base_model_init = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=({"": device_str} if device_str else None),
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    base_model_init.config.use_cache = False

    # construct LoRA config with CLI args (this init simulates theta_0 adapter)
    peft_config_init = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[t.strip() for t in lora_target.split(",") if t.strip()],
        init_lora_weights=True
    )
    init_peft_model = get_peft_model(base_model_init, peft_config_init)
    init_peft_model.eval()

    # ensure init adapter params require_grad True
    adapter_names_init = []
    sd_init = init_peft_model.state_dict()
    name_set_init = set([n for n, _ in init_peft_model.named_parameters()])
    for k in sd_init.keys():
        kl = k.lower()
        if ("lora" in kl) or ("peft" in kl) or ("adapter" in kl):
            if k in name_set_init:
                adapter_names_init.append(k)
    if len(adapter_names_init) == 0:
        for n, _ in init_peft_model.named_parameters():
            nl = n.lower()
            if "lora" in nl or "peft" in nl or "adapter" in nl:
                adapter_names_init.append(n)
    if len(adapter_names_init) == 0:
        raise RuntimeError("Failed to detect adapter parameters in init PEFT model (unexpected).")

    for n, p in init_peft_model.named_parameters():
        p.requires_grad = (n in adapter_names_init)

    # move init model to device if requested
    if device_for_runs is not None:
        init_peft_model.to(device_for_runs)

    # -----------------------
    # NOTE: warm-up has been removed. Directly compute grads_before_list on init model (theta_0)
    # -----------------------

    print("[Phase B] computing grads_before_list on init model (theta_0)")
    grads_before_list = []
    init_peft_model.train()
    for batch in dataloader:
        batch = {k: v.to(next(init_peft_model.parameters()).device) for k, v in batch.items()}
        init_peft_model.zero_grad()
        outputs = init_peft_model(**batch)
        loss = outputs.loss
        loss.backward()
        gvec = collect_adapter_grads_flattened(init_peft_model, adapter_names_init)
        grads_before_list.append(gvec)
        init_peft_model.zero_grad()
        del outputs, loss, batch
        torch.cuda.empty_cache()
        gc.collect()

    # extract init adapter flattened vector (theta_0)
    init_adapter_vec = flatten_adapter_from_state_dict(init_peft_model, adapter_names_init)

    # free init model (we have grads_before_list and init vec)
    del init_peft_model, base_model_init
    torch.cuda.empty_cache()
    gc.collect()

    # -----------------------
    # Phase C: compute delta and metrics
    # -----------------------
    # Align adapter element orders: we assume adapter_names_init and adapter_names_ckpt follow same naming/order
    # To be robust, try to align by name ordering: construct unified ordered list by ckpt adapter names if possible.
    # For minimal change, we will assume shapes/names match; otherwise user checkpoint structure must be consistent.
    delta = final_adapter_vec - init_adapter_vec
    delta_norm = float(delta.norm().item()) if delta.numel() > 0 else 0.0

    # compute quadratic forms
    C_start = 0.5 * quadratic_form_from_grads_list(grads_before_list, delta)
    C_end = 0.5 * quadratic_form_from_grads_list(grads_after_list, delta)

    # average grad after and V_align
    avg_grad_after = avg_of_vectors(grads_after_list) if len(grads_after_list) > 0 else torch.tensor([], dtype=torch.float32)
    dot_avggrad_delta = float(torch.dot(avg_grad_after, delta)) if avg_grad_after.numel() > 0 else 0.0
    V_align = - dot_avggrad_delta

    # average grad before (theta_0) and corresponding dot / V_align_before
    avg_grad_before = avg_of_vectors(grads_before_list) if len(grads_before_list) > 0 else torch.tensor([], dtype=torch.float32)
    dot_avggrad_delta_before = float(torch.dot(avg_grad_before, delta)) if avg_grad_before.numel() > 0 else 0.0
    V_align_before = - dot_avggrad_delta_before

    results = {
        "base_model_path": base_model_path,
        "lora_checkpoint_path": lora_checkpoint_path,
        "dataset_path": dataset_path,
        "delta_norm": delta_norm,
        "dot_avggrad_delta": dot_avggrad_delta,
        "V_align": V_align,
        "dot_avggrad_delta_before": dot_avggrad_delta_before,
        "V_align_before": V_align_before,
        "C_start": C_start,
        "C_end": C_end,
        "num_samples_used": len(grads_after_list)
    }

    # Save results: output_path 已在开头验证并返回 out_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved results to {out_path}")

    # cleanup
    del grads_after_list, grads_before_list, final_adapter_vec, init_adapter_vec, delta
    torch.cuda.empty_cache()
    gc.collect()

    return results

# ---------------------------
# CLI
# ---------------------------

def build_parser():
    p = argparse.ArgumentParser(description="theorem_experiment (C_start without warm-up)")
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--lora_checkpoint_path", type=str, required=True,
                   help="checkpoint directory that includes adapter files (we expect this to exist)")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target", type=str, default="q_proj,v_proj")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_path", type=str, default=None,
                   help="**必须**传入完整目标文件路径（例如 /abs/path/domain_ckpt__testkey.json），不能传目录。")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--template_override", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    return p

def main():
    args = build_parser().parse_args()
    res = run_experiment(
        base_model_path=args.base_model_path,
        lora_checkpoint_path=args.lora_checkpoint_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target=args.lora_target,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        template_override=args.template_override,
        device=args.device
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()