#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theorem_experiment_random.py
(基于你提供的版本，增加了 --reuse_base_model 选项以减少重复 from_pretrained 开销)

新增要点（最小修改）：
- CLI 增加 --reuse_base_model（bool，default False）
- 若 reuse_base_model=True：先加载 base model template 到 CPU（一次），
  每次 run 时用 copy.deepcopy(template) 在 CPU 上复制出一份，再移动到 GPU；run 结束后删除该拷贝。
- 其他逻辑（流式写 grads 到 tmp_dir、按文件逐个计算平均和 quadratic）保持不变。
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import sys
import json
import argparse
import time
import gc
import random
import shutil
import tempfile
import uuid
import copy
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# transformers / peft
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType
except Exception as e:
    print("Error importing transformers/peft:", e, file=sys.stderr)
    raise

# ---------------------------
# Configurable warm-up hyperparams
# ---------------------------
WARMUP_BATCH_SIZE = 8
WARMUP_LR = 1e-4
WARMUP_UPDATES = 1

# ---------------------------
# Utilities & data handling (与你原版保持一致)
# ---------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

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
# Helper utils for adapters & grads (与你原版保持一致)
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

def quadratic_sum_proj_from_files(file_paths: List[str], v_vec: torch.Tensor) -> float:
    if not file_paths:
        return 0.0
    v = v_vec.to(torch.float32).cpu().numpy()
    s = 0.0
    n = 0
    for p in file_paths:
        try:
            arr = np.load(p, mmap_mode=None)
            if arr.size == 0:
                continue
            g = arr.ravel()
            c = float(np.dot(g, v))
            s += c * c
            n += 1
        except Exception:
            continue
    return float(s / n) if n > 0 else 0.0

def avg_vector_from_files(file_paths: List[str]) -> torch.Tensor:
    if not file_paths:
        return torch.tensor([], dtype=torch.float32)
    total = None
    n = 0
    for p in file_paths:
        try:
            arr = np.load(p, mmap_mode=None)
            if arr.size == 0:
                continue
            if total is None:
                total = np.array(arr, dtype=np.float64)
            else:
                total += arr.astype(np.float64)
            n += 1
        except Exception:
            continue
    if total is None or n == 0:
        return torch.tensor([], dtype=torch.float32)
    mean_np = (total / n).astype(np.float32)
    return torch.from_numpy(mean_np).to(torch.float32).cpu()

def flatten_adapter_from_state_dict(m: torch.nn.Module, adapter_names: List[str]) -> torch.Tensor:
    sd = m.state_dict()
    parts = []
    for name in adapter_names:
        if name in sd:
            parts.append(sd[name].detach().reshape(-1).cpu().float())
        else:
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
# Model loading helpers (single-process / DDP-compatible)
# ---------------------------
def load_tokenizer_and_base(base_model_path: str, device_str: Optional[str], device_map_auto: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_bf16 = torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # DDP/torchrun 环境下建议 device_map=None（我们会显式 .to(device_str)）
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    base_model.config.use_cache = False
    if device_str is not None:
        base_model = base_model.to(device_str)
    return base_model, tokenizer

# ---------------------------
# Atomic file helper
# ---------------------------
def atomic_write_json(data: dict, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    dir_name = os.path.dirname(os.path.abspath(out_path))
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp.json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

# ---------------------------
# Distributed helpers (DDP path still unchanged)
# ---------------------------
def init_dist() -> bool:
    if "RANK" not in os.environ:
        return False
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return True

def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1

def is_main() -> bool:
    return get_rank() == 0

def gather_and_save_grads_to_disk(local_grads: List[torch.Tensor], tmp_dir: Optional[str] = None, tag: str = "grads"):
    if not dist.is_initialized() or get_world_size() == 1:
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp(prefix=f"{tag}_")
        ensure_dir(tmp_dir)
        file_paths = []
        for i, g in enumerate(local_grads):
            path = os.path.join(tmp_dir, f"{tag}_local_{i}_{uuid.uuid4().hex}.npy")
            np.save(path, g.numpy() if isinstance(g, torch.Tensor) else np.asarray(g))
            file_paths.append(path)
        return file_paths, len(file_paths)

    world_size = get_world_size()
    rank = get_rank()

    send_obj = []
    for g in local_grads:
        try:
            send_obj.append(g.numpy())
        except Exception:
            send_obj.append(np.array([], dtype=np.float32))

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, send_obj)

    if rank != 0:
        return [], 0

    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix=f"{tag}_")
    ensure_dir(tmp_dir)
    file_paths = []
    for r in range(world_size):
        lst = gathered[r]
        if not lst:
            continue
        for i, arr in enumerate(lst):
            try:
                fname = os.path.join(tmp_dir, f"{tag}_r{r}_{i}_{uuid.uuid4().hex}.npy")
                np.save(fname, np.asarray(arr))
                file_paths.append(fname)
            except Exception:
                continue
    total_count = len(file_paths)
    return file_paths, total_count

# ---------------------------
# Helper: resolve deterministic out_path
# ---------------------------
def resolve_out_path(output_path: Optional[str], lora_checkpoint_path: Optional[str], base_model_path: str, dataset_path: str) -> str:
    ckpt_name = os.path.basename(os.path.normpath(lora_checkpoint_path)) if lora_checkpoint_path else "no_ckpt"
    data_name = os.path.splitext(os.path.basename(dataset_path))[0]
    if output_path:
        if os.path.isdir(output_path):
            return os.path.join(output_path, f"{ckpt_name}__{data_name}.json")
        else:
            return output_path
    else:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(base_model_path)), "experiment_results")
        ensure_dir(base_dir)
        return os.path.join(base_dir, f"{ckpt_name}__{data_name}.json")

# ---------------------------
# Execute single run (现在支持 base_model_template reuse)
# ---------------------------
def execute_single_run(
    run_idx: int,
    n_runs: int,
    base_model_path: str,
    device_str: Optional[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target: str,
    texts: List[str],
    tokenizer: Any,
    max_length: int,
    pad_id: int,
    dataloader: DataLoader,
    tmp_dir: str,
    base_model_template: Optional[torch.nn.Module] = None,
) -> Tuple[List[str], int, torch.Tensor]:
    """
    执行单次 Phase B：
    - 若 base_model_template 提供，则使用 copy.deepcopy(template)（在 CPU 上复制）并 .to(device_str)；
    - 否则走 from_pretrained（原有逻辑）。
    返回: before_files, before_count, init_adapter_vec（CPU tensor）
    """
    if is_main():
        print(f"[Phase B][run {run_idx+1}/{n_runs}] load fresh base_model and construct init LoRA adapter")

    use_bf16 = torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # ---------- load or deepcopy ----------
    if base_model_template is not None:
        # 基本策略：deepcopy template（template 在调用方可保留在 CPU），
        # deepcopy 会复制 tensor（在 CPU），之后把复制的模型移动到 device。
        model_copy = copy.deepcopy(base_model_template)
        if device_str is not None:
            model_copy = model_copy.to(device_str)
        base_model_init = model_copy
        base_model_init.config.use_cache = False
    else:
        base_model_init = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        if device_str is not None:
            base_model_init = base_model_init.to(device_str)
        base_model_init.config.use_cache = False

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

    # detect adapter names
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

    if device_str is not None:
        torch.cuda.synchronize()

    # warm-up (synchronized indices in DDP path; here we assume single-process or broadcast outside)
    if len(texts) > 0:
        warm_n = min(WARMUP_BATCH_SIZE, len(texts))
        if is_main():
            _warm_indices = random.sample(range(len(texts)), warm_n)
        else:
            _warm_indices = [0] * warm_n
        # in DDP we'd broadcast _warm_indices; here we rely on torchrun/ ddp env if needed
        warm_samples = [texts[i] for i in _warm_indices]
        warm_ds = TextDatasetFromJson(warm_samples, tokenizer, max_length=max_length)
        warm_batch = collate_fn([warm_ds[i] for i in range(len(warm_ds))], pad_id, label_pad_token_id=-100)

        adapter_params = [p for n, p in init_peft_model.named_parameters() if n in adapter_names_init and p.requires_grad]
        if len(adapter_params) > 0:
            opt = torch.optim.AdamW(adapter_params, lr=WARMUP_LR)
            init_peft_model.train()
            for _ in range(WARMUP_UPDATES):
                wb = {k: v.to(next(init_peft_model.parameters()).device) for k, v in warm_batch.items()}
                opt.zero_grad()
                outputs = init_peft_model(**wb)
                loss = outputs.loss
                loss.backward()
                opt.step()
                del outputs, loss
                for k in list(wb.keys()):
                    del wb[k]
                torch.cuda.empty_cache()
                gc.collect()
            del warm_batch, warm_ds, warm_samples
            del opt
            torch.cuda.empty_cache()
            gc.collect()
        else:
            del warm_batch, warm_ds, warm_samples
            torch.cuda.empty_cache()
            gc.collect()

    # compute grads_before (per-sample)
    local_before_grads = []
    init_peft_model.train()
    for batch in dataloader:
        batch_dev = {k: v.to(next(init_peft_model.parameters()).device) for k, v in batch.items()}
        init_peft_model.zero_grad()
        outputs = init_peft_model(**batch_dev)
        loss = outputs.loss
        loss.backward()
        gvec = collect_adapter_grads_flattened(init_peft_model, adapter_names_init)
        local_before_grads.append(gvec)
        init_peft_model.zero_grad()
        del outputs, loss
        for k in list(batch_dev.keys()):
            del batch_dev[k]
        torch.cuda.empty_cache()
        gc.collect()

    if dist.is_initialized():
        dist.barrier()

    _before_tmp = os.path.join(tmp_dir, f"before_run{run_idx}")
    before_files, before_count = gather_and_save_grads_to_disk(local_before_grads, tmp_dir=_before_tmp, tag=f"before_run{run_idx}")
    del local_before_grads
    gc.collect()

    init_adapter_vec = flatten_adapter_from_state_dict(init_peft_model, adapter_names_init).cpu()

    # clean up model copy to free memory
    del init_peft_model
    if base_model_template is None:
        # base_model_init came from from_pretrained inside this function -> delete
        del base_model_init
    else:
        # base_model_init is a deep copy of template -> delete copy
        del base_model_init
    torch.cuda.empty_cache()
    gc.collect()

    return before_files, before_count, init_adapter_vec

# ---------------------------
# Main runner (大段逻辑保留，只加入 reuse_base_model 支持)
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
    n_runs: int = 10,
    tmp_dir: Optional[str] = None,
    reuse_base_model: bool = False,   # NEW FLAG
):
    # init dist if torchrun
    init_dist()
    device_str = device
    if dist.is_initialized():
        device_str = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"

    device_for_runs = torch.device(device_str) if device_str else None

    # derive deterministic out/tmp path
    _out_path = resolve_out_path(output_path, lora_checkpoint_path, base_model_path, dataset_path)
    if tmp_dir is None:
        tmp_dir = os.path.splitext(os.path.abspath(_out_path))[0] + "_tmp"
    _after_tmp = os.path.join(tmp_dir, "after")
    ensure_dir(tmp_dir)
    ensure_dir(_after_tmp)

    if is_main():
        print(json.dumps({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "base_model_path": base_model_path,
            "lora_checkpoint_path": lora_checkpoint_path,
            "dataset_path": dataset_path,
            "device": device_str,
            "world_size": get_world_size(),
            "batch_size": batch_size,
            "max_samples": max_samples,
            "n_runs": n_runs,
            "tmp_dir": tmp_dir,
            "reuse_base_model": reuse_base_model,
        }, ensure_ascii=False))

    # -----------------------
    # Phase A: load checkpoint PEFT model (compute grads_after)
    # -----------------------
    if is_main():
        print("[Phase A] load checkpoint PEFT model (用于 C_end) -- single execution")
    base_model_ckpt, tokenizer = load_tokenizer_and_base(base_model_path, device_str, device_map_auto=True)
    if lora_checkpoint_path and os.path.exists(lora_checkpoint_path):
        peft_ckpt_model = PeftModel.from_pretrained(base_model_ckpt, lora_checkpoint_path, is_trainable=False)
    else:
        peft_ckpt_model = get_peft_model(base_model_ckpt, LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=lora_r,
            lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=[t.strip() for t in lora_target.split(",") if t.strip()]
        ))
    peft_ckpt_model.eval()

    # prepare dataset -> texts / dataloader (与之前相同)
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

    tokenized = tokenizer(texts, truncation=True, padding=False, max_length=max_length)
    ds = TextDatasetFromJson(texts, tokenizer, max_length=max_length)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    if dist.is_initialized() and get_world_size() > 1:
        _sampler = DistributedSampler(ds, shuffle=False, drop_last=False)
        dataloader = DataLoader(
            ds, batch_size=1, sampler=_sampler,
            collate_fn=lambda batch: collate_fn(batch, pad_id, label_pad_token_id=-100)
        )
    else:
        dataloader = DataLoader(
            ds, batch_size=1, shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, pad_id, label_pad_token_id=-100)
        )

    # detect adapter params in checkpoint model
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
        raise RuntimeError("Checkpoint PEFT model contains no adapter parameters; 请检查 checkpoint 路径是否正确。")

    for n, p in peft_ckpt_model.named_parameters():
        if n in adapter_names_ckpt:
            p.requires_grad = True
        else:
            p.requires_grad = False

    if device_for_runs is not None:
        peft_ckpt_model.to(device_for_runs)

    # compute grads_after and stream to disk
    if is_main():
        print("[Phase A] computing grads_after (per-sample) and streaming to disk")
    local_after_grads = []
    peft_ckpt_model.train()
    for batch in dataloader:
        batch = {k: v.to(next(peft_ckpt_model.parameters()).device) for k, v in batch.items()}
        peft_ckpt_model.zero_grad()
        outputs = peft_ckpt_model(**batch)
        loss = outputs.loss
        loss.backward()
        gvec = collect_adapter_grads_flattened(peft_ckpt_model, adapter_names_ckpt)
        local_after_grads.append(gvec)
        peft_ckpt_model.zero_grad()
        del outputs, loss
        for k in list(batch.keys()):
            del batch[k]
        torch.cuda.empty_cache()
        gc.collect()

    if dist.is_initialized():
        dist.barrier()
    after_files, after_count = gather_and_save_grads_to_disk(local_after_grads, tmp_dir=_after_tmp, tag="after")
    del local_after_grads
    gc.collect()

    if is_main():
        print(f"[Phase A] collected grads_after: {after_count} samples -> stored {len(after_files)} files")

    final_adapter_vec = flatten_adapter_from_state_dict(peft_ckpt_model, adapter_names_ckpt).cpu()

    # cleanup checkpoint model
    del peft_ckpt_model
    torch.cuda.empty_cache()
    gc.collect()
    if dist.is_initialized():
        dist.barrier()

    # 如果启用 reuse_base_model：提前加载 base model template 到 CPU（只做一次）
    base_model_template = None
    if reuse_base_model:
        if is_main():
            print("[INFO] Preloading base model template for reuse (keeps template on CPU).")
        # load with device_map=None (will be in CPU), ensure low_cpu_mem_usage may still pin some tensors to CPU
        use_bf16 = torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
        # IMPORTANT: load to CPU (device_map=None) to keep GPU free; we'll deepcopy and move to GPU per-run.
        base_model_template = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        base_model_template.config.use_cache = False
        # keep on CPU by .cpu()
        base_model_template = base_model_template.cpu()
        # note: this keeps a CPU copy; deep copying this template per-run will duplicate CPU memory.

    # accumulators for means
    delta_norm_sum = 0.0
    dot_after_sum = 0.0
    V_align_sum = 0.0
    dot_before_sum = 0.0
    V_align_before_sum = 0.0
    C_start_sum = 0.0
    C_end_sum = 0.0
    num_samples_sum = 0

    avg_grad_after = avg_vector_from_files(after_files) if is_main() and after_files else torch.tensor([], dtype=torch.float32)

    # Phase B runs — 使用 execute_single_run；传入 base_model_template（可能为 None）
    for run_idx in range(n_runs):
        if is_main():
            print(f"[Phase B][run {run_idx+1}/{n_runs}] starting run")
        torch.cuda.empty_cache()
        gc.collect()

        before_files, before_count, init_adapter_vec = execute_single_run(
            run_idx=run_idx,
            n_runs=n_runs,
            base_model_path=base_model_path,
            device_str=device_str,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target=lora_target,
            texts=texts,
            tokenizer=tokenizer,
            max_length=max_length,
            pad_id=pad_id,
            dataloader=dataloader,
            tmp_dir=tmp_dir,
            base_model_template=base_model_template
        )

        if is_main():
            delta = final_adapter_vec - init_adapter_vec
            delta = delta.cpu()

            delta_norm = float(delta.norm().item()) if delta.numel() > 0 else 0.0
            C_start = 0.5 * quadratic_sum_proj_from_files(before_files, delta)
            C_end = 0.5 * quadratic_sum_proj_from_files(after_files, delta)

            dot_avggrad_delta = 0.0
            if isinstance(avg_grad_after, torch.Tensor) and avg_grad_after.numel() > 0:
                dot_avggrad_delta = float(torch.dot(avg_grad_after, delta))
            V_align = - dot_avggrad_delta

            avg_grad_before = avg_vector_from_files(before_files) if before_files else torch.tensor([], dtype=torch.float32)
            dot_avggrad_delta_before = 0.0
            if avg_grad_before.numel() > 0:
                dot_avggrad_delta_before = float(torch.dot(avg_grad_before, delta))
            V_align_before = - dot_avggrad_delta_before

            delta_norm_sum += delta_norm
            dot_after_sum += dot_avggrad_delta
            V_align_sum += V_align
            dot_before_sum += dot_avggrad_delta_before
            V_align_before_sum += V_align_before
            C_start_sum += C_start
            C_end_sum += C_end
            num_samples_sum += len(after_files)

            del init_adapter_vec, delta

            # cleanup before run tmp
            _before_tmp_path = os.path.dirname(before_files[0]) if before_files else os.path.join(tmp_dir, f"before_run{run_idx}")
            shutil.rmtree(_before_tmp_path, ignore_errors=True)

        if dist.is_initialized():
            dist.barrier()

    # final aggregation (means only)
    results = {}
    if is_main():
        n_runs_eff = float(n_runs) if n_runs > 0 else 1.0
        results = {
            "base_model_path": base_model_path,
            "lora_checkpoint_path": lora_checkpoint_path,
            "dataset_path": dataset_path,
            "delta_norm": delta_norm_sum / n_runs_eff,
            "dot_avggrad_delta": dot_after_sum / n_runs_eff,
            "V_align": V_align_sum / n_runs_eff,
            "dot_avggrad_delta_before": dot_before_sum / n_runs_eff,
            "V_align_before": V_align_before_sum / n_runs_eff,
            "C_start": C_start_sum / n_runs_eff,
            "C_end": C_end_sum / n_runs_eff,
            "num_samples_used": int(round(num_samples_sum / n_runs_eff)),
            "n_runs": n_runs,
        }

        atomic_write_json(results, _out_path)
        print(f"[INFO] Saved aggregated results to {_out_path}")

        # cleanup after files
        shutil.rmtree(_after_tmp, ignore_errors=True)

    # final cleanup
    if base_model_template is not None:
        # free CPU template
        del base_model_template
    torch.cuda.empty_cache()
    gc.collect()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    return results if is_main() else {}

# ---------------------------
# CLI
# ---------------------------
def build_parser():
    p = argparse.ArgumentParser(description="theorem_experiment_random (DDP/single-process; reuse_base_model optional)")
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--lora_checkpoint_path", type=str, required=True)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target", type=str, default="q_proj,v_proj")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_path", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--template_override", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--n_runs", type=int, default=5)
    p.add_argument("--tmp_dir", type=str, default=None)
    p.add_argument("--reuse_base_model", action="store_true",
                   help="若设置，则进程内预加载 base model template（CPU），每次 run deep-copy 后移动到 GPU，减少磁盘 load 次数（需较多 RAM）。")
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
        device=args.device,
        n_runs=args.n_runs,
        tmp_dir=args.tmp_dir,
        reuse_base_model=args.reuse_base_model,
    )
    if int(os.environ.get("RANK", "0")) == 0 and res:
        print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()