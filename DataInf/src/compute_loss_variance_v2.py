#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
import argparse
import math
import traceback
from typing import Optional, List, Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------- 新增：模板格式化函数 ----------
def alpaca_format(user_content: str, assistant_response: str) -> str:
    return f"### Instruction:\n{user_content}\n\n### Response:\n{assistant_response}"

def gsm8k_format(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"

def choose_template_by_dataset_path(dataset_path: str) -> str:
    """根据数据集路径自动选择模板：'alpaca' 或 'gsm8k'"""
    name = os.path.basename(dataset_path).lower()
    if "gsm8k" in name or "multiarith" in name or ("multi" in name and "arith" in name):
        return "gsm8k"
    return "alpaca"
# -----------------------------------------

def smart_parse_example(example: Dict) -> Tuple[str, str]:
    keys = set(example.keys())
    if "instruction" in keys and "output" in keys:
        instr = example.get("instruction", "")
        extra = example.get("input", "")
        if extra:
            instr = instr + "\n" + extra
        return instr, example.get("output", "")
    if "instruction" in keys and "response" in keys:
        instr = example.get("instruction", "")
        extra = example.get("input", "")
        if extra:
            instr = instr + "\n" + extra
        return instr, example.get("response", "")
    if "question" in keys and "answer" in keys:
        return example.get("question", ""), example.get("answer", "")
    if "goal" in keys and "target" in keys:
        return example.get("goal", ""), example.get("target", "")
    if "prompt" in keys and ("canonical_solution" in keys or "buggy_solution" in keys):
        instruction = example.get("instruction", "")
        full_prompt = f"{instruction}\n{example['prompt']}" if instruction else example['prompt']
        solution = example.get("canonical_solution", example.get("output", ""))
        return full_prompt, solution
    if "input" in keys and "output" in keys:
        return example.get("input", ""), example.get("output", "")
    return example.get("text", example.get("input", "")), example.get("label", example.get("response", ""))

def build_input_ids_list(dataset_path: str, tokenizer, max_length: int, max_samples: Optional[int]=None) -> List[List[int]]:
    ds = load_dataset("json", data_files={"test": dataset_path}, split="test")
    texts = []
    # 根据数据集路径选择模板
    template = choose_template_by_dataset_path(dataset_path)
    for ex in ds:
        pr, resp = smart_parse_example(ex)
        if pr is None:
            pr = ""
        if resp is None:
            resp = ""
        # ---------- 修改：使用模板格式化文本 ----------
        if template == "gsm8k":
            combined = gsm8k_format(pr, resp)
        else:
            combined = alpaca_format(pr, resp)
        combined = combined.strip()
        # -----------------------------------------
        if combined == "":
            continue
        texts.append(combined)
        if max_samples is not None and len(texts) >= max_samples:
            break
    if not texts:
        return []
    enc = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
    return enc["input_ids"]

def collate_batch(batch_input_ids: List[List[int]], pad_token_id: int, label_pad_token_id: int = -100):
    max_len = max(len(x) for x in batch_input_ids)
    input_ids = []
    attention_mask = []
    labels = []
    for ids in batch_input_ids:
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids_p = ids + [pad_token_id] * pad_len
            att = [1] * len(ids) + [0] * pad_len
            lab = ids + [label_pad_token_id] * pad_len
        else:
            ids_p = ids
            att = [1] * len(ids)
            lab = ids
        input_ids.append(ids_p)
        attention_mask.append(att)
        labels.append(lab)
    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }
    return batch

def _per_sample_losses_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    B, L, V = logits.shape
    per_losses = np.zeros(B, dtype=np.float64)
    per_counts = np.zeros(B, dtype=np.int64)
    for j in range(B):
        lj = labels[j]  # (L,)
        valid_mask = (lj != -100)
        n_tokens = int(valid_mask.sum().item())
        if n_tokens == 0:
            continue
        logits_j = logits[j]    # (L, V)
        labels_j = lj
        loss_j = F.cross_entropy(logits_j, labels_j, reduction='none', ignore_index=-100)  # (L,)
        loss_j = loss_j[valid_mask]
        per_losses[j] = float(loss_j.sum().item() / max(1, n_tokens))
        per_counts[j] = n_tokens
    return per_losses, per_counts

def compute_dataset_variance(model, tokenizer, dataset_path: str, batch_size: int, device_str: str, max_length: int, max_samples: Optional[int]=None) -> Tuple[Optional[float], Optional[str]]:
    if not os.path.isfile(dataset_path):
        return None, f"file not found: {dataset_path}"
    input_ids_list = build_input_ids_list(dataset_path, tokenizer, max_length, max_samples)
    if not input_ids_list:
        return None, "no valid samples"

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model.eval()

    count = 0
    mean = 0.0
    M2 = 0.0

    with torch.no_grad():
        i = 0
        while i < len(input_ids_list):
            batch_ids = input_ids_list[i:i+batch_size]
            batch = collate_batch(batch_ids, pad_id, label_pad_token_id=-100)
            try:
                try:
                    model_device = next(model.parameters()).device
                except StopIteration:
                    model_device = torch.device(device_str if isinstance(device_str, str) else device_str)
                batch = {k: v.to(model_device) for k, v in batch.items()}

                outputs = model(**batch, return_dict=True)
                logits = outputs.logits
                labels = batch["labels"]

                # compute per-sample losses without flattening whole batch
                try:
                    per_losses, per_counts = _per_sample_losses_from_logits(logits, labels)
                except RuntimeError as e_inner:
                    # if memory error inside per-sample loop, fallback to sample-by-sample
                    per_losses = np.zeros(logits.size(0), dtype=np.float64)
                    per_counts = np.zeros(logits.size(0), dtype=np.int64)
                    for j in range(logits.size(0)):
                        try:
                            logits_j = logits[j].unsqueeze(0)   # (1, L, V)
                            labels_j = labels[j].unsqueeze(0)   # (1, L)
                            pl, pc = _per_sample_losses_from_logits(logits_j, labels_j)
                            per_losses[j] = pl[0]
                            per_counts[j] = int(pc[0])
                            del logits_j, labels_j
                        except Exception:
                            per_losses[j] = 0.0
                            per_counts[j] = 0
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                # accumulate Welford stats
                for j in range(len(per_losses)):
                    n_tokens = int(per_counts[j])
                    if n_tokens <= 0:
                        continue
                    sample_loss = float(per_losses[j])
                    count += 1
                    delta = sample_loss - mean
                    mean += delta / count
                    delta2 = sample_loss - mean
                    M2 += delta * delta2

                del batch, outputs, logits, labels, per_losses, per_counts
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                i += batch_size

            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg or "cuda out of memory" in msg or "oom" in msg:
                    # fallback: process current batch sample-by-sample
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for k in range(len(batch_ids)):
                        single = [batch_ids[k]]
                        single_batch = collate_batch(single, pad_id)
                        try:
                            try:
                                model_device = next(model.parameters()).device
                            except StopIteration:
                                model_device = torch.device(device_str if isinstance(device_str, str) else device_str)
                            single_batch = {kk: vv.to(model_device) for kk, vv in single_batch.items()}
                            out = model(**single_batch, return_dict=True)
                            logits_s = out.logits
                            labels_s = single_batch["labels"]
                            pl_s, pc_s = _per_sample_losses_from_logits(logits_s, labels_s)
                            if int(pc_s[0]) > 0:
                                sample_loss = float(pl_s[0])
                                count += 1
                                delta = sample_loss - mean
                                mean += delta / count
                                delta2 = sample_loss - mean
                                M2 += delta * delta2
                            del single_batch, out, logits_s, labels_s, pl_s, pc_s
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            # skip this sample if it still fails
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                    i += batch_size
                else:
                    tb = traceback.format_exc()
                    return None, f"runtime_error: {tb}"

    if count < 1:
        return None, "no valid samples"
    variance = M2 / count
    return float(variance), None

def safe_fmt(x: Optional[float], prec: int = 12) -> str:
    if x is None:
        return "MISSING"
    if math.isinf(x):
        return "inf"
    return f"{x:.{prec}f}"

def load_model_with_fallback(base_model_path: str, lora_path: Optional[str], device_str: str, prefer_auto_on_fail: bool=False, verbose: bool=False):
    device_available = isinstance(device_str, str) and device_str.startswith("cuda")
    torch_dtype = torch.bfloat16 if (device_available and torch.cuda.is_bf16_supported()) else (torch.float16 if device_available else torch.float32)
    device_map_single = {"": "cuda:0"} if device_available else None

    try:
        if verbose:
            print(f"[LOAD] attempt single-device load base={base_model_path} device_map={device_map_single}", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map_single,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        base.config.use_cache = False
        model = base
        if lora_path and os.path.exists(lora_path):
            try:
                model = PeftModel.from_pretrained(base, lora_path, is_trainable=False)
            except Exception as e:
                if verbose:
                    print(f"[WARN] loading PEFT failed, using base. err: {e}", flush=True)
                model = base
        try:
            model.to(device_str)
        except Exception:
            pass
        return model
    except Exception:
        tb = traceback.format_exc()
        print("[ERROR] single-device load failed:", file=sys.stderr)
        print(tb, file=sys.stderr)
        if not prefer_auto_on_fail:
            raise

    try:
        if verbose:
            print("[LOAD] attempting fallback: device_map='auto' (shard across visible GPUs)", flush=True)
        base2 = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        base2.config.use_cache = False
        model2 = base2
        if lora_path and os.path.exists(lora_path):
            try:
                model2 = PeftModel.from_pretrained(base2, lora_path, is_trainable=False)
            except Exception as e:
                if verbose:
                    print(f"[WARN] loading PEFT on fallback failed, using base2. err: {e}", flush=True)
                model2 = base2
        return model2
    except Exception:
        tb2 = traceback.format_exc()
        print("[ERROR] fallback auto-shard load failed:", file=sys.stderr)
        print(tb2, file=sys.stderr)
        raise RuntimeError("Both single-device and auto-shard loading failed; see logs")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datainf_root", type=str, default=None, help="DataInf 根目录（默认脚本上级）")
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--checkpoint_root_epoch1", type=str, default=None, help="epoch1 专用 checkpoint 根目录")
    p.add_argument("--checkpoint_root", type=str, default=None, help="epoch5/其他 checkpoint 根目录")
    p.add_argument("--model_name_short", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--epoch", type=str, required=True, choices=["epoch_0","epoch_1","epoch_5"])
    p.add_argument("--method", type=str, required=True, choices=["sdft","sft"])
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--datasets", type=str, default="alpaca_eval,gsm8k,humaneval,multiarith,openfunction")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--output_root", type=str, required=True, help="最终存放目录（脚本会在此下新建子目录保存 txt）")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--prefer_auto_on_fail", action="store_true", help="如果单卡加载失败则尝试 device_map='auto' 在多卡上分片加载")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATAINF_ROOT = os.path.abspath(args.datainf_root) if args.datainf_root else os.path.normpath(os.path.join(script_dir, ".."))

    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_path = None
    if args.epoch == "epoch_0":
        lora_path = None
    elif args.epoch == "epoch_1":
        if args.checkpoint_root_epoch1:
            candidate = os.path.join(args.checkpoint_root_epoch1, args.model, args.method)
            if os.path.exists(candidate):
                lora_path = candidate
    else:
        if args.checkpoint_root:
            candidate = os.path.join(args.checkpoint_root, args.model, args.method)
            if os.path.exists(candidate):
                lora_path = candidate

    if args.verbose:
        print(f"[INFO] model={args.model} epoch={args.epoch} method={args.method} lora_path={lora_path} device={device}", flush=True)

    model = load_model_with_fallback(args.base_model_path, lora_path, device, prefer_auto_on_fail=args.prefer_auto_on_fail, verbose=args.verbose)

    DATASETS_MAP = {
        "gsm8k": os.path.join(args.data_root, "gsm8k", "gsm8k_test.json"),
        "openfunction": os.path.join(args.data_root, "openfunction", "openfunction_test.json"),
        "humaneval": os.path.join(args.data_root, "humanevalpack_test.jsonl"),
        "multiarith": os.path.join(args.data_root, "multiarith_test.json"),
        "alpaca_eval": os.path.join(args.data_root, "alpaca_eval.json")
    }

    dataset_names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    results: List[Tuple[str, Optional[float], Optional[float], Optional[str]]] = []

    for name in dataset_names:
        dpath = DATASETS_MAP.get(name)
        if not dpath:
            results.append((name, None, None, "unknown dataset"))
            continue
        if not os.path.isabs(dpath) or not os.path.exists(dpath):
            alt = os.path.join(args.data_root, os.path.basename(dpath))
            if os.path.isfile(alt):
                dpath = alt
        if not os.path.isfile(dpath):
            results.append((name, None, None, f"file not found: {dpath}"))
            continue
        if args.verbose:
            print(f"[RUN] computing dataset={name} path={dpath}", flush=True)
        var, err = compute_dataset_variance(model, tokenizer, dpath, batch_size=args.batch_size, device_str=device, max_length=args.max_length, max_samples=args.max_samples)
        if err is not None:
            results.append((name, None, None, err))
        else:
            inv = float("inf") if var == 0.0 else (1.0 / var)
            results.append((name, float(var), float(inv), None))

    base_out_root = os.path.abspath(args.output_root)
    os.makedirs(base_out_root, exist_ok=True)
    safe_folder_base = f"variance_results_{args.model}"
    out_folder = os.path.join(base_out_root, safe_folder_base)
    os.makedirs(out_folder, exist_ok=True)

    out_fname = f"{args.model_name_short}_{args.epoch}_{args.method}.txt"
    outpath = os.path.join(out_folder, out_fname)

    with open(outpath, "w", encoding="utf-8") as fo:
        fo.write(f"model: {args.model}\n")
        fo.write(f"epoch: {args.epoch}\n")
        fo.write(f"method: {args.method}\n")
        fo.write(f"datasets: {','.join(dataset_names)}\n\n")
        fo.write("per-dataset variance and inverse_of_variance:\n")
        for dn, var, inv, err in results:
            if err is not None:
                fo.write(f"{dn}: ERROR -> {err}\n")
            else:
                fo.write(f"{dn}: variance={safe_fmt(var,12)}, inverse_of_variance={safe_fmt(inv,12)}\n")
    print(outpath, flush=True)

if __name__ == "__main__":
    main()