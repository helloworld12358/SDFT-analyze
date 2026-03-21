#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute token-level loss tables (epoch x task) for each train dataset.

Goal:
- For each train dataset A:
  - rows: epoch_0 / epoch_1 / epoch_5
  - cols: alpaca_eval, gsm8k, humaneval, multiarith, openfunction
  - cell: mean token-level CE loss on that test task

Important:
- epoch_0 always uses base model only (no adapter).
- math tasks (gsm8k, multiarith) use GSM8K template.
- language/code tasks (alpaca_eval, humaneval, openfunction) use Alpaca template.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAINF_ROOT_DEFAULT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(DATAINF_ROOT_DEFAULT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gram_scheme_a_utils import (  # type: ignore  # noqa: E402
    DEFAULT_EPOCHS,
    DEFAULT_TASKS,
    DEFAULT_TRAIN_DATASETS,
    detect_datainf_root,
    resolve_checkpoint_path,
    resolve_result_root,
    resolve_sdft_root,
    write_rows_csv,
    write_unavailable_note,
)


TASKS_5 = ["alpaca_eval", "gsm8k", "humaneval", "multiarith", "openfunction"]
TEMPLATE_BY_TASK = {
    "alpaca_eval": "alpaca",
    "gsm8k": "gsm8k",
    "humaneval": "alpaca",
    "multiarith": "gsm8k",
    "openfunction": "alpaca",
}


def split_csv_arg(s: str, default: Sequence[str]) -> List[str]:
    if not s.strip():
        return list(default)
    out = [x.strip() for x in s.split(",") if x.strip()]
    return out if out else list(default)


def to_optional_int(v: int) -> Optional[int]:
    if v <= 0:
        return None
    return v


def choose_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def alpaca_format(user_content: str, assistant_response: str) -> str:
    return f"### Instruction:\n{user_content}\n\n### Response:\n{assistant_response}"


def gsm8k_format(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def smart_parse_example(example: Dict[str, object]) -> Tuple[str, str]:
    keys = set(example.keys())
    if "instruction" in keys and "output" in keys:
        instr = str(example.get("instruction", "") or "")
        extra = str(example.get("input", "") or "")
        if extra:
            instr = instr + "\n" + extra
        return instr, str(example.get("output", "") or "")
    if "instruction" in keys and "response" in keys:
        instr = str(example.get("instruction", "") or "")
        extra = str(example.get("input", "") or "")
        if extra:
            instr = instr + "\n" + extra
        return instr, str(example.get("response", "") or "")
    if "question" in keys and "answer" in keys:
        return str(example.get("question", "") or ""), str(example.get("answer", "") or "")
    if "goal" in keys and "target" in keys:
        return str(example.get("goal", "") or ""), str(example.get("target", "") or "")
    if "prompt" in keys and ("canonical_solution" in keys or "buggy_solution" in keys):
        instruction = str(example.get("instruction", "") or "")
        prompt = str(example.get("prompt", "") or "")
        full_prompt = f"{instruction}\n{prompt}" if instruction else prompt
        solution = str(example.get("canonical_solution", example.get("output", "")) or "")
        return full_prompt, solution
    if "input" in keys and "output" in keys:
        return str(example.get("input", "") or ""), str(example.get("output", "") or "")
    return str(example.get("text", example.get("input", "")) or ""), str(example.get("label", example.get("response", "")) or "")


def collate_batch(batch_input_ids: List[List[int]], pad_token_id: int, label_pad_token_id: int = -100) -> Dict[str, torch.Tensor]:
    max_len = max(len(x) for x in batch_input_ids)
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    labels: List[List[int]] = []
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
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def resolve_eval_dataset_paths(data_root: str) -> Dict[str, str]:
    def pick(cands: Sequence[str]) -> str:
        for p in cands:
            if os.path.isfile(p):
                return p
        return cands[0]

    return {
        "alpaca_eval": pick(
            [
                os.path.join(data_root, "alpaca_eval.json"),
                os.path.join(data_root, "alpaca", "alpaca_eval.json"),
            ]
        ),
        "gsm8k": pick(
            [
                os.path.join(data_root, "gsm8k", "gsm8k_test.json"),
                os.path.join(data_root, "gsm8k_test.json"),
            ]
        ),
        "humaneval": pick(
            [
                os.path.join(data_root, "humanevalpack_test.jsonl"),
                os.path.join(data_root, "humaneval", "humanevalpack_test.jsonl"),
                os.path.join(data_root, "humaneval_test.jsonl"),
            ]
        ),
        "multiarith": pick(
            [
                os.path.join(data_root, "multiarith_test.json"),
                os.path.join(data_root, "multiarith", "multiarith_test.json"),
            ]
        ),
        "openfunction": pick(
            [
                os.path.join(data_root, "openfunction", "openfunction_test.json"),
                os.path.join(data_root, "openfunction_test.json"),
            ]
        ),
    }


def build_input_ids_list(
    dataset_path: str,
    task_name: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    max_samples: Optional[int],
    truncation: bool,
) -> Tuple[List[List[int]], Optional[str]]:
    try:
        ds = load_dataset("json", data_files={"eval": dataset_path}, split="eval")
    except Exception as e:
        return [], f"failed to load dataset: {e}"

    texts: List[str] = []
    template = TEMPLATE_BY_TASK.get(task_name, "alpaca")
    for ex in ds:
        prompt, resp = smart_parse_example(ex)
        if prompt is None:
            prompt = ""
        if resp is None:
            resp = ""
        if template == "gsm8k":
            combined = gsm8k_format(prompt, resp).strip()
        else:
            combined = alpaca_format(prompt, resp).strip()
        if not combined:
            continue
        texts.append(combined)
        if max_samples is not None and len(texts) >= max_samples:
            break

    if not texts:
        return [], "no valid samples after parsing/template formatting"

    input_ids_list: List[List[int]] = []
    try:
        if truncation:
            enc = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
            input_ids_list = list(enc["input_ids"])
        else:
            # no truncation mode: keep original tokenized length
            for t in texts:
                ids = tokenizer(t, truncation=False, padding=False)["input_ids"]
                if isinstance(ids, list):
                    input_ids_list.append(ids)
    except Exception as e:
        return [], f"tokenization failed: {e}"
    return input_ids_list, None


def _is_oom_error(e: BaseException) -> bool:
    s = str(e).lower()
    return "out of memory" in s or "cuda out of memory" in s or "oom" in s


def _probe_batch_forward(
    model: torch.nn.Module,
    input_ids_list: List[List[int]],
    pad_id: int,
    device: str,
    batch_size: int,
    probe_batches: int,
) -> Tuple[bool, Optional[str], bool]:
    """Return: (ok, error_message, is_oom)."""
    if batch_size <= 0:
        return False, "batch_size<=0", False
    if not input_ids_list:
        return False, "empty input_ids_list", False

    sample_pool = sorted(input_ids_list, key=lambda x: len(x), reverse=True)
    n_batches = max(1, int(probe_batches))
    with torch.no_grad():
        for b in range(n_batches):
            start = b * batch_size
            if start >= len(sample_pool):
                break
            batch_ids = sample_pool[start : start + batch_size]
            if not batch_ids:
                break
            batch = collate_batch(batch_ids, pad_id, label_pad_token_id=-100)
            try:
                try:
                    model_device = next(model.parameters()).device
                except StopIteration:
                    model_device = torch.device(device)
                batch = {k: v.to(model_device) for k, v in batch.items()}
                out = model(**batch, return_dict=True)
                _ = out.loss
                del out, batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                is_oom = _is_oom_error(e)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False, str(e), is_oom
    return True, None, False


def detect_max_batch_size(
    model: torch.nn.Module,
    input_ids_list: List[List[int]],
    pad_id: int,
    device: str,
    requested_bs: int,
    probe_max_bs: int,
    probe_batches: int,
    verbose: bool,
) -> Tuple[int, Optional[str]]:
    if not input_ids_list:
        return 1, "empty input list"
    upper = max(1, min(len(input_ids_list), max(requested_bs, probe_max_bs)))

    ok1, err1, oom1 = _probe_batch_forward(
        model=model,
        input_ids_list=input_ids_list,
        pad_id=pad_id,
        device=device,
        batch_size=1,
        probe_batches=probe_batches,
    )
    if not ok1:
        return 1, f"bs=1 probe failed: {err1}"

    lo = 1
    hi = upper
    best = 1
    non_oom_note: Optional[str] = None

    while lo <= hi:
        mid = (lo + hi) // 2
        ok, err, is_oom = _probe_batch_forward(
            model=model,
            input_ids_list=input_ids_list,
            pad_id=pad_id,
            device=device,
            batch_size=mid,
            probe_batches=probe_batches,
        )
        if ok:
            best = mid
            lo = mid + 1
        else:
            if is_oom:
                hi = mid - 1
            else:
                non_oom_note = f"non-oom probe error at bs={mid}: {err}"
                hi = mid - 1
        if verbose:
            state = "OK" if ok else ("OOM" if is_oom else "ERR")
            print(f"[BS-PROBE] bs={mid} -> {state}", flush=True)

    return max(1, best), non_oom_note


def load_model_with_optional_lora(
    base_model_path: str,
    lora_path: Optional[str],
    device: str,
    prefer_auto_on_fail: bool = False,
    verbose: bool = False,
) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
    dtype = choose_dtype(device)
    device_map_single = {"": "cuda:0"} if device.startswith("cuda") else None

    try:
        if verbose:
            print(f"[LOAD] base={base_model_path} lora={lora_path} device={device}", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map=device_map_single,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        base.config.use_cache = False
        model = base
        if lora_path and os.path.isdir(lora_path):
            model = PeftModel.from_pretrained(base, lora_path, is_trainable=False)
        try:
            model.to(device)
        except Exception:
            pass
        model.eval()
        return model, None
    except Exception:
        if not prefer_auto_on_fail:
            return None, traceback.format_exc()

    try:
        if verbose:
            print("[LOAD] fallback device_map=auto", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        base.config.use_cache = False
        model = base
        if lora_path and os.path.isdir(lora_path):
            model = PeftModel.from_pretrained(base, lora_path, is_trainable=False)
        model.eval()
        return model, None
    except Exception:
        return None, traceback.format_exc()


def compute_dataset_loss(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    dataset_path: str,
    task_name: str,
    batch_size: int,
    device: str,
    max_length: int,
    max_samples: Optional[int],
    auto_batch_size: bool,
    batch_probe_max_bs: int,
    batch_probe_batches: int,
    verbose: bool,
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    if not os.path.isfile(dataset_path):
        return None, f"dataset file not found: {dataset_path}"

    truncation = max_length > 0
    input_ids_list, build_err = build_input_ids_list(
        dataset_path=dataset_path,
        task_name=task_name,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
        truncation=truncation,
    )
    if build_err is not None:
        return None, build_err
    if not input_ids_list:
        return None, "no valid tokenized samples"

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    max_input_tokens_observed = max(len(x) for x in input_ids_list) if input_ids_list else 0
    model_limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if (not truncation) and model_limit and max_input_tokens_observed > int(model_limit):
        return (
            None,
            f"no-truncation requested but sample length={max_input_tokens_observed} exceeds model limit={int(model_limit)}; set --max_length <= {int(model_limit)}",
        )

    total_loss_times_tokens = 0.0
    total_tokens = 0
    total_samples = 0

    i = 0
    requested_bs = max(1, int(batch_size))
    effective_bs = requested_bs
    probe_note: Optional[str] = None
    if auto_batch_size:
        effective_bs, probe_note = detect_max_batch_size(
            model=model,
            input_ids_list=input_ids_list,
            pad_id=pad_id,
            device=device,
            requested_bs=requested_bs,
            probe_max_bs=max(1, int(batch_probe_max_bs)),
            probe_batches=max(1, int(batch_probe_batches)),
            verbose=verbose,
        )
        if verbose:
            print(f"[BS-PROBE] selected_bs={effective_bs} requested_bs={requested_bs}", flush=True)
    model.eval()

    with torch.no_grad():
        while i < len(input_ids_list):
            batch_ids = input_ids_list[i : i + effective_bs]
            batch = collate_batch(batch_ids, pad_id, label_pad_token_id=-100)
            try:
                try:
                    model_device = next(model.parameters()).device
                except StopIteration:
                    model_device = torch.device(device)
                batch = {k: v.to(model_device) for k, v in batch.items()}

                outputs = model(**batch, return_dict=True)
                loss_val = float(outputs.loss.detach().float().item())
                labels = batch["labels"]
                token_count = int((labels[:, 1:] != -100).sum().item())
                if token_count > 0 and math.isfinite(loss_val):
                    total_loss_times_tokens += loss_val * token_count
                    total_tokens += token_count
                    total_samples += len(batch_ids)
                i += effective_bs

                del outputs, labels, batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if _is_oom_error(e) and effective_bs > 1:
                    effective_bs = max(1, effective_bs // 2)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                return None, f"runtime_error: {e}"
            except Exception:
                return None, traceback.format_exc()

    if total_tokens <= 0:
        return None, "no valid tokens after forward pass"

    loss_mean_token = total_loss_times_tokens / float(total_tokens)
    return {
        "loss_mean_token": float(loss_mean_token),
        "n_samples": int(total_samples),
        "n_tokens": int(total_tokens),
        "batch_size_requested": int(requested_bs),
        "batch_size_used": int(effective_bs),
        "auto_batch_size": bool(auto_batch_size),
        "batch_probe_note": probe_note or "",
        "truncation_enabled": bool(truncation),
        "max_length_setting": int(max_length),
        "max_input_tokens_observed": int(max_input_tokens_observed),
        "model_max_position_embeddings": int(model_limit) if model_limit else None,
    }, None


def fmt_loss(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{x:.6f}"


def render_dataset_table_txt(
    train_dataset: str,
    method: str,
    rows: Sequence[Dict[str, object]],
    epochs: Sequence[str],
    tasks: Sequence[str],
) -> str:
    lookup: Dict[Tuple[str, str], Optional[float]] = {}
    for r in rows:
        ep = str(r.get("epoch", ""))
        task = str(r.get("test_task", ""))
        status = str(r.get("status", ""))
        loss = r.get("loss_mean_token")
        v: Optional[float] = None
        try:
            if status == "ok" and loss is not None:
                v = float(loss)
        except Exception:
            v = None
        lookup[(ep, task)] = v

    header_first = "epoch\\task"
    col_names = list(tasks)
    widths: Dict[str, int] = {header_first: len(header_first)}
    for c in col_names:
        widths[c] = max(len(c), 10)
    widths[header_first] = max(widths[header_first], max(len(e) for e in epochs))

    lines: List[str] = []
    lines.append(f"train_dataset={train_dataset} | method={method} | metric=mean_token_ce_loss (lower is better)")
    lines.append("")

    head = header_first.ljust(widths[header_first]) + " | " + " | ".join(c.ljust(widths[c]) for c in col_names)
    sep = "-" * widths[header_first] + "-+-" + "-+-".join("-" * widths[c] for c in col_names)
    lines.append(head)
    lines.append(sep)
    for ep in epochs:
        vals = [fmt_loss(lookup.get((ep, t))) for t in col_names]
        row = ep.ljust(widths[header_first]) + " | " + " | ".join(vals[i].ljust(widths[col_names[i]]) for i in range(len(col_names)))
        lines.append(row)
    lines.append("")
    return "\n".join(lines)


def save_json(path: str, obj: object) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Compute loss tables (3x5) per train dataset.")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default="", help="Default: <result_root>/loss_eval")
    p.add_argument("--train_dataset", type=str, default="", help="single train dataset name")
    p.add_argument("--all_train_datasets", action="store_true")
    p.add_argument("--methods", type=str, default="sft,sdft", help="comma-separated: sft,sdft")
    p.add_argument("--epochs", type=str, default="epoch_0,epoch_1,epoch_5")
    p.add_argument("--tasks", type=str, default="alpaca_eval,gsm8k,humaneval,multiarith,openfunction")
    p.add_argument("--base_model_path", type=str, default="")
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_length", type=int, default=1024, help=">0: truncate to this length; <=0: no truncation")
    p.add_argument("--auto_batch_size", action="store_true", help="auto probe max non-OOM batch size for each task")
    p.add_argument("--batch_probe_max_bs", type=int, default=64, help="upper bound when probing auto batch size")
    p.add_argument("--batch_probe_batches", type=int, default=1, help="how many worst-case batches to test per probe step")
    p.add_argument("--max_samples", type=int, default=0, help="<=0 means use all samples")
    p.add_argument("--device", type=str, default="", help="Default: cuda:0 if available else cpu")
    p.add_argument("--prefer_auto_on_fail", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    sdft_root = resolve_sdft_root(datainf_root)
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    output_root = args.output_root.strip() or os.path.join(result_root, "loss_eval")
    os.makedirs(output_root, exist_ok=True)

    if args.all_train_datasets:
        train_datasets = list(DEFAULT_TRAIN_DATASETS)
    else:
        train_datasets = split_csv_arg(args.train_dataset, DEFAULT_TRAIN_DATASETS)

    methods = split_csv_arg(args.methods, ["sft", "sdft"])
    methods = [m for m in methods if m in ("sft", "sdft")]
    if not methods:
        raise ValueError(f"--methods invalid: {args.methods}")

    epochs = split_csv_arg(args.epochs, DEFAULT_EPOCHS)
    tasks = split_csv_arg(args.tasks, TASKS_5)
    if len(tasks) != 5:
        raise ValueError(f"--tasks must contain exactly 5 tasks, got {len(tasks)}: {tasks}")

    device = args.device.strip() or ("cuda:0" if torch.cuda.is_available() else "cpu")
    max_samples = to_optional_int(args.max_samples)

    base_model_path = args.base_model_path.strip() or os.path.join(sdft_root, "model", "Llama-2-7b-chat-hf")
    data_root = args.data_root.strip() or os.path.join(sdft_root, "data")
    eval_paths = resolve_eval_dataset_paths(data_root)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    epoch_tag = epochs[0] if len(epochs) == 1 else "all_epochs"

    all_rows: List[Dict[str, object]] = []
    unavailable_rows: List[Dict[str, object]] = []

    by_ds_root = os.path.join(output_root, "by_train_dataset")
    os.makedirs(by_ds_root, exist_ok=True)

    for train_dataset in train_datasets:
        ds_dir = os.path.join(by_ds_root, train_dataset)
        os.makedirs(ds_dir, exist_ok=True)

        for method in methods:
            ds_rows: List[Dict[str, object]] = []
            for epoch in epochs:
                # epoch_0 must not use adapter
                lora_path: Optional[str]
                if epoch == "epoch_0":
                    lora_path = None
                else:
                    lora_path = resolve_checkpoint_path(sdft_root, epoch, train_dataset, method)

                if epoch != "epoch_0" and (not lora_path or not os.path.isdir(lora_path)):
                    reason = f"checkpoint missing for {train_dataset}/{epoch}/{method}"
                    for task in tasks:
                        row = {
                            "train_dataset": train_dataset,
                            "method": method,
                            "epoch": epoch,
                            "test_task": task,
                            "template_used": TEMPLATE_BY_TASK.get(task, "alpaca"),
                            "dataset_path": eval_paths.get(task, ""),
                            "loss_mean_token": None,
                            "n_samples": 0,
                        "n_tokens": 0,
                        "batch_size_requested": int(args.batch_size),
                        "batch_size_used": 0,
                        "auto_batch_size": bool(args.auto_batch_size),
                        "batch_probe_note": "",
                        "truncation_enabled": bool(args.max_length > 0),
                        "max_length_setting": int(args.max_length),
                        "max_input_tokens_observed": 0,
                        "model_max_position_embeddings": None,
                        "status": "missing_checkpoint",
                        "error": reason,
                            "base_model_path": base_model_path,
                            "lora_path": lora_path or "",
                        }
                        ds_rows.append(row)
                        unavailable_rows.append(dict(row))
                    continue

                model, load_err = load_model_with_optional_lora(
                    base_model_path=base_model_path,
                    lora_path=lora_path,
                    device=device,
                    prefer_auto_on_fail=args.prefer_auto_on_fail,
                    verbose=args.verbose,
                )
                if load_err is not None or model is None:
                    for task in tasks:
                        row = {
                            "train_dataset": train_dataset,
                            "method": method,
                            "epoch": epoch,
                            "test_task": task,
                            "template_used": TEMPLATE_BY_TASK.get(task, "alpaca"),
                            "dataset_path": eval_paths.get(task, ""),
                            "loss_mean_token": None,
                            "n_samples": 0,
                            "n_tokens": 0,
                            "batch_size_requested": int(args.batch_size),
                            "batch_size_used": 0,
                            "auto_batch_size": bool(args.auto_batch_size),
                            "batch_probe_note": "",
                            "truncation_enabled": bool(args.max_length > 0),
                            "max_length_setting": int(args.max_length),
                            "max_input_tokens_observed": 0,
                            "model_max_position_embeddings": None,
                            "status": "model_load_error",
                            "error": load_err or "unknown load error",
                            "base_model_path": base_model_path,
                            "lora_path": lora_path or "",
                        }
                        ds_rows.append(row)
                        unavailable_rows.append(dict(row))
                    continue

                for task in tasks:
                    dpath = eval_paths.get(task, "")
                    row = {
                        "train_dataset": train_dataset,
                        "method": method,
                        "epoch": epoch,
                        "test_task": task,
                        "template_used": TEMPLATE_BY_TASK.get(task, "alpaca"),
                        "dataset_path": dpath,
                        "loss_mean_token": None,
                        "n_samples": 0,
                        "n_tokens": 0,
                        "batch_size_requested": int(args.batch_size),
                        "batch_size_used": 0,
                        "auto_batch_size": bool(args.auto_batch_size),
                        "batch_probe_note": "",
                        "truncation_enabled": bool(args.max_length > 0),
                        "max_length_setting": int(args.max_length),
                        "max_input_tokens_observed": 0,
                        "model_max_position_embeddings": None,
                        "status": "init",
                        "error": "",
                        "base_model_path": base_model_path,
                        "lora_path": lora_path or "",
                    }
                    if not dpath or not os.path.isfile(dpath):
                        row["status"] = "missing_dataset_file"
                        row["error"] = f"dataset file not found: {dpath}"
                        ds_rows.append(row)
                        unavailable_rows.append(dict(row))
                        continue

                    metrics, err = compute_dataset_loss(
                        model=model,
                        tokenizer=tokenizer,
                        dataset_path=dpath,
                        task_name=task,
                        batch_size=args.batch_size,
                        device=device,
                        max_length=args.max_length,
                        max_samples=max_samples,
                        auto_batch_size=args.auto_batch_size,
                        batch_probe_max_bs=args.batch_probe_max_bs,
                        batch_probe_batches=args.batch_probe_batches,
                        verbose=args.verbose,
                    )
                    if err is not None or metrics is None:
                        row["status"] = "compute_error"
                        row["error"] = err or "unknown compute error"
                        ds_rows.append(row)
                        unavailable_rows.append(dict(row))
                    else:
                        row["status"] = "ok"
                        row["loss_mean_token"] = float(metrics["loss_mean_token"])
                        row["n_samples"] = int(metrics["n_samples"])
                        row["n_tokens"] = int(metrics["n_tokens"])
                        row["batch_size_requested"] = int(metrics["batch_size_requested"])
                        row["batch_size_used"] = int(metrics["batch_size_used"])
                        row["auto_batch_size"] = bool(metrics["auto_batch_size"])
                        row["batch_probe_note"] = str(metrics["batch_probe_note"])
                        row["truncation_enabled"] = bool(metrics["truncation_enabled"])
                        row["max_length_setting"] = int(metrics["max_length_setting"])
                        row["max_input_tokens_observed"] = int(metrics["max_input_tokens_observed"])
                        row["model_max_position_embeddings"] = metrics["model_max_position_embeddings"]
                        ds_rows.append(row)

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_rows.extend(ds_rows)

            ds_json = os.path.join(ds_dir, f"loss_rows_{train_dataset}_{method}_{epoch_tag}.json")
            ds_csv = os.path.join(ds_dir, f"loss_rows_{train_dataset}_{method}_{epoch_tag}.csv")
            ds_txt = os.path.join(ds_dir, f"loss_table_{train_dataset}_{method}_{epoch_tag}.txt")
            save_json(ds_json, ds_rows)
            write_rows_csv(ds_csv, ds_rows)
            with open(ds_txt, "w", encoding="utf-8") as f:
                f.write(render_dataset_table_txt(train_dataset, method, ds_rows, epochs, tasks))
            print(os.path.abspath(ds_json))
            print(os.path.abspath(ds_csv))
            print(os.path.abspath(ds_txt))

    train_tag = "all_datasets" if len(train_datasets) != 1 else train_datasets[0]
    method_tag = "__".join(methods)
    all_json = os.path.join(output_root, f"loss_rows_all_{train_tag}_{method_tag}_{epoch_tag}.json")
    all_csv = os.path.join(output_root, f"loss_rows_all_{train_tag}_{method_tag}_{epoch_tag}.csv")
    save_json(all_json, all_rows)
    write_rows_csv(all_csv, all_rows)
    print(os.path.abspath(all_json))
    print(os.path.abspath(all_csv))

    unavailable_json = os.path.join(output_root, f"unavailable_loss_eval_{train_tag}_{method_tag}_{epoch_tag}.json")
    if unavailable_rows:
        save_json(unavailable_json, unavailable_rows)
    else:
        write_unavailable_note(
            unavailable_json,
            reason="none",
            context={"message": "all rows are available"},
        )
    print(os.path.abspath(unavailable_json))


if __name__ == "__main__":
    main()
