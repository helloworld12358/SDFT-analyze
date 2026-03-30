#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forward-only loss collection for theory diagnostics.

This script computes per-sample loss statistics (Li, Lbar_i, Ti) and stores:
- sample-level rows (csv/json)
- random token-loss subsamples (csv/json)
- random token-sequence probes for dependence diagnostics (csv/json)

Constraints:
- no training
- no parameter updates
- no backward/grad/Hessian/MI
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAINF_ROOT_DEFAULT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(DATAINF_ROOT_DEFAULT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gram_scheme_a_utils import (  # type: ignore  # noqa: E402
    DEFAULT_TRAIN_DATASETS,
    detect_datainf_root,
    ensure_dir,
    normalize_epoch_list,
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

CODE_HINT_RE = re.compile(r"(```|def\s+\w+\(|class\s+\w+|#include|public\s+static\s+void|function\s+\w+\(|=>|<\w+>)")
MATH_HINT_RE = re.compile(r"(\d+\s*[\+\-\*/=]\s*\d+|\\frac|\\sum|\\int|\btheorem\b|\bproof\b)")


def split_csv_arg(s: str, default: Sequence[str]) -> List[str]:
    if not s.strip():
        return list(default)
    out = [x.strip() for x in s.split(",") if x.strip()]
    return out if out else list(default)


def choose_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


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


def alpaca_format(user_content: str, assistant_response: str) -> str:
    return f"### Instruction:\n{user_content}\n\n### Response:\n{assistant_response}"


def gsm8k_format(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def build_text_for_task(task: str, prompt: str, answer: str) -> str:
    template = TEMPLATE_BY_TASK.get(task, "alpaca")
    if template == "gsm8k":
        return gsm8k_format(prompt, answer).strip()
    return alpaca_format(prompt, answer).strip()


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


def iter_records_stream(path: str) -> Iterator[Dict[str, object]]:
    try:
        ds = load_dataset("json", data_files={"eval": path}, split="eval", streaming=True)
        for row in ds:
            if isinstance(row, dict):
                yield dict(row)
        return
    except Exception:
        pass

    if path.endswith(".jsonl") or path.endswith(".ndjson"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        yield obj
                except Exception:
                    continue
        return

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        for row in obj:
            if isinstance(row, dict):
                yield row
    elif isinstance(obj, dict):
        for key in ("data", "records", "examples", "items"):
            val = obj.get(key)
            if isinstance(val, list):
                for row in val:
                    if isinstance(row, dict):
                        yield row
                return
        yield obj


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


def _is_oom_error(e: BaseException) -> bool:
    s = str(e).lower()
    return "out of memory" in s or "cuda out of memory" in s or "oom" in s


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


def to_bool_flag(x: bool) -> int:
    return 1 if bool(x) else 0


def stable_hash_uniform(seed: int, key: str) -> float:
    raw = f"{seed}|{key}".encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    v = int.from_bytes(digest, "big", signed=False)
    return v / float(2**64 - 1)


def choose_meta_value(example: Dict[str, object], keys: Sequence[str], default: str = "") -> str:
    for k in keys:
        if k in example and example.get(k) is not None:
            v = str(example.get(k))
            if v.strip():
                return v.strip()
    return default


def contains_code(text: str) -> bool:
    return bool(CODE_HINT_RE.search(text))


def contains_math(text: str) -> bool:
    return bool(MATH_HINT_RE.search(text))


def save_json(path: str, obj: object) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def maybe_write_parquet(csv_path: str) -> Optional[str]:
    try:
        import pandas as pd  # type: ignore

        out = os.path.splitext(csv_path)[0] + ".parquet"
        df = pd.read_csv(csv_path)
        df.to_parquet(out, index=False)
        return out
    except Exception:
        return None


@dataclass
class Combo:
    train_dataset: str
    method: str
    epoch: str
    task: str
    eval_path: str

    @property
    def key(self) -> str:
        return f"{self.train_dataset}__{self.method}__{self.epoch}__{self.task}"


def build_combos(
    train_datasets: Sequence[str],
    methods: Sequence[str],
    epochs: Sequence[str],
    tasks: Sequence[str],
    eval_paths: Dict[str, str],
) -> List[Combo]:
    combos: List[Combo] = []
    for train_dataset in train_datasets:
        for method in methods:
            for epoch in epochs:
                for task in tasks:
                    combos.append(
                        Combo(
                            train_dataset=train_dataset,
                            method=method,
                            epoch=epoch,
                            task=task,
                            eval_path=eval_paths.get(task, ""),
                        )
                    )
    combos.sort(key=lambda x: (x.train_dataset, x.method, x.epoch, x.task))
    return combos


def select_shard(combos: Sequence[Combo], shard_count: int, shard_index: int) -> List[Combo]:
    if shard_count <= 1:
        return list(combos)
    out: List[Combo] = []
    for idx, combo in enumerate(combos):
        if idx % shard_count == shard_index:
            out.append(combo)
    return out

def run_one_combo(
    combo: Combo,
    sdft_root: str,
    output_root: str,
    base_model_path: str,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int,
    max_length: int,
    max_samples_per_task: Optional[int],
    token_sample_rate: float,
    token_sample_cap_per_sample: int,
    seq_probe_sample_rate: float,
    seq_probe_max_tokens: int,
    flush_every: int,
    seed: int,
    prefer_auto_on_fail: bool,
    verbose: bool,
) -> Dict[str, object]:
    combo_dir = ensure_dir(os.path.join(output_root, "by_combo", combo.train_dataset, combo.method, combo.epoch, combo.task))
    sample_csv = os.path.join(combo_dir, "sample_stats.csv")
    token_csv = os.path.join(combo_dir, "token_subsample_stats.csv")
    seq_csv = os.path.join(combo_dir, "token_sequence_probe_stats.csv")
    state_json = os.path.join(combo_dir, "state.json")
    unavailable_json = os.path.join(combo_dir, "unavailable_forward_collect.json")

    if os.path.isfile(state_json):
        try:
            state_obj = json.load(open(state_json, "r", encoding="utf-8"))
            if isinstance(state_obj, dict) and bool(state_obj.get("done")):
                return {
                    "combo_key": combo.key,
                    "status": "skipped_done",
                    "sample_csv": sample_csv,
                    "token_csv": token_csv,
                    "seq_csv": seq_csv,
                    "state_json": state_json,
                }
        except Exception:
            pass

    if not combo.eval_path or (not os.path.isfile(combo.eval_path)):
        reason = f"eval dataset missing: {combo.eval_path}"
        write_unavailable_note(unavailable_json, reason=reason, context={"combo_key": combo.key})
        return {"combo_key": combo.key, "status": "unavailable_eval", "error": reason, "unavailable_json": unavailable_json}

    lora_path: Optional[str] = resolve_checkpoint_path(sdft_root, combo.epoch, combo.train_dataset, combo.method)
    if not lora_path or not os.path.isdir(lora_path):
        reason = f"checkpoint missing for {combo.train_dataset}/{combo.epoch}/{combo.method}"
        write_unavailable_note(unavailable_json, reason=reason, context={"combo_key": combo.key})
        return {"combo_key": combo.key, "status": "unavailable_ckpt", "error": reason, "unavailable_json": unavailable_json}

    model, load_err = load_model_with_optional_lora(
        base_model_path=base_model_path,
        lora_path=lora_path,
        device=device,
        prefer_auto_on_fail=prefer_auto_on_fail,
        verbose=verbose,
    )
    if model is None:
        write_unavailable_note(unavailable_json, reason="model load failed", context={"combo_key": combo.key, "error": load_err})
        return {"combo_key": combo.key, "status": "model_load_error", "error": load_err or "", "unavailable_json": unavailable_json}

    model_limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    start_idx = 0
    if os.path.isfile(state_json):
        try:
            obj = json.load(open(state_json, "r", encoding="utf-8"))
            if isinstance(obj, dict):
                start_idx = int(obj.get("next_sample_idx", 0))
        except Exception:
            start_idx = 0

    sample_header = [
        "combo_key",
        "train_dataset",
        "method",
        "epoch",
        "task",
        "sample_idx",
        "sample_uid",
        "doc_id",
        "source_id",
        "domain_label",
        "input_chars",
        "output_chars",
        "input_words",
        "output_words",
        "contains_code",
        "contains_math",
        "forced_truncation",
        "Ti",
        "Li",
        "Lbar_i",
        "eval_path",
    ]
    token_header = [
        "combo_key",
        "train_dataset",
        "method",
        "epoch",
        "task",
        "sample_idx",
        "token_pos",
        "u_i_t",
        "Ti",
        "Li",
        "Lbar_i",
        "sample_uid",
        "domain_label",
    ]
    seq_header = [
        "combo_key",
        "train_dataset",
        "method",
        "epoch",
        "task",
        "sample_idx",
        "token_pos",
        "u_i_t",
        "Ti",
        "sample_uid",
        "domain_label",
    ]

    if not os.path.isfile(sample_csv):
        with open(sample_csv, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=sample_header).writeheader()
    if not os.path.isfile(token_csv):
        with open(token_csv, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=token_header).writeheader()
    if not os.path.isfile(seq_csv):
        with open(seq_csv, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=seq_header).writeheader()

    sample_rows_buf: List[Dict[str, object]] = []
    token_rows_buf: List[Dict[str, object]] = []
    seq_rows_buf: List[Dict[str, object]] = []
    processed = 0
    total_tokens = 0
    oom_retries = 0
    current_batch_size = max(1, int(batch_size))
    done = False

    def flush_rows(next_sample_idx: int) -> None:
        nonlocal sample_rows_buf, token_rows_buf, seq_rows_buf
        if sample_rows_buf:
            with open(sample_csv, "a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sample_header)
                w.writerows(sample_rows_buf)
            sample_rows_buf = []
        if token_rows_buf:
            with open(token_csv, "a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=token_header)
                w.writerows(token_rows_buf)
            token_rows_buf = []
        if seq_rows_buf:
            with open(seq_csv, "a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=seq_header)
                w.writerows(seq_rows_buf)
            seq_rows_buf = []
        save_json(
            state_json,
            {
                "combo_key": combo.key,
                "next_sample_idx": int(next_sample_idx),
                "done": False,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    pending_ids: List[List[int]] = []
    pending_meta: List[Dict[str, object]] = []

    def run_pending_batch() -> None:
        nonlocal pending_ids, pending_meta, processed, total_tokens, current_batch_size, oom_retries
        if not pending_ids:
            return
        batch = collate_batch(pending_ids, pad_id, label_pad_token_id=-100)
        try:
            with torch.no_grad():
                try:
                    model_device = next(model.parameters()).device
                except StopIteration:
                    model_device = torch.device(device)
                batch = {k: v.to(model_device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=True,
                )
                logits = outputs.logits
                labels = batch["labels"]
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                valid_mask = shift_labels != -100
                safe_labels = torch.where(valid_mask, shift_labels, torch.zeros_like(shift_labels))
                token_log_probs = F.log_softmax(shift_logits.float(), dim=-1)
                token_nll = -token_log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
                token_nll = torch.where(valid_mask, token_nll, torch.zeros_like(token_nll))
                nll_cpu = token_nll.detach().cpu()
                mask_cpu = valid_mask.detach().cpu()

                for row_idx in range(nll_cpu.shape[0]):
                    vals = nll_cpu[row_idx][mask_cpu[row_idx]].tolist()
                    if not vals:
                        continue
                    ti = int(len(vals))
                    li = float(sum(vals))
                    lbar = float(li / float(ti))
                    meta = pending_meta[row_idx]

                    sample_rows_buf.append(
                        {
                            "combo_key": combo.key,
                            "train_dataset": combo.train_dataset,
                            "method": combo.method,
                            "epoch": combo.epoch,
                            "task": combo.task,
                            "sample_idx": int(meta["sample_idx"]),
                            "sample_uid": str(meta["sample_uid"]),
                            "doc_id": str(meta["doc_id"]),
                            "source_id": str(meta["source_id"]),
                            "domain_label": str(meta["domain_label"]),
                            "input_chars": int(meta["input_chars"]),
                            "output_chars": int(meta["output_chars"]),
                            "input_words": int(meta["input_words"]),
                            "output_words": int(meta["output_words"]),
                            "contains_code": int(meta["contains_code"]),
                            "contains_math": int(meta["contains_math"]),
                            "forced_truncation": int(meta["forced_truncation"]),
                            "Ti": ti,
                            "Li": li,
                            "Lbar_i": lbar,
                            "eval_path": combo.eval_path,
                        }
                    )

                    selected = 0
                    for pos, u_it in enumerate(vals):
                        if selected >= max(1, token_sample_cap_per_sample):
                            break
                        if stable_hash_uniform(seed, f"{combo.key}|{meta['sample_idx']}|{pos}|token") <= token_sample_rate:
                            token_rows_buf.append(
                                {
                                    "combo_key": combo.key,
                                    "train_dataset": combo.train_dataset,
                                    "method": combo.method,
                                    "epoch": combo.epoch,
                                    "task": combo.task,
                                    "sample_idx": int(meta["sample_idx"]),
                                    "token_pos": int(pos),
                                    "u_i_t": float(u_it),
                                    "Ti": ti,
                                    "Li": li,
                                    "Lbar_i": lbar,
                                    "sample_uid": str(meta["sample_uid"]),
                                    "domain_label": str(meta["domain_label"]),
                                }
                            )
                            selected += 1

                    if stable_hash_uniform(seed, f"{combo.key}|{meta['sample_idx']}|seq") <= seq_probe_sample_rate:
                        seq_take = min(int(seq_probe_max_tokens), ti)
                        for pos in range(seq_take):
                            seq_rows_buf.append(
                                {
                                    "combo_key": combo.key,
                                    "train_dataset": combo.train_dataset,
                                    "method": combo.method,
                                    "epoch": combo.epoch,
                                    "task": combo.task,
                                    "sample_idx": int(meta["sample_idx"]),
                                    "token_pos": int(pos),
                                    "u_i_t": float(vals[pos]),
                                    "Ti": ti,
                                    "sample_uid": str(meta["sample_uid"]),
                                    "domain_label": str(meta["domain_label"]),
                                }
                            )

                    processed += 1
                    total_tokens += ti

            del outputs, logits, labels, shift_logits, shift_labels, valid_mask, safe_labels, token_log_probs, token_nll, nll_cpu, mask_cpu, batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pending_ids = []
            pending_meta = []
        except RuntimeError as e:
            if _is_oom_error(e) and len(pending_ids) > 1:
                current_batch_size = max(1, len(pending_ids) // 2)
                oom_retries += 1
                half_ids = pending_ids[current_batch_size:]
                half_meta = pending_meta[current_batch_size:]
                pending_ids = pending_ids[:current_batch_size]
                pending_meta = pending_meta[:current_batch_size]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                run_pending_batch()
                pending_ids = half_ids
                pending_meta = half_meta
                run_pending_batch()
                return
            raise

    next_idx_for_state = start_idx
    try:
        for rec_idx, ex in enumerate(iter_records_stream(combo.eval_path)):
            if rec_idx < start_idx:
                continue
            if max_samples_per_task is not None and processed >= max_samples_per_task:
                done = True
                break

            prompt, answer = smart_parse_example(ex)
            text = build_text_for_task(combo.task, prompt, answer)
            if not text:
                next_idx_for_state = rec_idx + 1
                continue

            enc = tokenizer(text, truncation=(max_length > 0), max_length=max_length if max_length > 0 else None, padding=False)
            ids = enc.get("input_ids", [])
            if not isinstance(ids, list) or len(ids) < 2:
                next_idx_for_state = rec_idx + 1
                continue

            forced_truncation = 0
            if max_length <= 0 and model_limit and len(ids) > int(model_limit):
                ids = ids[: int(model_limit)]
                forced_truncation = 1
            if len(ids) < 2:
                next_idx_for_state = rec_idx + 1
                continue

            sample_uid = choose_meta_value(ex, ["id", "uid", "sample_id", "instance_id", "uuid"], default=f"{combo.task}_{rec_idx}")
            doc_id = choose_meta_value(ex, ["doc_id", "document_id", "id", "uid"], default="")
            source_id = choose_meta_value(ex, ["source_id", "source", "dataset", "origin"], default="")
            domain_label = choose_meta_value(ex, ["domain", "category", "topic"], default=combo.task)
            joined = f"{prompt}\n{answer}"

            pending_ids.append(ids)
            pending_meta.append(
                {
                    "sample_idx": rec_idx,
                    "sample_uid": sample_uid,
                    "doc_id": doc_id,
                    "source_id": source_id,
                    "domain_label": domain_label,
                    "input_chars": len(prompt),
                    "output_chars": len(answer),
                    "input_words": len(prompt.split()),
                    "output_words": len(answer.split()),
                    "contains_code": to_bool_flag(contains_code(joined)),
                    "contains_math": to_bool_flag(contains_math(joined)),
                    "forced_truncation": forced_truncation,
                }
            )
            next_idx_for_state = rec_idx + 1

            if len(pending_ids) >= max(1, current_batch_size):
                run_pending_batch()
                if processed > 0 and (processed % max(1, flush_every) == 0):
                    flush_rows(next_idx_for_state)
                    if verbose:
                        print(f"[FLUSH] combo={combo.key} processed={processed} next={next_idx_for_state}", flush=True)

        if pending_ids:
            run_pending_batch()
        done = True
    except Exception as e:
        flush_rows(next_idx_for_state)
        write_unavailable_note(
            unavailable_json,
            reason="forward collection exception",
            context={"combo_key": combo.key, "error": str(e), "trace": traceback.format_exc()[:4000]},
        )
        return {
            "combo_key": combo.key,
            "status": "failed",
            "error": str(e),
            "processed": processed,
            "next_sample_idx": next_idx_for_state,
            "state_json": state_json,
            "unavailable_json": unavailable_json,
        }

    flush_rows(next_idx_for_state)
    save_json(
        state_json,
        {
            "combo_key": combo.key,
            "next_sample_idx": int(next_idx_for_state),
            "done": bool(done),
            "processed_samples": int(processed),
            "processed_tokens": int(total_tokens),
            "oom_retries": int(oom_retries),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    sample_parquet = maybe_write_parquet(sample_csv)
    token_parquet = maybe_write_parquet(token_csv)
    seq_parquet = maybe_write_parquet(seq_csv)

    return {
        "combo_key": combo.key,
        "status": "ok",
        "processed_samples": int(processed),
        "processed_tokens": int(total_tokens),
        "oom_retries": int(oom_retries),
        "sample_csv": sample_csv,
        "token_csv": token_csv,
        "seq_csv": seq_csv,
        "sample_parquet": sample_parquet or "",
        "token_parquet": token_parquet or "",
        "seq_parquet": seq_parquet or "",
        "state_json": state_json,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Forward-only sample/token loss collector for theory diagnostics.")
    p.add_argument("--datainf_root", type=str, default=None)
    p.add_argument("--output_root", type=str, default="", help="Default: <result_root>/loss_theory")
    p.add_argument("--train_dataset", type=str, default="", help="comma-separated; empty means all")
    p.add_argument("--all_train_datasets", action="store_true")
    p.add_argument("--methods", type=str, default="sft,sdft", help="comma-separated")
    p.add_argument("--epochs", type=str, default="epoch_1,epoch_5", help="comma-separated")
    p.add_argument("--tasks", type=str, default="alpaca_eval,gsm8k,humaneval,multiarith,openfunction", help="comma-separated")
    p.add_argument("--base_model_path", type=str, default="")
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=0, help="<=0 means no explicit truncation")
    p.add_argument("--max_samples_per_task", type=int, default=0, help="<=0 means use all")
    p.add_argument("--device", type=str, default="", help="Default: cuda:0 if available else cpu")
    p.add_argument("--token_sample_rate", type=float, default=0.01)
    p.add_argument("--token_sample_cap_per_sample", type=int, default=64)
    p.add_argument("--seq_probe_sample_rate", type=float, default=0.002)
    p.add_argument("--seq_probe_max_tokens", type=int, default=512)
    p.add_argument("--flush_every", type=int, default=64)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefer_auto_on_fail", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    datainf_root = detect_datainf_root(args.datainf_root)
    sdft_root = resolve_sdft_root(datainf_root)
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    output_root = args.output_root.strip() or os.path.join(result_root, "loss_theory")
    ensure_dir(output_root)

    if args.all_train_datasets:
        train_datasets = list(DEFAULT_TRAIN_DATASETS)
    else:
        train_datasets = split_csv_arg(args.train_dataset, DEFAULT_TRAIN_DATASETS)
    methods = [x for x in split_csv_arg(args.methods, ["sft", "sdft"]) if x in ("sft", "sdft")]
    if not methods:
        raise ValueError(f"invalid --methods: {args.methods}")
    epochs = normalize_epoch_list(split_csv_arg(args.epochs, ["epoch_1", "epoch_5"]))
    epochs = [e for e in epochs if e in ("epoch_1", "epoch_5")]
    if not epochs:
        raise ValueError("this suite is configured for epoch_1/epoch_5 only")
    tasks = split_csv_arg(args.tasks, TASKS_5)
    for t in tasks:
        if t not in TASKS_5:
            raise ValueError(f"unknown task: {t}")

    shard_count = max(1, int(args.shard_count))
    shard_index = int(args.shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count-1}]")

    device = args.device.strip() or ("cuda:0" if torch.cuda.is_available() else "cpu")
    max_samples_per_task = args.max_samples_per_task if args.max_samples_per_task > 0 else None

    base_model_path = args.base_model_path.strip() or os.path.join(sdft_root, "model", "Llama-2-7b-chat-hf")
    data_root = args.data_root.strip() or os.path.join(sdft_root, "data")
    eval_paths = resolve_eval_dataset_paths(data_root)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    combos = build_combos(
        train_datasets=train_datasets,
        methods=methods,
        epochs=epochs,
        tasks=tasks,
        eval_paths=eval_paths,
    )
    shard_combos = select_shard(combos, shard_count=shard_count, shard_index=shard_index)

    print(
        f"[loss_theory_collect] total_combos={len(combos)} shard_count={shard_count} shard_index={shard_index} assigned={len(shard_combos)}",
        flush=True,
    )

    rows: List[Dict[str, object]] = []
    unavailable_rows: List[Dict[str, object]] = []
    for combo in shard_combos:
        if args.verbose:
            print(f"[RUN] {combo.key}", flush=True)
        r = run_one_combo(
            combo=combo,
            sdft_root=sdft_root,
            output_root=output_root,
            base_model_path=base_model_path,
            tokenizer=tokenizer,
            device=device,
            batch_size=max(1, int(args.batch_size)),
            max_length=int(args.max_length),
            max_samples_per_task=max_samples_per_task,
            token_sample_rate=max(0.0, min(1.0, float(args.token_sample_rate))),
            token_sample_cap_per_sample=max(1, int(args.token_sample_cap_per_sample)),
            seq_probe_sample_rate=max(0.0, min(1.0, float(args.seq_probe_sample_rate))),
            seq_probe_max_tokens=max(1, int(args.seq_probe_max_tokens)),
            flush_every=max(1, int(args.flush_every)),
            seed=int(args.seed),
            prefer_auto_on_fail=bool(args.prefer_auto_on_fail),
            verbose=bool(args.verbose),
        )
        rows.append(r)
        status = str(r.get("status", ""))
        if status.startswith("unavailable") or status in ("failed", "model_load_error"):
            unavailable_rows.append(dict(r))

    shard_tag = f"shard_{shard_index}_of_{shard_count}"
    run_dir = ensure_dir(os.path.join(output_root, "runs"))
    rows_csv = os.path.join(run_dir, f"collect_rows_{shard_tag}.csv")
    rows_json = os.path.join(run_dir, f"collect_rows_{shard_tag}.json")
    unavailable_json = os.path.join(run_dir, f"unavailable_collect_{shard_tag}.json")
    write_rows_csv(rows_csv, rows)
    save_json(rows_json, rows)
    if unavailable_rows:
        save_json(unavailable_json, unavailable_rows)
        print(os.path.abspath(unavailable_json))
    print(os.path.abspath(rows_csv))
    print(os.path.abspath(rows_json))


if __name__ == "__main__":
    main()
