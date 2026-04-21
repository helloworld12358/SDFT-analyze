#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import random
import sys
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAINF_ROOT_DEFAULT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(DATAINF_ROOT_DEFAULT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gram_scheme_a_utils import (  # type: ignore  # noqa: E402
    DEFAULT_TASKS,
    DEFAULT_TRAIN_DATASETS,
    detect_datainf_root,
    ensure_dir,
    resolve_checkpoint_path,
    resolve_result_root,
    resolve_sdft_root,
)

TASKS_5 = list(DEFAULT_TASKS)
METHODS_EVAL = ["sft", "sdft"]


def split_csv_arg(s: str, default: Sequence[str]) -> List[str]:
    if not s.strip():
        return list(default)
    out = [x.strip() for x in s.split(",") if x.strip()]
    return out if out else list(default)


def resolve_output_root(datainf_root: str, output_root: str) -> str:
    if output_root.strip():
        return os.path.abspath(output_root.strip())
    result_root = resolve_result_root(datainf_root, prefer_plural=True)
    return os.path.join(result_root, "embedding_cluster")


def choose_dtype(device: str) -> torch.dtype:
    dev = (device or "").strip().lower()
    if dev == "auto":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    if dev.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def should_clear_cuda_cache(device: str) -> bool:
    dev = (device or "").strip().lower()
    return dev.startswith("cuda") or dev == "auto"


def resolve_model_input_device(model: torch.nn.Module, preferred_device: str) -> torch.device:
    dev = (preferred_device or "").strip().lower()
    if dev and dev != "auto":
        return torch.device(dev)

    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict):
        vals = []
        for v in hf_map.values():
            if isinstance(v, str):
                vals.append(v)
            elif isinstance(v, int):
                vals.append(f"cuda:{v}")
        for v in vals:
            if v and v != "cpu":
                return torch.device(v)
        if vals:
            return torch.device(vals[0])

    try:
        return next(model.parameters()).device
    except Exception:
        pass
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def configure_torch_runtime_for_speed(device: str) -> None:
    dev = (device or "").strip().lower()
    use_cuda = dev.startswith("cuda") or (dev == "auto" and torch.cuda.is_available())
    if not use_cuda:
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


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


def build_prompt_only_text(task: str, prompt: str) -> str:
    p = (prompt or "").strip()
    if task in ("gsm8k", "multiarith"):
        return f"Question: {p}\nAnswer:"
    return f"### Instruction:\n{p}\n\n### Response:\n"


def load_records(path: str) -> List[Dict[str, object]]:
    if path.endswith(".jsonl") or path.endswith(".ndjson"):
        out: List[Dict[str, object]] = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
        return out

    ds = load_dataset("json", data_files={"eval": path}, split="eval")
    out = []
    for x in ds:
        if isinstance(x, dict):
            out.append(dict(x))
    return out


def sample_prompt_pool(
    eval_paths: Dict[str, str],
    tasks: Sequence[str],
    per_task: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    stats: List[Dict[str, object]] = []

    for task in tasks:
        path = eval_paths.get(task, "")
        task_rows: List[Dict[str, object]] = []
        if path and os.path.isfile(path):
            records = load_records(path)
            for i, rec in enumerate(records):
                prompt, _ = smart_parse_example(rec)
                text = build_prompt_only_text(task, prompt)
                if not text.strip():
                    continue
                uid = str(rec.get("id", rec.get("uid", rec.get("sample_id", f"{task}_{i}"))))
                task_rows.append(
                    {
                        "task": task,
                        "label": task,
                        "uid": uid,
                        "sample_idx_src": i,
                        "text": text,
                    }
                )

        before = len(task_rows)
        if before > per_task:
            idx = list(range(before))
            rng.shuffle(idx)
            keep = set(idx[:per_task])
            task_rows = [task_rows[j] for j in range(before) if j in keep]
        rows.extend(task_rows)
        stats.append(
            {
                "task": task,
                "dataset_path": path,
                "available": before,
                "selected": len(task_rows),
            }
        )

    # deterministic order for reproducibility
    rows = sorted(rows, key=lambda x: (str(x["task"]), int(x["sample_idx_src"]), str(x["uid"])))
    return rows, stats


def load_model_with_optional_lora(
    base_model_path: str,
    lora_path: Optional[str],
    device: str,
    prefer_auto_on_fail: bool = True,
) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
    configure_torch_runtime_for_speed(device)
    dtype = choose_dtype(device)
    dev = (device or "").strip().lower()
    if dev == "auto":
        device_map_single = "auto"
    elif dev.startswith("cuda"):
        device_map_single = {"": dev}
    else:
        device_map_single = None

    try:
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
        if dev and dev != "auto":
            try:
                model.to(dev)
            except Exception:
                pass
        model.eval()
        return model, None
    except Exception:
        if not prefer_auto_on_fail:
            return None, traceback.format_exc()

    try:
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


def infer_layer_count(model: torch.nn.Module, tokenizer: AutoTokenizer, device: str) -> int:
    in_dev = resolve_model_input_device(model, device)
    enc = tokenizer("hello", return_tensors="pt")
    enc = {k: v.to(in_dev) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise RuntimeError("model output does not include hidden_states")
    if len(hs) < 2:
        raise RuntimeError("unexpected hidden_states length")
    return int(len(hs) - 1)  # ignore embedding slot at index 0


def parse_layers_spec(layers_spec: str, n_layers: int) -> List[int]:
    s = (layers_spec or "").strip().lower()
    if s in ("", "all", "full"):
        return list(range(1, n_layers + 1))
    vals: List[int] = []
    for tok in layers_spec.split(","):
        t = tok.strip()
        if not t:
            continue
        if "-" in t:
            a, b = t.split("-", 1)
            ia, ib = int(a), int(b)
            lo, hi = min(ia, ib), max(ia, ib)
            for k in range(lo, hi + 1):
                if 1 <= k <= n_layers:
                    vals.append(k)
        else:
            k = int(t)
            if 1 <= k <= n_layers:
                vals.append(k)
    vals = sorted(set(vals))
    if not vals:
        vals = list(range(1, n_layers + 1))
    return vals


def _is_cuda_oom_error(err: BaseException) -> bool:
    s = str(err).lower()
    return ("out of memory" in s) or ("cuda oom" in s) or ("cublas_status_alloc_failed" in s)


def _pick_probe_text(tokenizer: AutoTokenizer, texts: Sequence[str], max_length: int) -> str:
    if not texts:
        return "Hello world"
    cand = list(texts[: min(64, len(texts))])
    best = cand[0]
    best_len = -1
    for t in cand:
        try:
            ids = tokenizer(
                t,
                truncation=(max_length > 0),
                max_length=max_length if max_length > 0 else None,
                add_special_tokens=True,
            )["input_ids"]
            l = len(ids)
        except Exception:
            l = len(t)
        if l > best_len:
            best_len = l
            best = t
    return best


def auto_probe_batch_size(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    device: str,
    max_length: int,
    start_batch: int = 8,
    max_probe_batch: int = 256,
) -> int:
    in_dev = resolve_model_input_device(model, device)
    if in_dev.type != "cuda":
        return max(1, start_batch)

    probe_text = _pick_probe_text(tokenizer, texts, max_length)

    def _try(bs: int) -> bool:
        batch = [probe_text] * int(bs)
        try:
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=(max_length > 0),
                max_length=max_length if max_length > 0 else None,
            )
            input_ids = enc["input_ids"].to(in_dev)
            attn = enc["attention_mask"].to(in_dev)
            with torch.inference_mode():
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    output_hidden_states=False,
                    use_cache=False,
                )
            del input_ids, attn, enc, batch
            torch.cuda.synchronize(device=in_dev)
            return True
        except RuntimeError as e:
            if _is_cuda_oom_error(e):
                torch.cuda.empty_cache()
                return False
            raise

    low = 1
    high_fail = 0
    cur = max(1, int(start_batch))
    cur = min(cur, int(max_probe_batch))

    if _try(cur):
        low = cur
        while low < max_probe_batch:
            nxt = min(max_probe_batch, low * 2)
            if nxt == low:
                break
            if _try(nxt):
                low = nxt
            else:
                high_fail = nxt
                break
    else:
        high_fail = cur

    if high_fail == 0:
        return low

    l, r = low, high_fail - 1
    best = max(1, l)
    while l <= r:
        mid = (l + r) // 2
        if _try(mid):
            best = mid
            l = mid + 1
        else:
            r = mid - 1
    return max(1, best)


def extract_last_token_representations(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    layers: Sequence[int],
    batch_size: int,
    device: str,
    max_length: int,
    auto_tune_batch: bool = True,
    max_probe_batch: int = 256,
) -> Dict[int, np.ndarray]:
    in_dev = resolve_model_input_device(model, device)
    eff_batch = max(1, int(batch_size))
    if auto_tune_batch or int(batch_size) <= 0:
        start_bs = max(1, int(batch_size)) if int(batch_size) > 0 else 8
        try:
            eff_batch = auto_probe_batch_size(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                device=device,
                max_length=max_length,
                start_batch=start_bs,
                max_probe_batch=max_probe_batch,
            )
        except Exception:
            eff_batch = start_bs

    out_by_layer: Dict[int, List[np.ndarray]] = {int(l): [] for l in layers}
    st = 0
    while st < len(texts):
        cur_bs = min(eff_batch, len(texts) - st)
        try:
            batch = list(texts[st : st + cur_bs])
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=(max_length > 0),
                max_length=max_length if max_length > 0 else None,
            )
            input_ids = enc["input_ids"].to(in_dev)
            attn = enc["attention_mask"].to(in_dev)
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hs = outputs.hidden_states  # tuple: [emb, layer1, ... layerN]
            last_idx = torch.clamp(attn.sum(dim=1) - 1, min=0).long()
            for l in layers:
                h = hs[int(l)]  # [B,T,H]
                bsz = h.shape[0]
                picked = h[torch.arange(bsz, device=h.device), last_idx, :]
                out_by_layer[int(l)].append(picked.detach().to(dtype=torch.float32, device="cpu").numpy())

            del outputs, hs, input_ids, attn, enc, batch
            st += cur_bs
        except RuntimeError as e:
            if _is_cuda_oom_error(e) and cur_bs > 1:
                eff_batch = max(1, cur_bs // 2)
                if should_clear_cuda_cache(device):
                    torch.cuda.empty_cache()
                continue
            raise

    return {k: np.concatenate(v, axis=0) if v else np.zeros((0, 0), dtype=np.float32) for k, v in out_by_layer.items()}


def knn_purity(x: np.ndarray, y_int: np.ndarray, k: int = 10) -> float:
    if len(x) <= 1:
        return float("nan")
    k_eff = max(1, min(k, len(x) - 1))
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nbrs.fit(x)
    idx = nbrs.kneighbors(x, return_distance=False)[:, 1:]  # remove self
    neigh_labels = y_int[idx]
    matches = (neigh_labels == y_int[:, None]).astype(np.float32)
    return float(matches.mean())


def compute_cluster_metrics(x: np.ndarray, y_int: np.ndarray) -> Dict[str, float]:
    if x.ndim != 2 or len(x) < 3:
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
            "knn_purity": float("nan"),
        }
    unique = np.unique(y_int)
    if len(unique) < 2:
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
            "knn_purity": float("nan"),
        }
    out = {
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
        "calinski_harabasz": float("nan"),
        "knn_purity": float("nan"),
    }
    try:
        out["silhouette"] = float(silhouette_score(x, y_int, metric="euclidean"))
    except Exception:
        pass
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(x, y_int))
    except Exception:
        pass
    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(x, y_int))
    except Exception:
        pass
    try:
        out["knn_purity"] = float(knn_purity(x, y_int, k=10))
    except Exception:
        pass
    return out


def build_label_arrays(rows: Sequence[Dict[str, object]], tasks: Sequence[str]) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    task_to_idx = {t: i for i, t in enumerate(tasks)}
    labels: List[int] = []
    names: List[str] = []
    for r in rows:
        t = str(r.get("task", ""))
        if t not in task_to_idx:
            raise ValueError(f"unknown task label in sample pool: {t}")
        labels.append(task_to_idx[t])
        names.append(t)
    return np.asarray(labels, dtype=np.int64), names, task_to_idx


def build_model_specs_epoch5(train_dataset: str, include_base: bool) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    for method in METHODS_EVAL:
        specs.append(
            {
                "model_tag": f"{train_dataset}__{method}__epoch_5",
                "train_dataset": train_dataset,
                "method": method,
                "epoch": "epoch_5",
                "use_lora": True,
            }
        )
    if include_base:
        specs.append(
            {
                "model_tag": "base__epoch_0",
                "train_dataset": "base",
                "method": "base",
                "epoch": "epoch_0",
                "use_lora": False,
            }
        )
    return specs


def build_model_specs_story(train_dataset: str, include_base: bool) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    for method in METHODS_EVAL:
        for epoch in ("epoch_1", "epoch_5"):
            specs.append(
                {
                    "model_tag": f"{train_dataset}__{method}__{epoch}",
                    "train_dataset": train_dataset,
                    "method": method,
                    "epoch": epoch,
                    "use_lora": True,
                }
            )
    if include_base:
        specs.append(
            {
                "model_tag": "base__epoch_0",
                "train_dataset": "base",
                "method": "base",
                "epoch": "epoch_0",
                "use_lora": False,
            }
        )
    return specs


def resolve_model_paths_for_spec(sdft_root: str, base_model_path: str, spec: Dict[str, object]) -> Tuple[str, Optional[str], Optional[str]]:
    use_lora = bool(spec.get("use_lora", False))
    if not use_lora:
        return base_model_path, None, None
    train_dataset = str(spec.get("train_dataset", ""))
    method = str(spec.get("method", ""))
    epoch = str(spec.get("epoch", ""))
    lora_path = resolve_checkpoint_path(sdft_root, epoch, train_dataset, method)
    if not lora_path or not os.path.isdir(lora_path):
        return base_model_path, None, f"checkpoint missing for {train_dataset}/{method}/{epoch}"
    return base_model_path, lora_path, None


def init_runtime_paths(datainf_root_arg: Optional[str], output_root_arg: str, base_model_path_arg: str, data_root_arg: str) -> Dict[str, str]:
    datainf_root = detect_datainf_root(datainf_root_arg)
    sdft_root = resolve_sdft_root(datainf_root)
    output_root = resolve_output_root(datainf_root, output_root_arg)
    ensure_dir(output_root)
    base_model_path = base_model_path_arg.strip() or os.path.join(sdft_root, "model", "Llama-2-7b-chat-hf")
    data_root = data_root_arg.strip() or os.path.join(sdft_root, "data")
    return {
        "datainf_root": datainf_root,
        "sdft_root": sdft_root,
        "output_root": output_root,
        "base_model_path": base_model_path,
        "data_root": data_root,
    }
