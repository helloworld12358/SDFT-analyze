#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_pairwise_from_grads_tagged.py (modified to save results to script_dir/result/...)

保留原有行为（从已保存的 .pt 加载平均 LoRA 梯度向量并计算 pairwise 内积 / 余弦矩阵），
并将计算结果保存到脚本目录下的 result/<model>/<epoch>/<method>/ 下。
当使用 --recompute_from_data 重算缺失的 grad 时，重算得到的 .pt 会保存到
script_dir/result/output_grad/<epoch>/<method>/<model>/<dataset>.pt（优先写入该位置）。
"""
from __future__ import annotations

import os
import argparse
import json
from glob import glob
import torch
import numpy as np
from typing import List, Optional, Dict, Tuple

# --- helper math / i/o functions ---


def _make_deterministic_eigvecs(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v
    v = v.copy()
    n, m = v.shape
    for k in range(m):
        vk = v[:, k]
        norm = np.linalg.norm(vk)
        if not np.isfinite(norm) or norm == 0:
            continue
        vk = vk / norm
        idx = int(np.argmax(np.abs(vk)))
        val = vk[idx]
        if abs(val) < 1e-16:
            v[:, k] = vk
            continue
        phase = val / abs(val)
        vk = vk / phase
        if np.isrealobj(vk) and vk[idx] < 0:
            vk = -vk
        v[:, k] = vk
    return v


def complex_to_str(z, prec=8):
    if abs(getattr(z, "imag", 0.0)) < 1e-12:
        return f"{float(getattr(z, 'real', z)):.{prec}g}"
    else:
        return f"({z.real:.{prec}g}{z.imag:+.{prec}g}j)"


def vector_to_str(vec, prec=8):
    return "[" + ", ".join(complex_to_str(x, prec) for x in vec.reshape(-1)) + "]"


def load_grad_vector(pt_path: str) -> np.ndarray:
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(pt_path)
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().view(-1).numpy()
    if isinstance(obj, (list, tuple, np.ndarray)):
        return np.asarray(obj).reshape(-1)
    if isinstance(obj, dict):
        parts = []
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                parts.append(v.detach().cpu().view(-1).numpy())
        if parts:
            return np.concatenate(parts, axis=0)
    raise ValueError(f"Unsupported grad file format: {pt_path}")


# --- 内置模板与解析（用于重算 avg grad 时的格式化） ---


def alpaca_format(user_content: str, assistant_response: str) -> str:
    return f"### Instruction:\n{user_content}\n\n### Response:\n{assistant_response}"


def gsm8k_format(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def smart_parse_example(example: Dict) -> Tuple[str, str]:
    keys = set(example.keys())
    if "instruction" in keys and ("output" in keys or "response" in keys):
        instr = example.get("instruction", "")
        extra_input = example.get("input", "")
        if extra_input:
            instr = instr + "\n" + extra_input
        resp = example.get("output", example.get("response", ""))
        return instr, resp
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


def choose_template_by_dataset_name(name: str) -> str:
    n = name.lower()
    if "gsm8k" in n or "multiarith" in n or "multiarith" in n:
        return "gsm8k"
    return "alpaca"


# --- 可选：当需要重算 avg grad 时，使用以下函数 ---


def find_dataset_file(data_root: str, dataset_name: str) -> Optional[str]:
    candidates = []
    base = os.path.join(data_root, "datasets")
    candidates.append(os.path.join(base, f"{dataset_name}.json"))
    candidates.append(os.path.join(base, f"{dataset_name}.jsonl"))
    candidates.append(os.path.join(base, f"{dataset_name}.ndjson"))
    candidates.append(os.path.join(base, dataset_name, f"{dataset_name}.json"))
    candidates.append(os.path.join(base, dataset_name, f"{dataset_name}.jsonl"))
    candidates.extend(glob(os.path.join(base, f"{dataset_name}*.json")))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    for root, _, files in os.walk(data_root):
        for fn in files:
            low = fn.lower()
            if dataset_name.lower() in low and low.endswith(".json"):
                return os.path.join(root, fn)
    return None


def compute_and_save_avg_grad_for_dataset(
    dataset_json_path: str,
    model_name_or_path: str,
    out_pt_path: str,
    lora_path: Optional[str],
    batch_size: int = 16,
    max_samples: Optional[int] = None,
    max_length: int = 1024,
):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel, LoraConfig, get_peft_model, TaskType
        from torch.utils.data import DataLoader, Dataset
    except Exception as e:
        raise RuntimeError(f"Missing required packages for recompute: {e}")

    examples = []
    with open(dataset_json_path, "r", encoding="utf-8") as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "[":
            j = json.load(fh)
            if isinstance(j, list):
                examples = j
            else:
                raise RuntimeError(f"Unsupported JSON format in {dataset_json_path}")
        else:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                examples.append(json.loads(line))

    if max_samples:
        examples = examples[:max_samples]

    device_available = torch.cuda.is_available()
    device = "cuda" if device_available else "cpu"
    torch_dtype = (
        torch.bfloat16 if (device_available and torch.cuda.is_bf16_supported()) else
        (torch.float16 if device_available else torch.float32)
    )

    device_map = None
    if device_available:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if vis != "":
            device_map = {"": "cuda:0"}
        else:
            device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if lora_path and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, peft_config)

    model.train()

    texts = []
    for ex in examples:
        prompt, response = smart_parse_example(ex)
        template_choice = choose_template_by_dataset_name(os.path.basename(dataset_json_path))
        if template_choice == "gsm8k":
            text = gsm8k_format(prompt, response)
        else:
            text = alpaca_format(prompt, response)
        texts.append(text)

    tok = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
    labels = [list(ids) for ids in tok["input_ids"]]

    class SimpleListDataset(Dataset):
        def __init__(self, input_ids_list, labels_list):
            self.input_ids_list = input_ids_list
            self.labels_list = labels_list

        def __len__(self):
            return len(self.input_ids_list)

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids_list[idx], "labels": self.labels_list[idx]}

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate_fn_local(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        attention_mask = []
        labels_batch = []
        for x in batch:
            ids = list(x["input_ids"])
            lab = list(x["labels"])
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = ids + [pad_id] * pad_len
                lab = lab + [-100] * pad_len
                att = [1] * (len(ids) - pad_len) + [0] * pad_len
            else:
                att = [1] * len(ids)
            input_ids.append(ids)
            attention_mask.append(att)
            labels_batch.append(lab)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }

    from torch.utils.data import DataLoader

    ds = SimpleListDataset(tok["input_ids"], labels)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_local)

    def get_lora_grad_vector_local(model_local: torch.nn.Module) -> Optional[torch.Tensor]:
        grad_list = []
        for name, param in model_local.named_parameters():
            if "lora" in name and param.requires_grad:
                if param.grad is not None:
                    grad_list.append(param.grad.detach().view(-1).cpu())
                else:
                    grad_list.append(torch.zeros_like(param).view(-1).cpu())
        if not grad_list:
            return None
        return torch.cat(grad_list)

    sum_grad_vector = None
    total_samples = 0

    for batch in dl:
        batch = {k: v.to("cuda" if device == "cuda" else "cpu") for k, v in batch.items()}
        current_batch_size = batch["input_ids"].shape[0]
        model.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        grad_vec = get_lora_grad_vector_local(model)
        if grad_vec is not None:
            weighted = grad_vec * current_batch_size
            if sum_grad_vector is None:
                sum_grad_vector = torch.zeros_like(grad_vec)
            sum_grad_vector += weighted
            total_samples += current_batch_size
        del batch, outputs, loss, grad_vec

    if sum_grad_vector is None or total_samples == 0:
        raise RuntimeError(f"[recompute] No gradients computed. total_samples={total_samples}")

    avg_grad_vector = sum_grad_vector / total_samples
    os.makedirs(os.path.dirname(os.path.abspath(out_pt_path)), exist_ok=True)
    torch.save(avg_grad_vector, out_pt_path)
    return out_pt_path


# --- main (整合原脚本的 pairwise 计算 + optional recompute) ---


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datainf_root", type=str, default=None,
                   help="DataInf 根目录（默认：脚本上级目录）")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--epoch", type=str, required=True, choices=["epoch_0", "epoch_1", "epoch_5"])
    p.add_argument("--method", type=str, required=True, choices=["sdft", "sft"])
    p.add_argument("--dataset_names", type=str, default=None,
                   help="逗号分隔的数据集名称列表（默认使用 alpaca_eval,gsm8k,humaneval,multiarith,openfunction）")
    p.add_argument("--normalize", action="store_true", help="是否使用余弦相似度（默认关闭）")
    p.add_argument("--dtype64", action="store_true", help="使用 float64 计算（默认 False）")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--recompute_from_data", action="store_true",
                   help="若 grad 文件缺失，尝试重新计算并保存 grad 文件到 script_dir/result/output_grad/...（需提供 --base_model_path）")
    p.add_argument("--base_model_path", type=str, default=None,
                   help="重算 grad 时所需的 base model 路径或 HF id（required if --recompute_from_data）")
    p.add_argument("--lora_path", type=str, default=None, help="重算 grad 时可选的 LoRA adapter path")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_length", type=int, default=1024)
    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.datainf_root:
        DATAINF_ROOT = os.path.abspath(args.datainf_root)
    else:
        DATAINF_ROOT = os.path.normpath(os.path.join(script_dir, ".."))

    if args.dataset_names:
        DATASET_NAMES = [s.strip() for s in args.dataset_names.split(",") if s.strip()]
    else:
        DATASET_NAMES = ["alpaca_eval", "gsm8k", "humaneval", "multiarith", "openfunction"]

    MODEL = args.model
    EPOCH = args.epoch
    METHOD = args.method

    # 原始 grad 文件查找目录（保留为备选）
    GRADS_BASE_DIR = os.path.join(DATAINF_ROOT, "output_grad", EPOCH, METHOD, MODEL)
    if args.verbose:
        print("DATAINF_ROOT:", DATAINF_ROOT)
        print("GRADS_BASE_DIR:", GRADS_BASE_DIR)

    # 新的本地 result grad 根（优先查找/写入）
    LOCAL_GRADS_ROOT = os.path.join(script_dir, "result", "output_grad", EPOCH, METHOD, MODEL)

    # verify or recompute grad files
    grad_paths: List[str] = []
    missing = []
    for name in DATASET_NAMES:
        # 首先尝试本地 result 下的路径（优先）
        local_pth = os.path.join(LOCAL_GRADS_ROOT, f"{name}.pt")
        if os.path.isfile(local_pth):
            grad_paths.append(local_pth)
            continue
        # 否则尝试原始 DATAINF_ROOT 下的路径（备用）
        alt_pth = os.path.join(GRADS_BASE_DIR, f"{name}.pt")
        if os.path.isfile(alt_pth):
            grad_paths.append(alt_pth)
            continue
        # 都没有则记录缺失，预期保存位置为 local_pth（若重算将写入 local_pth）
        missing.append((name, local_pth))
        grad_paths.append(local_pth)  # 占位，后续会确保文件存在或重算

    # if missing but user allowed recompute, attempt to find dataset files and recompute
    if missing and args.recompute_from_data:
        if not args.base_model_path:
            raise ValueError("--base_model_path required when --recompute_from_data is set")
        for name, expected_local_pth in missing:
            ds_file = find_dataset_file(DATAINF_ROOT, name)
            if ds_file is None:
                alt = find_dataset_file(os.path.join(DATAINF_ROOT, ".."), name)
                if alt:
                    ds_file = alt
            if ds_file is None:
                raise FileNotFoundError(f"[recompute] Cannot find dataset file for '{name}' under {DATAINF_ROOT}")
            if args.verbose:
                print(f"[recompute] Found dataset for {name}: {ds_file} -> will compute avg grad to {expected_local_pth}")
            computed = compute_and_save_avg_grad_for_dataset(
                dataset_json_path=ds_file,
                model_name_or_path=args.base_model_path,
                out_pt_path=expected_local_pth,
                lora_path=args.lora_path,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                max_length=args.max_length,
            )
            if args.verbose:
                print(f"[recompute] Saved computed grad for {name} to {computed}")

    # 检查仍然缺失的文件
    still_missing = [p for p in grad_paths if not os.path.isfile(p)]
    if still_missing:
        raise FileNotFoundError(f"Missing grad vector files: {still_missing}")

    # load vectors
    vecs = [load_grad_vector(p) for p in grad_paths]
    dims = [v.size for v in vecs]
    if len(set(dims)) != 1:
        raise ValueError(f"Inconsistent grad vector dims: {dims}")
    D = dims[0]
    n = len(vecs)

    dtype = np.float64 if args.dtype64 else np.float32
    V = np.stack([v.astype(dtype) for v in vecs], axis=1)  # shape (D, n)

    if args.normalize:
        norms = np.linalg.norm(V, axis=0)
        norms[norms == 0] = 1.0
        Vn = V / norms[None, :]
        M = (Vn.T @ Vn).astype(dtype)
    else:
        M = (V.T @ V).astype(dtype)

    # 结果保存到脚本目录下的 result/<model>/<epoch>/<method>/
    RESULTS_DIR = os.path.join(script_dir, "result", MODEL, EPOCH, METHOD)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    base_tag = f"pairwise_matrix_{MODEL}_{EPOCH}_{METHOD}"
    mat_npy = os.path.join(RESULTS_DIR, base_tag + ".npy")
    mat_csv = os.path.join(RESULTS_DIR, base_tag + ".csv")
    mat_txt = os.path.join(RESULTS_DIR, base_tag + ".txt")
    eigvals_npy = os.path.join(RESULTS_DIR, f"eigenvalues_{MODEL}_{EPOCH}_{METHOD}.npy")
    eigvecs_npy = os.path.join(RESULTS_DIR, f"eigenvectors_{MODEL}_{EPOCH}_{METHOD}.npy")

    np.save(mat_npy, M)
    np.savetxt(mat_csv, M, delimiter=",", fmt="%.18e")

    # eigen decomposition
    try:
        w, v = np.linalg.eigh(M)
        idx = np.argsort(-w.real)
        w = w[idx]
        v = v[:, idx]
    except Exception:
        w, v = np.linalg.eig(M)
        idx = np.argsort(-w.real)
        w = w[idx]
        v = v[:, idx]

    v = _make_deterministic_eigvecs(v)

    np.save(eigvals_npy, w)
    np.save(eigvecs_npy, v)

    # write readable txt (matrix + eigenvalues + eigenvectors)
    with open(mat_txt, "w", encoding="utf-8") as f:
        f.write(f"Model: {MODEL}\nEpoch: {EPOCH}\nMethod: {METHOD}\nDatasets: {DATASET_NAMES}\n\n")
        f.write("Pairwise matrix (rows/cols order = datasets order above):\n")
        with np.printoptions(precision=8, suppress=True):
            for row in M:
                f.write("  " + ", ".join(f"{float(x):.8e}" for x in row) + "\n")
        f.write("\nEigenvalues (descending by real part):\n")
        for val in w:
            if abs(getattr(val, "imag", 0.0)) < 1e-12:
                f.write(f"  {float(getattr(val,'real',val)):.18e}\n")
            else:
                f.write(f"  {val}\n")
        f.write("\nEigenvectors (each vector listed as column; same ordering as eigenvalues above):\n")
        for k in range(v.shape[1]):
            vec = v[:, k]
            f.write(f"eig[{k+1}] = {complex_to_str(w[k], prec=12)}\n")
            f.write("  vec: [" + ", ".join(complex_to_str(x, prec=12) for x in vec.reshape(-1)) + "]\n\n")

    meta = {
        "model": MODEL,
        "epoch": EPOCH,
        "method": METHOD,
        "datasets": DATASET_NAMES,
        "matrix_npy": os.path.abspath(mat_npy),
        "matrix_csv": os.path.abspath(mat_csv),
        "matrix_txt": os.path.abspath(mat_txt),
        "eigvals_npy": os.path.abspath(eigvals_npy),
        "eigvecs_npy": os.path.abspath(eigvecs_npy),
        "normalize": bool(args.normalize),
        "dtype": str(dtype)
    }
    with open(os.path.join(RESULTS_DIR, f"summary_{MODEL}_{EPOCH}_{METHOD}.json"), "w", encoding="utf-8") as jf:
        json.dump(meta, jf, ensure_ascii=False, indent=2)

    if args.verbose:
        print("Saved matrix npy:", mat_npy)
        print("Saved matrix csv:", mat_csv)
        print("Saved summary json:", os.path.join(RESULTS_DIR, f"summary_{MODEL}_{EPOCH}_{METHOD}.json"))
    print(os.path.abspath(mat_txt))


if __name__ == "__main__":
    main()

