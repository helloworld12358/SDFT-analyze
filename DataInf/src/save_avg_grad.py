# Modified save_avg_grad.py
# 保持原功能（计算并保存 dataset 的平均 LoRA 梯度），但在模型加载时
# 更明确地将模型绑定到单个可见 GPU（如果 CUDA_VISIBLE_DEVICES 被设置）
# 以避免多进程启动时 device_map="auto" 导致的跨进程冲突/分配问题。

import argparse
import torch
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="计算并保存数据集的平均 LoRA 梯度 (支持多卡并行+大Batch)")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    # LoRA Init Params
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target", type=str, default="q_proj,v_proj")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()

def get_lora_grad_vector(model):
    """提取 LoRA 参数的梯度并展平为一个向量"""
    grad_list = []
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad:
            if param.grad is not None:
                grad_list.append(param.grad.view(-1).cpu())
            else:
                grad_list.append(torch.zeros_like(param).view(-1).cpu())
    if not grad_list:
        return None
    return torch.cat(grad_list)

def smart_parse_example(example):
    """解析多种数据格式"""
    keys = example.keys()
    if "question" in keys and "answer" in keys:
        return example["question"], example["answer"]
    if "goal" in keys and "target" in keys:
        return example["goal"], example["target"]
    if "prompt" in keys and ("canonical_solution" in keys or "buggy_solution" in keys):
        instruction = example.get("instruction", "")
        full_prompt = f"{instruction}\n{example['prompt']}" if instruction else example['prompt']
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

def main():
    args = parse_args()

    # device selection (use cuda if available)
    device_available = torch.cuda.is_available()
    device = "cuda" if device_available else "cpu"

    # choose dtype: prefer bf16 when supported, otherwise use float16 for GPU
    torch_dtype = torch.bfloat16 if (device_available and torch.cuda.is_bf16_supported()) else (torch.float16 if device_available else torch.float32)

    print(f"Loading Model: {args.base_model_path} (dtype={torch_dtype}, device={device})")

    # When this process has CUDA_VISIBLE_DEVICES set to a single visible device,
    # binding model to "cuda:0" keeps each process isolated on its own physical GPU.
    # Inside the process, cuda:0 corresponds to the first visible GPU.
    device_map = None
    if device_available:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        # If CUDA_VISIBLE_DEVICES is set (even if it's a list), prefer mapping whole model to cuda:0
        if vis != "":
            device_map = {"": "cuda:0"}
        else:
            # no visibility override; let transformers decide automatically
            device_map = "auto"

    # Load base model
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"[Error] Failed to load base model: {e}")
        raise

    base_model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
        inputs = []
        # HuggingFace dataset examples is dict of lists
        for i in range(len(examples[list(examples.keys())[0]])):
            ex = {k: examples[k][i] for k in examples}
            inp, out = smart_parse_example(ex)
            if inp and out:
                inputs.append(f"{inp}\n{out}")
            else:
                inputs.append("")  # placeholder, will be filtered
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding=False)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    processed_dataset = processed_dataset.filter(lambda x: len(x["input_ids"]) > 0)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    dataloader = DataLoader(
        processed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    # --- 3. 批量计算梯度 ---
    print(f"Computing Gradients (Batch Size: {args.batch_size})...")
    sum_grad_vector = None
    total_samples = 0

    for batch in tqdm(dataloader):
        # move tensors to device (if model lives on cuda, this will map to cuda:0 inside process)
        batch = {k: v.to("cuda" if device == "cuda" else "cpu") for k, v in batch.items()}
        current_batch_size = batch["input_ids"].shape[0]

        # Forward -> note: ensure grads are cleared
        model.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss

        # Backward
        loss.backward()

        # 提取梯度
        grad_vec = get_lora_grad_vector(model)

        if grad_vec is not None:
            weighted_grad = grad_vec * current_batch_size
            if sum_grad_vector is None:
                sum_grad_vector = torch.zeros_like(grad_vec)
            sum_grad_vector += weighted_grad
            total_samples += current_batch_size

        # free
        del batch, outputs, loss, grad_vec

    if sum_grad_vector is None or total_samples == 0:
        print(f"[Error] No gradients computed. total_samples={total_samples}")
        return

    avg_grad_vector = sum_grad_vector / total_samples

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(avg_grad_vector, args.output_path)
    print(f"Saved to: {args.output_path} (Total Samples: {total_samples})")

if __name__ == "__main__":
    main()
