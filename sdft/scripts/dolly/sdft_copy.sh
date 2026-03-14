#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "${SCRIPT_DIR}/../model" ] && [ -d "${SCRIPT_DIR}/../data" ]; then
  SDFT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  SDFT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
set -e
source "scripts/utils.sh"

# Configurations
model_path="${SDFT_ROOT}/model/Llama-2-7b-chat-hf"
cuda_visible_devices="0,1,2,3"
type=sdft
train_dataset=dolly
output_folder="predictions/${train_dataset}/${type}"
result_file="results/${train_dataset}/${type}.log"
checkpoint_dir="checkpoints/${train_dataset}/${type}"
tmp_predictions_dir="predictions/${train_dataset}/${type}/distilled"

# Hyperparameters
epoch=2
lr=1e-4
per_device_train_batch_size=2

create_empty_file ${result_file}
echo -e "Fine-tuning using ${type}\n" >> ${result_file}

# Export visible GPUs and set NCCL options for distributed training
export CUDA_VISIBLE_DEVICES=${cuda_visible_devices}
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL

# Generate distilled dataset - Distributed prediction
torchrun \
  --standalone \
  --nproc_per_node=4 \
  --master_port=12345 \
  main.py \
    --stage sft \
    --model_name_or_path ${model_path} \
    --do_predict \
    --dataset ${train_dataset}_train \
    --template alpaca_distill_using \
    --output_dir ${tmp_predictions_dir} \
    --per_device_eval_batch_size 1 \
    --max_samples 9999999999999 \
    --predict_with_generate \
    --overwrite_cache \
    --fp16 \
    --ddp_find_unused_parameters=False

# Process predictions to generate distilled dataset (single-threaded, as it's lightweight)
python "eval/gen_distilled_data.py" \
    --dataset ${train_dataset}_train \
    --predict_jsonl ${tmp_predictions_dir}/generated_predictions.jsonl

# Train on distilled dataset - Distributed training
torchrun \
  --standalone \
  --nproc_per_node=4 \
  --master_port=12346 \
  main.py \
    --stage sft \
    --model_name_or_path ${model_path} \
    --do_train \
    --dataset distilled_${train_dataset} \
    --template alpaca \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ${checkpoint_dir} \
    --overwrite_output_dir \
    --overwrite_cache \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate ${lr} \
    --num_train_epochs ${epoch} \
    --plot_loss \
    --bf16 \
    --ddp_find_unused=0 \
    --ddp_find_unused_parameters=False

# Evaluate math reasoning capabilities
for math_dataset in gsm8k multiarith;
do
    echo "Evaluation on ${math_dataset}:" >> ${result_file}
    output_dir="${output_folder}/${math_dataset}"
    torchrun \
      --standalone \
      --nproc_per_node=4 \
      --master_port=12347 \
      main.py \
        --stage sft \
        --model_name_or_path ${model_path} \
        --adapter_name_or_path ${checkpoint_dir} \
        --do_predict \
        --dataset "${math_dataset}_test" \
        --template gsm8k_infer \
        --output_dir ${output_dir} \
        --per_device_eval_batch_size 1 \
        --max_samples 9999999999999 \
        --predict_with_generate \
        --overwrite_cache \
        --fp16 \
        --ddp_find_unused_parameters=False

    python "eval/eval_math.py" --input_file "${output_dir}/generated_predictions.jsonl" >> ${result_file}
done

# Evaluate on OpenFunctions
echo "Evaluation on OpenFunctions:" >> ${result_file}
output_dir="${output_folder}/openfunction"
torchrun \
  --standalone \
  --nproc_per_node=4 \
  --master_port=12348 \
  main.py \
    --stage sft \
    --model_name_or_path ${model_path} \
    --adapter_name_or_path ${checkpoint_dir} \
    --do_predict \
    --dataset openfunction_test \
    --template alpaca \
    --output_dir ${output_dir} \
    --per_device_eval_batch_size 1 \
    --max_samples 9999999999999 \
    --predict_with_generate \
    --overwrite_cache \
    --fp16 \
    --ddp_find_unused_parameters=False

python "eval/eval_openfunction.py" --input_file "${output_dir}/generated_predictions.jsonl" >> ${result_file}

# Evaluate on HumanEval (single GPU due to small computational load)
output_path="${output_folder}/humaneval/result.json"
create_empty_file ${output_path}
CUDA_VISIBLE_DEVICES=0 python bigcode-evaluation-harness/main.py \
    --model ${model_path} \
    --peft_model ${checkpoint_dir} \
    --tasks humanevalsynthesize-python \
    --prompt octocoder \
    --do_sample False \
    --batch_size 1 \
    --allow_code_execution \
    --trust_remote_code \
    --metric_output_path ${output_path} \
    --max_length_generation 2048 \
    --precision fp16

python "eval/eval_humaneval.py" --input_file ${output_path} >> ${result_file}

# Predict on alpaca_eval (single GPU due to small computational load)
output_dir="${output_folder}/alpaca_eval"
CUDA_VISIBLE_DEVICES=0 python main.py \
    --stage sft \
    --model_name_or_path ${model_path} \
    --adapter_name_or_path ${checkpoint_dir} \
    --do_predict \
    --dataset alpaca_eval \
    --template alpaca \
    --output_dir ${output_dir} \
    --per_device_eval_batch_size 1 \
    --max_samples 9999999999999 \
    --predict_with_generate \
    --overwrite_cache \
    --fp16

python "eval/prepare_alpaca_eval.py" --input_file "${output_dir}/generated_predictions.jsonl" --output_file "${output_dir}/outputs.json"
# Execute the line below yourself if you want. Configuration of OpenAI API needed. The evaluation takes about $15.
# alpaca_eval --model_outputs "${output_dir}/outputs.json"

# Evaluate safety
for template in "alpaca" "alpaca_gcg";
do
    if [ ${template} == "alpaca" ];
    then
        safety_type="raw"
    else
        safety_type="jailbreak"
    fi
    echo "Evaluation on ${safety_type} safety:" >> ${result_file}
    output_dir="${output_folder}/advbench-${safety_type}"
    torchrun \
      --standalone \
      --nproc_per_node=4 \
      --master_port=12349 \
      main.py \
        --stage sft \
        --model_name_or_path ${model_path} \
        --adapter_name_or_path ${checkpoint_dir} \
        --do_predict \
        --dataset advbench \
        --template ${template} \
        --output_dir ${output_dir} \
        --per_device_eval_batch_size 1 \
        --max_samples 9999999999999 \
        --predict_with_generate \
        --overwrite_cache \
        --fp16 \
        --ddp_find_unused_parameters=False
    
    python "eval/keyword_eval_safety.py" --input_file "${output_dir}/generated_predictions.jsonl" >> ${result_file}
done

echo "Evaluation after fine-tuning on ${train_dataset} using ${type} finished successfully. Results are saved in ${result_file}."