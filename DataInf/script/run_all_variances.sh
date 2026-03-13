#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ================= 配置区域 (保留了你的修改) =================
PYTHON_SCRIPT="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/DataInf/src/compute_loss_variance_v2.py"
BASE_MODEL="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/model/Llama-2-7b-chat-hf"
CHECKPOINT_ROOT_EPOCH1="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/epoch1_checkpoints"
CHECKPOINT_ROOT="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/checkpoints"
DATA_ROOT="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/data"
OUTPUT_PARENT="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/DataInf/result"

MODEL="Llama2_7b"
MODEL_SHORT="Llama2_7b"
EPOCHS=("epoch_0" "epoch_1" "epoch_5")
METHODS=("sdft" "sft")
DATASETS="alpaca_eval,gsm8k,humaneval,multiarith,openfunction"

BATCH_SIZE=8
NUM_GPUS=4
PYTHON_EXEC="python"
LOG_ROOT="${SCRIPT_DIR}/logs_variance"

# ================= 准备工作 =================
mkdir -p "$LOG_ROOT"
mkdir -p "$OUTPUT_PARENT"

if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Error: python script not found at $PYTHON_SCRIPT" >&2
  exit 1
fi

# 设置主日志 (保留你的 exec 逻辑)
MASTER_LOG="${LOG_ROOT}/launcher_$(date +%s).log"
echo "All output redirected to: $MASTER_LOG"
exec >>"$MASTER_LOG" 2>&1

# ================= 生成任务列表 =================
TASKS=()
for EPOCH in "${EPOCHS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    ARGS=(--base_model_path "$BASE_MODEL" \
          --checkpoint_root_epoch1 "$CHECKPOINT_ROOT_EPOCH1" \
          --checkpoint_root "$CHECKPOINT_ROOT" \
          --model_name_short "$MODEL_SHORT" \
          --model "$MODEL" \
          --epoch "$EPOCH" \
          --method "$METHOD" \
          --data_root "$DATA_ROOT" \
          --datasets "$DATASETS" \
          --batch_size "$BATCH_SIZE" \
          --max_length 1024 \
          --output_root "$OUTPUT_PARENT" \
          --verbose)
    # 使用 printf %q 确保参数被安全转义，方便后续 eval
    TASKS+=("$(printf "%q " "${ARGS[@]}")")
  done
done

TOTAL=${#TASKS[@]}
echo "Total tasks: $TOTAL"
echo "Starting Batch Execution (Step size: $NUM_GPUS)"

# ================= 主循环 (分批执行) =================
# 每次步进 NUM_GPUS 个任务
for ((i=0; i<TOTAL; i+=NUM_GPUS)); do
  
  # 定义当前批次的数组
  BATCH_PIDS=()
  BATCH_LOGS=()
  BATCH_ARGS=()
  BATCH_GPU_ID=()

  echo "----------------------------------------------------------------"
  echo "Starting Batch from index $i to $((i+NUM_GPUS-1))"

  # 1. 启动当前批次的所有任务
  for ((g=0; g<NUM_GPUS; g++)); do
    idx=$((i + g))
    
    # 防止索引越界
    if [ $idx -ge $TOTAL ]; then
      break
    fi

    ARGS_STR="${TASKS[$idx]}"
    LOGFILE="${LOG_ROOT}/task_${idx}_gpu${g}_$(date +%s).log"
    
    echo "[LAUNCH] Task $idx on GPU $g"
    echo "Logs: $LOGFILE"

    # 后台启动任务
    eval "CUDA_VISIBLE_DEVICES=$g $PYTHON_EXEC \"$PYTHON_SCRIPT\" $ARGS_STR > \"$LOGFILE\" 2>&1 &"
    pid=$!
    
    # 记录信息以便后续 wait
    BATCH_PIDS+=("$pid")
    BATCH_LOGS+=("$LOGFILE")
    BATCH_ARGS+=("$ARGS_STR")
    BATCH_GPU_ID+=("$g")
    
    # 稍微错开启动时间，避免瞬间 IO 拥堵
    sleep 2
  done

  # 2. 等待当前批次的所有任务结束
  # 即使某个任务失败，我们也要等这一批全部跑完再处理下一批，保证显存安全
  for ((j=0; j<${#BATCH_PIDS[@]}; j++)); do
    pid="${BATCH_PIDS[$j]}"
    logfile="${BATCH_LOGS[$j]}"
    args_str="${BATCH_ARGS[$j]}"
    gpu="${BATCH_GPU_ID[$j]}"

    echo "Waiting for PID $pid (GPU $gpu)..."
    
    # 使用 if wait ... 结构，防止 set -e 因为子进程失败而退出主脚本
    if wait "$pid"; then
      echo "[OK] Task PID $pid finished successfully."
    else
      exit_code=$?
      echo "[ERR] Task PID $pid failed with exit code $exit_code."
      echo "See log: $logfile"
      
      echo "---- tail of $logfile ----"
      tail -n 20 "$logfile" || true
      echo "---- end tail ----"

      # OOM 检测与 Fallback 逻辑
      if grep -i -E "out of memory|cuda out of memory|killed|oom" "$logfile" >/dev/null 2>&1; then
        echo "[INFO] Detected OOM/KILL in $logfile -> Attempting FALLBACK (Serial execution without GPU restriction)"
        
        FALLBACK_LOG="${LOG_ROOT}/fallback_task_${pid}_$(date +%s).log"
        
        # Fallback: 不设置 CUDA_VISIBLE_DEVICES，让脚本/PyTorch 自动决定（或者改为 cpu，视代码逻辑而定）
        # 注意：这里使用串行阻塞执行，不加 &
        if eval "unset CUDA_VISIBLE_DEVICES; $PYTHON_EXEC \"$PYTHON_SCRIPT\" $args_str --prefer_auto_on_fail > \"$FALLBACK_LOG\" 2>&1"; then
           echo "[FALLBACK-OK] Fallback execution succeeded."
        else
           echo "[FALLBACK-ERR] Fallback execution also failed."
           echo "See log: $FALLBACK_LOG"
           tail -n 20 "$FALLBACK_LOG" || true
        fi
      else
        echo "[INFO] Failure was not recognized as OOM. No fallback attempted."
      fi
    fi
  done
  
  echo "Batch finished. Proceeding to next batch..."
done

echo "================================================================"
echo "All tasks finished."
echo "Results placed under: $OUTPUT_PARENT"