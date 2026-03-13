#!/bin/bash

# run_cosine_computation.sh
# 运行余弦相似度计算脚本的bash脚本（已同步到只使用 log，不再生成 CSV）
# 修改说明：
# - 不再传递 --csv_name 参数（Python 已移除 CSV 写入）
# - 改为只传递 --log_name（包含 OUTPUT_PREFIX）
# - 保持：不重定向 python 输出、使用 python3（可由 PYTHON_BIN 覆盖）
# - 在 python 失败时立即退出并返回 python 的退出码

set -e
set -o pipefail

# 可覆盖的 python 可执行文件（默认为 python3）
PYTHON_BIN="${PYTHON_BIN:-python3}"

# 设置基本路径
ROOT_DIR="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/analysis"
RESULTS_DIR="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/results"
PYTHON_SCRIPT="/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/compute_avg_grad_cosines.py"

# 确保结果目录存在
mkdir -p "$RESULTS_DIR"

# 设置参数
DATASETS=("alpaca" "gsm8k" "openfunction" "magicoder" "dolly" "lima" "openhermes")
METHODS=("sdft" "sft")
COMPARATORS=(
    "gradient_analysis_METHOD_alpacaeval"
    "gradient_analysis_METHOD_gsm8ktest"
    "gradient_analysis_METHOD_openfunctiontest"
)

# 生成时间戳用于文件命名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "余弦相似度计算脚本 - 多次调用模式"
echo "=========================================="
echo "Root directory: $ROOT_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Python script: $PYTHON_SCRIPT"
echo "Python bin: $PYTHON_BIN"
echo "Timestamp: $TIMESTAMP"
echo ""

# 计算期望的配对数量
TOTAL_DATASETS=${#DATASETS[@]}
TOTAL_METHODS=${#METHODS[@]}
TOTAL_COMPARATORS=${#COMPARATORS[@]}
TOTAL_CALLS=$((TOTAL_DATASETS * TOTAL_METHODS))

echo "Configuration:"
echo "  Datasets ($TOTAL_DATASETS): ${DATASETS[*]}"
echo "  Methods ($TOTAL_METHODS): ${METHODS[*]}"
echo "  Comparators per call ($TOTAL_COMPARATORS): gradient_analysis_METHOD_alpacaeval gradient_analysis_METHOD_gsm8ktest gradient_analysis_METHOD_openfunctiontest"
echo "  Total Python calls: $TOTAL_CALLS"
echo "  Total comparisons: $((TOTAL_CALLS * TOTAL_COMPARATORS))"
echo ""

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    echo "Please make sure compute_avg_grad_cosines.py exists at the specified path"
    exit 1
fi

# 检查根目录是否存在
if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Root directory does not exist: $ROOT_DIR"
    exit 1
fi

echo "Starting computation..."
echo ""

# 计数器
CALL_COUNT=0
SUCCESS_COUNT=0
FAIL_COUNT=0

# 循环遍历所有数据集和方法组合
for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        CALL_COUNT=$((CALL_COUNT + 1))

        echo "[$CALL_COUNT/$TOTAL_CALLS] Processing: Dataset=$DATASET, Method=$METHOD"

        # 设置基准文件路径
        BASE_FILE="$ROOT_DIR/$DATASET/gradient_analysis_$METHOD/merged/avg_grad.pt"

        # 构建目标文件路径列表
        TARGET_FILES=()
        for COMP in "${COMPARATORS[@]}"; do
            # 替换METHOD占位符
            COMP_NAME=${COMP/METHOD/$METHOD}
            TARGET_FILE="$ROOT_DIR/$DATASET/$COMP_NAME/merged/avg_grad.pt"
            TARGET_FILES+=("$TARGET_FILE")
        done

        # 生成输出文件前缀（仅用于 log 名）
        OUTPUT_PREFIX="${DATASET}_${METHOD}_${TIMESTAMP}"
        LOG_NAME="${OUTPUT_PREFIX}_cosine_results.log"

        echo "  Base file: $BASE_FILE"
        echo "  Target files:"
        for TF in "${TARGET_FILES[@]}"; do
            echo "    $TF"
        done
        echo "  Log name: $LOG_NAME"

        # 检查基准文件是否存在
        if [ ! -f "$BASE_FILE" ]; then
            echo "  [WARNING] Base file does not exist: $BASE_FILE"
            echo "  Skipping this combination..."
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo ""
            continue
        fi

        # 调用Python脚本（**不重定向输出**，以便直接看到 traceback）
        echo "  Running Python script..."
        "$PYTHON_BIN" "$PYTHON_SCRIPT" \
            --base_files "$BASE_FILE" \
            --target_files "${TARGET_FILES[@]}" \
            --results_dir "$RESULTS_DIR" \
            --log_name "$LOG_NAME"

        PY_EXIT=$?
        if [ $PY_EXIT -ne 0 ]; then
            echo "  [ERROR] Python script failed with exit code $PY_EXIT."
            echo "  The script will terminate immediately. Check the Python traceback above for details."
            # 立刻终止主脚本并返回 python 的退出码
            exit $PY_EXIT
        else
            echo "  [SUCCESS] Completed successfully"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi

        echo ""
    done
done

echo "=========================================="
echo "所有计算完成！"
echo "=========================================="
echo "Statistics:"
echo "  Total calls: $CALL_COUNT"
echo "  Successful: $SUCCESS_COUNT"
echo "  Failed: $FAIL_COUNT"
# 计算成功率，避免除以0
if [ "$CALL_COUNT" -gt 0 ]; then
    SUCCESS_RATE=$(awk "BEGIN{printf \"%.1f\", $SUCCESS_COUNT * 100 / $CALL_COUNT}")
else
    SUCCESS_RATE="0.0"
fi
echo "  Success rate: ${SUCCESS_RATE}%"
echo ""
echo "Results saved in: $RESULTS_DIR"
echo "Output files pattern: ${TIMESTAMP}_cosine_results.log"
echo ""

# 显示生成的文件（仅匹配本次 timestamp 的 log 文件）
echo "Generated files:"
ls -la "$RESULTS_DIR"/*${TIMESTAMP}*.log 2>/dev/null || echo "No files found with timestamp $TIMESTAMP"

echo ""
echo "Done!"
