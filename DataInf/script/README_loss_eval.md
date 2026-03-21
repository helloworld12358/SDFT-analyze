# Loss Eval (Epoch0/1/5 x 5 Tasks)

## 目标

对每个训练集（7个）计算两张 `3x5` 表（SFT + SDFT）：
- 行：`epoch_0, epoch_1, epoch_5`
- 列：`alpaca_eval, gsm8k, humaneval, multiarith, openfunction`
- 值：该模型在该测试集上的 `token-level CE loss`（越小越好）

并汇总成一个总 `txt`，包含 14 张 `3x5` 表（每个训练集下 SFT 和 SDFT 紧挨着）。

## 模板规则

- 数学模板（`gsm8k_format`）：`gsm8k`, `multiarith`
- 语言模板（`alpaca_format`）：`alpaca_eval`, `humaneval`, `openfunction`

## 关键约束

- `epoch_0` 强制不加载 adapter（LoRA）。
- `epoch_1` / `epoch_5` 按 `train_dataset + method` 自动找 checkpoint。
- 支持 `max_length<=0` 关闭截断（尽量保留原始文本长度；若超过模型上下文上限会报错并提示）。
- 支持自动 batch size 探测（按每个任务、每个模型组合自动找最大不 OOM 的 batch）。

## 新增脚本

- 主计算：
  - `loss_eval_01_compute_loss_tables.py`
- 最终汇总：
  - `loss_eval_02_final_summary.py`
- 画图：
  - `loss_eval_03_plot_curves.py`
- 单脚本（按参数指定训练集，适合7台服务器并行）：
  - `run_loss_eval_epoch015_dataset.sh`
- 总汇总：
  - `run_loss_eval_final_summary.sh`

## 输出目录

默认输出到：
- `DataInf/results/loss_eval`（若不存在则兼容到 `DataInf/result/loss_eval`）

关键文件：
- 每个训练集长表：`by_train_dataset/<train_dataset>/loss_rows_<train_dataset>_<method>_<epoch_tag>.csv|json`
- 每个训练集 `3x5` 表：`by_train_dataset/<train_dataset>/loss_table_<train_dataset>_<method>_<epoch_tag>.txt`
- 全部长表：`loss_rows_all_<train_tag>_<method_tag>_<epoch_tag>.csv|json`
- 最终14表汇总：`loss_tables_7x3x5_sft__sdft.txt|csv|json`
- 折线图：`plots/loss_curves_<train_dataset>.png`（共7张）

## 常用环境变量

- `LOSS_EVAL_DATAINF_ROOT`
- `LOSS_EVAL_PYTHON`
- `LOSS_EVAL_METHODS`（默认 `sft,sdft`）
- `LOSS_EVAL_BATCH_SIZE`（默认 `16`，建议H200保留或再调）
- `LOSS_EVAL_MAX_LENGTH`（默认 `0`，表示不截断；>0 表示截断到该长度）
- `LOSS_EVAL_AUTO_BATCH_SIZE`（默认 `1`）
- `LOSS_EVAL_BATCH_PROBE_MAX_BS`（默认 `64`）
- `LOSS_EVAL_BATCH_PROBE_BATCHES`（默认 `1`）
- `LOSS_EVAL_GPU_IDS`（默认读取 `SCHEMEA_GPU_IDS` 或 `CUDA_VISIBLE_DEVICES`）
- `LOSS_EVAL_TRAIN_DATASET`（若不用脚本参数传入）
- `LOSS_EVAL_DEVICE`（如 `cuda:0`）
- `LOSS_EVAL_OUTPUT_ROOT`（可选自定义输出目录）
