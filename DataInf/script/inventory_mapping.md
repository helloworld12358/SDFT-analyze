# Pair-level 实验 Inventory 说明（代码侧）

本文件说明 `pair_pred_00_inventory_and_loader.py` 在代码实现中如何确认并复用现有仓库逻辑。

## 已确认并复用的现有逻辑

1. 矩阵文本解析（复用）

- `gram_scheme_a_utils.parse_method_matrices_from_analysis_txt`
- `gram_scheme_a_utils.load_existing_ownh_from_analysis`

用途：从

- `analysis/analysis_log.txt` 读取 `T`
- `analysis_safe/analysis_corr_safe*.txt` 读取 `C`

并提取 `sft/sdft` 两套 5x5 矩阵。

2. 路径兼容（复用）

- `gram_scheme_a_utils.resolve_result_root`
- `gram_scheme_a_utils.resolve_existing_result_roots`

用途：自动兼容 `DataInf/result` 与 `DataInf/results`。

3. 性能读取（复用优先级）

- 优先：`DataInf/results/schemeA/final_summary/schemeA_per_method_summary.*`
- 回退：`gram_scheme_a_utils.discover_performance_rows`

## 任务顺序约定

默认沿用 `gram_scheme_a_utils.DEFAULT_TASKS`：

1. `alpaca_eval`
2. `gsm8k`
3. `humaneval`
4. `multiarith`
5. `openfunction`

pair-level 特征提取按该顺序解释 5x5 行列索引。

## 与本地仓库现状的差异（已在代码中自适应）

- 本地 Git 仓库通常不包含完整云端 `DataInf/result/<dataset>/<epoch>/analysis*` 文件；
- 代码不会假设本地必须有这些数据，而是：
  - 在运行时扫描可用 result roots；
  - 对缺失组合写出 `unavailable_*.json`；
  - 在最终汇总中显式列出 unavailable。

## 运行后应查看的动态 mapping 文件

执行 `pair_pred_00_inventory_and_loader.py` 后，会在 `pair_pred` 输出目录生成：

- `inventory_mapping.json`
- `inventory_mapping.md`

该动态文件包含真实扫描到的路径、存在性、来源与缺失情况，是最终实验排查依据。
