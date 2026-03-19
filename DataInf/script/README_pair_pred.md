# Pair-level 预测实验（SDFT 相对 SFT 增益）

本目录新增了基于现有 5x5 几何矩阵的 pair-level 预测流水线，目标是预测：

- 符号：`Perf_SDFT(A→B) - Perf_SFT(A→B)` 是否为正
- 数值：尽可能拟合该差值大小

其中：

- `A`：训练集（7 个）
- `B`：测试任务（5 个，默认顺序：`alpaca_eval, gsm8k, humaneval, multiarith, openfunction`）

## 新增脚本

1) `pair_pred_00_inventory_and_loader.py`

- 作用：
  - 扫描矩阵来源（`analysis_log.txt` / `analysis_corr_safe.txt`）
  - 复用现有逻辑生成标准化性能表（优先 `schemeA_per_method_summary`，否则回退日志解析）
  - 输出 `inventory_mapping.json/md`
- 主要输出（默认在 `DataInf/results/pair_pred` 或 `DataInf/result/pair_pred`）：
  - `inventory_mapping.json`
  - `inventory_mapping.md`
  - `matrix_source_scan.csv`
  - `perf/perf_standardized.csv`
  - `perf/perf_standardized.json`

2) `pair_pred_01_extract_pair_features.py`

- 作用：
  - 读取 SFT/SDFT 的 `T(5x5)` 和 `C(5x5)`
  - 提取并计算：
    - `Cent_sft/sdft`, `DeltaCent`
    - `Load_sft/sdft`, `DeltaLoad`
    - `Self_sft/sdft`, `DeltaSelf`
    - `true_perf_diff = Perf_SDFT - Perf_SFT`
    - `true_sign`
- 主要输出：
  - `pair_features_all.csv/json`
  - `pair_features_epoch_0.csv/json`
  - `pair_features_epoch_1.csv/json`
  - `pair_features_epoch_5.csv/json`
  - `unavailable_pair_features.json`

3) `pair_pred_02_score_predictor.py`

- 作用：
  - 计算 `pred_score = alpha*DeltaCent + beta*DeltaLoad + gamma*DeltaSelf`
  - 支持模式：
    - `manual`：手动 `alpha/beta/gamma`
    - `grid`：全局网格搜索
    - `lodo`：按训练集留一（leave-one-train-dataset-out）网格拟合
- 输出指标：
  - `sign accuracy`
  - `Pearson / Spearman`（不可计算时给出原因）
  - `top-k overlap / precision`
- 主要输出：
  - `pair_pred_scores_<epoch>.csv/json/txt`
  - `pair_pred_summary_<epoch>.json`
  - `pair_pred_by_dataset_<epoch>.csv`
  - `pair_pred_fit_<epoch>.csv`（仅 `lodo`）

4) `pair_pred_03_final_summary.py`

- 作用：
  - 汇总 feature-level 与 predictor-level 结果
  - 汇总 unavailable 列表
- 主要输出：
  - `pair_pred_feature_eval.csv`
  - `pair_pred_summary.csv`
  - `pair_pred_summary.json`
  - `pair_pred_summary.md`

## bash 批量脚本

- `run_pair_pred_full.sh`
  - 顺序执行 00 -> 01 -> 02 -> 03
  - 默认在 `epoch_1` 做预测（可用环境变量改）
- `run_pair_pred_epoch1.sh`
  - 针对 `epoch_1` 的完整流程

## 快速最小验证（推荐先跑）

```bash
cd /inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/SDFT-analysis
export PAIR_PRED_DATAINF_ROOT=/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/SDFT-analysis/DataInf
export PAIR_PRED_PYTHON=/opt/conda/bin/python
bash DataInf/script/run_pair_pred_epoch1.sh
```

## 可复用缓存/结果

- 矩阵文本：`DataInf/result/<dataset>/<epoch>/analysis/analysis_log.txt` 与 `analysis_safe/analysis_corr_safe*.txt`
- 已有汇总：`DataInf/results/schemeA/final_summary/schemeA_per_method_summary.*`
- 日志解析回退：`gram_scheme_a_utils.discover_performance_rows`

## 注意事项

- 脚本会自动兼容 `result/` 与 `results/` 两套目录。
- 若输入缺失，会输出 `unavailable_*.json`，不会静默失败。
- 当前实现保留原始矩阵数值，不做负特征值截断或谱修正。
