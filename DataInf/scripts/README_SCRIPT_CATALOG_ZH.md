# DataInf/scripts 脚本说明（中文）

本文档对应 `README_SCRIPT_CATALOG.md`（位于 `DataInf/scripts`）。

## 一、本目录路径约定

本目录多数脚本默认使用：
- 梯度目录：`DataInf/output_grads/...`（注意是 `output_grads`）
- 结果目录：`DataInf/results/...`（注意是 `results`）
- pairwise 子目录：`pairwise_results`（复数）

## 二、推荐运行顺序

1. 先 `batch_run_grads_epoch*.sh` 生成梯度；
2. 再 `pairwise_tasks*.sh` + `gpu_scheduler*.sh`（或 `run_pairwise_epoch0_1_5.sh`）；
3. 再做差分/分析（`aggregate_*`, `extract_top3_*`, `analyze_*`）；
4. 方差统计用 `run_all_variances.sh`。

## 三、各脚本功能与依赖（简表）

- `aggregate_sft_sdft_diffs.sh`：计算 `D=(M_sft-M_sdft)*1e5` 并做特征分解。依赖：先有 sft/sdft 矩阵。
- `analyze_pairwise_matrices.py`：矩阵与特征向量日志分析。依赖：先有矩阵或 `pairwise_results/sim_*.json`。
- `analyze_pairwise_matrices_write_txt_with_corrs.py`：相关系数形式分析。依赖：先有矩阵。
- `batch_run_grads_epoch0/1/5_{sdft,sft}.sh`：批量生成 `.pt` 梯度到 `output_grads`。
- `extract_top3_eigs_from_diffs.sh`：对差分 3x3 子矩阵做特征分析。
- `gpu_scheduler.sh`：批量调度多 epoch pairwise。
- `gpu_scheduler_epoch_0.sh`：epoch0 调度。
- `pairwise_runner.sh`：固定组合运行器。
- `pairwise_tasks.sh`：通用调度 + assemble。
- `pairwise_tasks_epoch_0.sh`：epoch0 版本。
- `run_all_variances.sh`：批量方差计算（调用 `compute_loss_variance.py`）。
- `run_pairwise_epoch0_1_5.sh`：顺序调用 `compute_pairwise_from_grads_tagged.py` 全量构矩阵。

## 四、公式补充

### 1) 差分矩阵
\[
D=(M_{sft}-M_{sdft})\times 10^5
\]

### 2) 协方差到相关系数
\[
\mathrm{corr}_{ij}=\frac{\mathrm{cov}_{ij}}{\sqrt{\mathrm{cov}_{ii}}\sqrt{\mathrm{cov}_{jj}}}
\]

### 3) `compute_pairwise_from_grads_tagged.py` 到底做什么

- 输入：读取已经存在的梯度向量  
  - `output_grads/<epoch>/<method>/<model>/<dataset>.pt`
- 核心：把多个梯度堆成矩阵 `V`，计算
  - 默认 `M=V^TV`，若开 `--normalize` 则是余弦相似矩阵。
- 输出：直接把矩阵和特征分解结果写到
  - `results/<model>/<epoch>/<method>/`
- 关键区别：
  - 它**不会**生成 `sim_i_j.json`；
  - 它**不会**调用 `assemble_matrix.py`；
  - 它是 `run_pairwise_epoch0_1_5.sh` 走的“直接构矩阵”路径。

## 五、是否可独立运行

- 梯度生成脚本可分开跑；
- pairwise 依赖先有梯度；
- 矩阵分析依赖先有矩阵；
- 方差可独立于 pairwise 流程运行。
