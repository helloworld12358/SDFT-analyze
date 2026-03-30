# DataInf/script 脚本说明（中文）

本文档对应 `README_SCRIPT_CATALOG.md`（位于 `DataInf/script`）。

## 一、本目录路径约定

本目录多数脚本默认使用：
- 梯度目录：`DataInf/output_grad/...`
- 结果目录：`DataInf/result/...`

## 二、推荐运行顺序

1. 先运行 `batch_run_grads_epoch*.sh` 生成梯度 `.pt`。
2. 再运行 `pairwise_tasks*.sh` / `gpu_scheduler_epoch_*.sh` 或 `run_pairwise_epoch0_1_5.sh` 生成 pairwise 矩阵。
3. 然后运行 `aggregate_sft_sdft_diffs.sh`、`extract_top3_eigs_from_diffs.sh`、`analyze_pairwise_matrices*.py`。
4. 最后可运行 `run_all_variances.sh` 生成方差结果。

补充说明：  
- 标准版“直接构矩阵”脚本是 `compute_pairwise_from_grads_tagged.py`（在 `DataInf/scripts` 路径里常用）。  
- 本目录对应的是 `compute_pairwise_from_grads_tagged_localresults.py`，核心矩阵计算一致，但输出目录偏好和缺失梯度补算逻辑不同。

## 三、各脚本功能与依赖（简表）

- `aggregate_sft_sdft_diffs.sh`：计算 `D=(M_sft-M_sdft)*1e5`，保存差分矩阵与特征值/向量。依赖：先有 sdft/sft pairwise 矩阵。
- `analyze_pairwise_matrices.py`：读取矩阵并写分析日志与图。依赖：先有矩阵或 `sim_*.json`。
- `analyze_pairwise_matrices_write_txt_with_corrs.py`：矩阵转相关系数后再做特征分析。依赖：先有矩阵。
- `batch_run_grads_epoch0/1/5_{sdft,sft}.sh`：批量生成平均梯度 `.pt`。依赖：模型/数据/对应 checkpoint。
- `extract_top3_eigs_from_diffs.sh`：对差分矩阵左上 3x3 做特征分解并输出文本。依赖：先有 `diff_matrix_*.npy`。
- `gpu_scheduler_epoch_0.sh`：批量调度 epoch0 pairwise。
- `gpu_scheduler_epoch_1.sh`：批量调度 epoch1 pairwise。
- `gpu_scheduler_epoch_5.sh`：批量调度 epoch5 pairwise。
- `pairwise_runner.sh`：固定组合的 pairwise 运行器（单套配置）。依赖：先有梯度 `.pt`。
- `pairwise_tasks.sh`：通用 pairwise 调度 + assemble。依赖：先有梯度 `.pt`。
- `pairwise_tasks_epoch_0.sh`：epoch0 版本（不带 `--lora_path`）。依赖：先有 epoch0 梯度。
- `run_all_variances.sh`：批量方差计算。与 pairwise 结果独立。
- `run_pairwise_epoch0_1_5.sh`：全量顺序调用 `compute_pairwise_from_grads_tagged_localresults.py`。
- `schemeA_10_train_test_rect.py`：构建训练集-测试集矩形矩阵（`7x5`），输出每个 `(epoch,method)` 的 `T/C`（`.npy/.csv/.json`）及逐行汇总，支持可选 `sft_minus_sdft` 差分。
- `run_schemeA_train_test_rect_all_epochs.sh`：Step10 运行入口；默认“全训练集 + 全 epoch + 双方法”，也支持位置参数 `<epoch> <method> [train_dataset]`，便于多机按 `(epoch,method)` 拆分并行。
- `schemeA_11_train_test_rect_hessian.py`：构建 Hessian（非逆）版本的训练集-测试集矩形矩阵（`7x5`），输出每个 `(epoch,method)` 的 `T/C`（`.npy/.csv/.json`）及逐行汇总，支持可选 `sft_minus_sdft` 差分。
- `run_schemeA_train_test_rect_hessian_all_epochs.sh`：Step11（Hessian）运行入口；支持位置参数 `<epoch> <method> [train_dataset]`，便于多机按 `(epoch,method)` 拆分并行。
- `loss_theory_01_forward_collect.py`：仅前向的理论诊断数据采集脚本（`epoch_1/epoch_5`），输出逐样本统计、随机 token 子样本统计、序列探针统计，支持流式读取、分批前向、断点续跑、分片并行。
- `loss_theory_02..09_*.py`：理论诊断分析脚本组（尾部形状、MGF、经验 Bernstein 覆盖、稳健均值、条件分析、长度归一化对照、依赖性、最终报告）。
- `run_loss_theory_collect_shard.sh`：前向采集分片运行入口（适合多机并行）。
- `run_loss_theory_analyze.sh`：分析脚本顺序运行入口（`02` 到 `09`）。

## 四、公式补充

### 1) 差分矩阵
\[
D=(M_{sft}-M_{sdft})\times 10^5
\]

### 2) 协方差到相关系数
\[
\mathrm{corr}_{ij}=\frac{\mathrm{cov}_{ij}}{\sqrt{\mathrm{cov}_{ii}}\sqrt{\mathrm{cov}_{jj}}}
\]

## 五、是否可独立运行

- 梯度脚本可按 epoch/method 独立跑；
- pairwise 相关脚本都依赖梯度 `.pt`；
- 分析/差分脚本依赖已生成的 pairwise 矩阵；
- 方差脚本可独立于 pairwise 流程。
