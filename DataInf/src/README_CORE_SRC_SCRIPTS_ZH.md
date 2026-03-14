# DataInf/src 核心脚本说明（中文）

本文档对应 `README_CORE_SRC_SCRIPTS.md`，仅覆盖你指定的 8 个文件：

- `assemble_matrix.py`
- `calc_dataset_similarity.py`
- `compute_loss_variance.py`
- `compute_loss_variance_v2.py`
- `compute_pairwise_from_grads_tagged.py`
- `compute_pairwise_from_grads_tagged_localresults.py`
- `save_avg_grad.py`
- `save_avg_grad_with_integrated_templates.py`

## 一、逐文件说明

### 1) `save_avg_grad.py`
功能：对单个数据集计算 LoRA 平均梯度向量。

核心计算：
\[
\texttt{avg\_grad\_vector}=\frac{\sum_b(g_b\cdot |b|)}{N}
\]
其中 `g_b` 为 batch 梯度，`N` 为样本总数。

输出：`--output_path` 对应的 `.pt`（变量名 `avg_grad_vector`）。

依赖：基础模型、可选 LoRA、数据集文件。

### 2) `save_avg_grad_with_integrated_templates.py`
功能：同上，但内置模板（alpaca/gsm8k）来构造训练文本。

核心计算：与 `save_avg_grad.py` 相同。

输出：
- 指定 `--output_path` 时按指定保存；
- 否则默认保存到 `DataInf/src/result/output_grad/...`。

依赖：同上，常作为批处理脚本的梯度生成入口。

### 3) `calc_dataset_similarity.py`
功能：基于两个梯度向量与训练集样本梯度，计算一个 pair 的相似度分数。

核心计算（代码变量）：
\[
c=\frac{\langle v_1,g\rangle}{\lambda+\|g\|^2},\quad r\leftarrow r+(v_1-cg)
\]
\[
\texttt{similarity\_score}=\langle r_{final}, v_2\rangle
\]

输出：可选 JSON（`score`, `n_train`, `damping` 等）。

依赖：`grad1_path`、`grad2_path`、训练数据路径。

### 4) `assemble_matrix.py`
功能：把全部 `sim_i_j.json` 聚合成对称矩阵。

核心逻辑：读取每个 pair 的 `score` 填入 `M[i,j]` 和 `M[j,i]`。

输出：CSV（必有）+ NPY（可选）。

依赖：先有 `sim_*.json` 结果。

### 5) `compute_pairwise_from_grads_tagged.py`
功能：直接从多个 `.pt` 梯度向量构建整张 pairwise 矩阵并做特征分解。

这类脚本的定位：
- 它是**离线矩阵构建器**，只读已经算好的梯度向量；
- 它**不会**重新跑模型前向/反向；
- 它**不会**生成 `sim_i_j.json`（那是 `calc_dataset_similarity.py` 路径会产生的中间结果）。

核心计算：
- 默认内积：\(M=V^TV\)
- 开 `--normalize` 时为余弦相似矩阵。

输入读取位置（默认）：
- `DataInf/output_grads/<epoch>/<method>/<model>/<dataset>.pt`
- 默认数据集顺序：`alpaca_eval,gsm8k,humaneval,multiarith,openfunction`
- 也可通过 `--dataset_names` 覆盖。

输出（`DataInf/results/<model>/<epoch>/<method>/`）：
- `pairwise_matrix_*.npy/.csv/.txt`
- `eigenvalues_*.npy`
- `eigenvectors_*.npy`
- `summary_*.json`

依赖：对应目录下各数据集 `.pt` 梯度先存在。

典型命令：
- `python compute_pairwise_from_grads_tagged.py --model <m> --epoch <e> --method <sdft|sft> --datainf_root <...>`

### 6) `compute_pairwise_from_grads_tagged_localresults.py`
功能：上一个脚本的本地结果版；缺失梯度时可自动重算。

核心计算：矩阵与特征分解同上；若缺失则先重算
\[
\overline g=\frac{1}{N}\sum_i g_i
\]
并保存后再继续。

输出：
- pairwise/eigen 到 `DataInf/src/result/<model>/<epoch>/<method>/`
- 重算梯度到 `DataInf/src/result/output_grad/...`

依赖：已有梯度或开启 `--recompute_from_data` 且提供基础模型。

### 7) `compute_loss_variance.py`
功能：计算指定 `(model, epoch, method)` 下各测试集 loss 方差。

核心计算：对每条样本平均 loss 做 Welford 在线方差更新，最终
\[
\texttt{variance}=M_2/k,
\quad
\texttt{inverse\_of\_variance}=1/\texttt{variance}
\]

输出：`variance_results_<model>/<model_short>_<epoch>_<method>.txt`。

依赖：模型、数据、对应 checkpoint（epoch_1/epoch_5 时）。

### 8) `compute_loss_variance_v2.py`
功能：方差计算增强版（模板处理更完整，OOM 处理更稳）。

核心目标与输出格式：与 v1 保持一致。

## 二、运行顺序建议

### 典型流程 A（先 pair JSON，再组矩阵）
1. `save_avg_grad*.py` 先为各数据集生成梯度 `.pt`。
2. `calc_dataset_similarity.py` 逐 pair 计算。
3. `assemble_matrix.py` 聚合矩阵。
4. `compute_loss_variance*.py` 可独立运行。

### 典型流程 B（直接构矩阵）
1. 先生成梯度 `.pt`。
2. 直接用 `compute_pairwise_from_grads_tagged*.py`。
3. 方差脚本独立运行。
