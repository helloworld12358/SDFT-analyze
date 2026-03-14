# sdft 根目录 Python 文档（中文）

本文档对应 `README_ROOT_PY.md`，覆盖 `sdft/` 根目录下除 `main.py` 外的 Python 文件：

- `analyze_gradients_llama_factory.py`
- `batch_token_stats_aligned.py`
- `check_adam_state.py`
- `compare_answers.py`
- `compute_avg_grad_cosines.py`
- `merge_gradients.py`

## 一、逐文件说明

### 1) `analyze_gradients_llama_factory.py`
功能：LoRA 梯度核心分析脚本（支持分布式）。

核心量：
- `avg_flat`（平均梯度）
\[
\texttt{avg\_flat}=\frac{1}{T}\sum_t g_t
\]
- `trace_covariance`
\[
\sum_i\left(E[g_i^2]-(E[g_i])^2\right)
\]

输出（每个 rank）：`avg_grad.pt`、`norms.npy`、`stats.json`、`step_*.json`、`grad_norms_hist.png`。

### 2) `merge_gradients.py`
功能：合并各 rank 输出。

核心量：
\[
\texttt{merged\_avg}=\frac{\sum_r s_r\bar g_r}{\sum_r s_r}
\]
其中 `s_r` 为 rank 的 `processed_steps`。

输出：`merged/avg_grad.pt`、`merged/norms.npy`、`merged_stats.json` 等。

### 3) `compute_avg_grad_cosines.py`
功能：计算多个 `avg_grad.pt` 的余弦相似度。

公式：
\[
\cos(a,b)=\frac{a^Tb}{\|a\|\|b\|}
\]

输出：日志文件（当前版本不再写 CSV）。

### 4) `batch_token_stats_aligned.py`
功能：单文件 token 统计与可视化。

输出：`*_log.txt`、`*_token_topM.png`。

### 5) `compare_answers.py`
功能：比较原始与 distilled 数据回答差异。

输出：`data/<dataset>/<dataset>_diff.log`。

### 6) `check_adam_state.py`
功能：检查 checkpoint 的 `optimizer.pt` 中 Adam 二阶矩 `exp_avg_sq`。

输出：终端诊断信息（不写文件）。

## 二、依赖与顺序

### 常见梯度分析链路
1. `analyze_gradients_llama_factory.py`
2. `merge_gradients.py`
3. `compute_avg_grad_cosines.py`（可选）

### 可独立脚本
- `batch_token_stats_aligned.py`
- `compare_answers.py`
- `check_adam_state.py`

这三者不依赖梯度分析产物。
