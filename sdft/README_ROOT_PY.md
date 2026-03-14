# sdft Root Python Scripts (excluding main.py)

This document covers Python files directly under `sdft/` except `main.py`.

Covered files:
- `analyze_gradients_llama_factory.py`
- `batch_token_stats_aligned.py`
- `check_adam_state.py`
- `compare_answers.py`
- `compute_avg_grad_cosines.py`
- `merge_gradients.py`

## 1) Script-by-script details

### `analyze_gradients_llama_factory.py`

Purpose:
- Core LoRA-gradient analyzer in llmtuner context, supporting distributed execution.

Core math and symbols:
- Per-step flattened gradient vector: `step_grad`.
- Average gradient:

\[
\texttt{avg\_flat} = \frac{1}{T}\sum_{t=1}^{T} g_t
\]

- Trace covariance estimator (code variable: `trace_cov`):

\[
\texttt{trace\_cov} = \sum_i \left(E[g_i^2] - (E[g_i])^2\right)
\]

(using `mean_sq - mean^2` then sum over dimensions).

Saved outputs per rank (`output_dir/rank<k>/`):
- `avg_grad.pt`
- `norms.npy`
- `stats.json` (contains `trace_covariance`, `avg_grad_norm`, etc.)
- `step_<idx>_meta.json`
- `grad_norms_hist.png`

Dependencies:
- Requires llmtuner runtime, model/dataset/template config.
- Usually followed by `merge_gradients.py`.

---

### `merge_gradients.py`

Purpose:
- Merge rank-wise gradient analysis outputs into a single merged result.

Core math and symbols:
- Weighted average by processed steps:

\[
\texttt{merged\_avg} = \frac{\sum_r s_r \cdot \bar g_r}{\sum_r s_r}
\]

where `s_r` is rank `processed_steps`.

- Optional weighted approximation of trace covariance across ranks.

Saved outputs (`<output_dir>/merged/` by default):
- `avg_grad.pt`
- `norms.npy` (concatenated)
- `merged_stats.json`
- `merged_step_meta.json` (if available)

Dependencies:
- Requires outputs from `analyze_gradients_llama_factory.py` (`rank*/`).

---

### `compute_avg_grad_cosines.py`

Purpose:
- Compute cosine similarities between `avg_grad.pt` files.

Core math:

\[
\cos(a,b)=\frac{a^T b}{\|a\|\|b\|}
\]

Uses float64 for stable dot/norm and clamps result to `[-1,1]`.

Saved outputs:
- Log-only output in `--results_dir` (default log name `cosine_results.log`).
- No CSV is written in current version (argument kept only for compatibility).

Dependencies:
- Requires existing `avg_grad.pt` files (usually after `merge_gradients.py`).

---

### `batch_token_stats_aligned.py`

Purpose:
- Compute per-file token frequency statistics and plot top tokens.

Key computed values:
- `total_texts`
- `total_tokens`
- `avg_tokens_per_text`
- top-token frequency counter (`Counter`).

Saved outputs per input file:
- `<basename>_log.txt`
- `<basename>_token_topM.png`

Dependencies:
- Requires tokenizer/model path and input data file.
- Independent from gradient pipeline.

---

### `compare_answers.py`

Purpose:
- Compare answers between original and distilled dataset files.

Key computed values:
- `total`
- differing indices list (`diffs`)
- reported `accuracy` (as implemented by script formula).

Saved output:
- `data/<dataset>/<dataset>_diff.log`

Dependencies:
- Requires both:
  - `data/<dataset>/<dataset>_train.json`
  - `data/<dataset>/distilled_<dataset>.json`

---

### `check_adam_state.py`

Purpose:
- Inspect `optimizer.pt` in a checkpoint and verify Adam second-moment buffers (`exp_avg_sq`).

Outputs:
- Console diagnostics only (shape/dtype/mean/min/max and validation summary).

Dependencies:
- Requires checkpoint directory containing `optimizer.pt`.
- Independent debugging utility.

## 2) Run-order and dependency guidance

### Typical gradient-analysis chain
1. `analyze_gradients_llama_factory.py`
2. `merge_gradients.py`
3. `compute_avg_grad_cosines.py` (optional comparison stage)

### Independent utilities
- `batch_token_stats_aligned.py`
- `compare_answers.py`
- `check_adam_state.py`

These three can run independently and do not require the gradient-analysis outputs.
