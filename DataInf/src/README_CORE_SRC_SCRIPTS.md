# DataInf/src Core Scripts (Targeted)

This document covers only the 8 Python files requested:

- `assemble_matrix.py`
- `calc_dataset_similarity.py`
- `compute_loss_variance.py`
- `compute_loss_variance_v2.py`
- `compute_pairwise_from_grads_tagged.py`
- `compute_pairwise_from_grads_tagged_localresults.py`
- `save_avg_grad.py`
- `save_avg_grad_with_integrated_templates.py`

## 1) Script-by-script behavior

### `save_avg_grad.py`

Purpose:
- Compute the average LoRA gradient vector for one dataset.

Core computation:
- For each batch, extract flattened LoRA gradient vector `grad_vec`.
- Accumulate weighted sum by batch size:

\[
\texttt{sum\_grad\_vector} \leftarrow \sum_b (g_b \cdot |b|)
\]

- Final average:

\[
\texttt{avg\_grad\_vector} = \frac{\texttt{sum\_grad\_vector}}{\texttt{total\_samples}}
\]

Saved output:
- `--output_path` as a `.pt` tensor (variable name in code: `avg_grad_vector`).

Dependencies:
- Requires base model path, optional LoRA path, and dataset JSON/JSONL.
- Usually runs before pairwise scripts.

---

### `save_avg_grad_with_integrated_templates.py`

Purpose:
- Same goal as `save_avg_grad.py`, but with built-in formatting templates (`alpaca` / `gsm8k`) for input text construction.

Core computation:
- Same averaging math as above (`sum_grad_vector`, `total_samples`, `avg_grad_vector`).
- Adds template selection logic (`choose_template_by_dataset_path`).

Saved output:
- If `--output_path` is provided: save there.
- Else default local save under:
  - `DataInf/src/result/output_grad/<model_tag>/<dataset_tag>.pt`

Dependencies:
- Same as `save_avg_grad.py`.
- Used by many shell launchers for gradient generation.

---

### `calc_dataset_similarity.py`

Purpose:
- Compute one pairwise similarity score between two gradient vectors (`v1`, `v2`) using training-data gradients.

Core computation (code variables):
- For each training sample gradient `g_train`:

\[
\texttt{dot\_product}=\langle v_1, g\rangle,
\quad
c = \frac{\langle v_1, g\rangle}{\lambda + \|g\|^2}
\]

\[
r \leftarrow r + (v_1 - c g)
\]

- Final vector:

\[
r_{final} =
\begin{cases}
\frac{r}{n\lambda}, & \lambda>0 \\
\frac{r}{n}, & \lambda=0
\end{cases}
\]

- Similarity score:

\[
\texttt{similarity\_score} = \langle r_{final}, v_2\rangle
\]

Saved output:
- Optional `--out_path` JSON fields include:
  - `grad1`, `grad2`, `score`, `n_train`, `damping`.

Dependencies:
- Needs precomputed grad vectors (`grad1_path`, `grad2_path`) and train dataset.
- Used by `pairwise_tasks*.sh`/`pairwise_runner.sh` style workflows.

---

### `assemble_matrix.py`

Purpose:
- Assemble all `sim_i_j.json` results into one symmetric pairwise matrix.

Core computation:
- Read similarity files from `--result_dir`.
- Fill matrix entry `M[i,j]=score` and mirror `M[j,i]`.

Saved output:
- CSV (`--out_csv` or default `pairwise_matrix.csv` in result dir).
- Optional `.npy` (`--out_npy`).

Dependencies:
- Requires per-pair JSONs already generated (usually by `calc_dataset_similarity.py`).

---

### `compute_pairwise_from_grads_tagged.py`

Purpose:
- Build full pairwise matrix directly from stored `.pt` gradient vectors (no per-pair JSON loop).
- This script is an **offline matrix builder**: it does **not** run model forward/backward and does **not** call Hessian-loop sampling.

Core computation:
- Stack gradient vectors into matrix `V` of shape `(D, n)`.
- If not normalized:

\[
M = V^T V
\]

- If `--normalize`:

\[
M = (\hat V)^T \hat V
\]
where each column of `V` is L2-normalized.

- Compute eigen decomposition of `M` (prefer `eigh` when possible).

Exactly what it reads:
- `DataInf/output_grads/<epoch>/<method>/<model>/<dataset>.pt`
- Default dataset order:
  - `alpaca_eval,gsm8k,humaneval,multiarith,openfunction`
  - (or use `--dataset_names` to override)

Exactly what it does **not** produce:
- No `sim_i_j.json` files (those are produced by `calc_dataset_similarity.py` pipelines).
- No call to `assemble_matrix.py`.

Saved outputs (under `DataInf/results/<model>/<epoch>/<method>/`):
- `pairwise_matrix_<...>.npy`
- `pairwise_matrix_<...>.csv`
- `pairwise_matrix_<...>.txt`
- `eigenvalues_<...>.npy`
- `eigenvectors_<...>.npy`
- `summary_<...>.json`

Dependencies:
- Requires gradient `.pt` files under `output_grads/<epoch>/<method>/<model>/`.
- Typical command style:
  - `python compute_pairwise_from_grads_tagged.py --model <m> --epoch <e> --method <sdft|sft> --datainf_root <...>`

---

### `compute_pairwise_from_grads_tagged_localresults.py`

Purpose:
- Variant of pairwise builder that prefers local `DataInf/src/result/...` layout and can recompute missing gradients from raw data.

Core computation:
- Pairwise math is same as `compute_pairwise_from_grads_tagged.py`.
- Optional missing-gradient recompute path computes:

\[
\texttt{avg\_grad\_vector} = \frac{1}{N}\sum_i g_i
\]

then writes missing `.pt` first.

Saved outputs:
- Pairwise/eigen files under:
  - `DataInf/src/result/<model>/<epoch>/<method>/...`
- Recomputed gradients under:
  - `DataInf/src/result/output_grad/<epoch>/<method>/<model>/<dataset>.pt`

Dependencies:
- Needs existing grads OR `--recompute_from_data` with `--base_model_path` and discoverable dataset files.

---

### `compute_loss_variance.py`

Purpose:
- Compute per-dataset loss variance for one `(model, epoch, method)` combination.

Core computation:
- For each sample, compute token-level CE loss and sample mean loss `\ell_i`.
- Use Welford online variance update:

\[
\delta = \ell_i - \mu,
\quad
\mu \leftarrow \mu + \frac{\delta}{k},
\quad
M_2 \leftarrow M_2 + \delta(\ell_i-\mu)
\]

\[
\texttt{variance} = \frac{M_2}{k}
\]

- Also report inverse:

\[
\texttt{inverse\_of\_variance} = \frac{1}{\texttt{variance}}
\]
(if variance is 0, treated as `inf`).

Saved output:
- Text file:
  - `variance_results_<model>/<model_short>_<epoch>_<method>.txt`
- Each dataset line stores:
  - `variance=...`, `inverse_of_variance=...`

Dependencies:
- Requires base model, dataset files, and optional LoRA checkpoint (epoch-dependent).

---

### `compute_loss_variance_v2.py`

Purpose:
- Enhanced variance computation with template formatting and stronger OOM fallback behavior.

Core computation:
- Same variance objective/variables as v1 (`mean`, `M2`, `variance`).
- Uses template-specific text construction before tokenization.

Saved output:
- Same output format and filename convention as v1.

Dependencies:
- Same as v1.
- Preferred when template consistency is needed.

## 2) Dependency and run-order guidance

### Typical pipeline A (per-pair JSON then assemble)
1. Run one of the gradient savers (`save_avg_grad.py` or `save_avg_grad_with_integrated_templates.py`) for each dataset.
2. Run `calc_dataset_similarity.py` for all dataset pairs.
3. Run `assemble_matrix.py` to build final matrix.
4. Optional: run variance scripts independently (`compute_loss_variance*.py`).

### Typical pipeline B (direct pairwise build)
1. Run one of the gradient savers for each dataset.
2. Run `compute_pairwise_from_grads_tagged.py` (or localresults variant).
3. Optional: run variance scripts independently.

### Independence notes
- `compute_loss_variance.py` / `compute_loss_variance_v2.py` do not require pairwise outputs.
- `assemble_matrix.py` requires prior `sim_*.json` files.
- `compute_pairwise_from_grads_tagged*` requires gradient `.pt` vectors first.
