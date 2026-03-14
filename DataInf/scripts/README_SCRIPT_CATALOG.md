# DataInf/scripts Script Catalog

This file documents all scripts under `DataInf/scripts`.

## Path convention used by this folder

Most scripts in this folder assume:
- gradient vectors under `DataInf/output_grads/...`
- pairwise and analysis outputs under `DataInf/results/...`
- pairwise JSON directory named `pairwise_results` (plural)

This is the main difference versus `DataInf/script`.

## End-to-end order (recommended)

1. Generate gradients (`batch_run_grads_epoch*.sh`).
2. Build pairwise outputs (`pairwise_tasks*.sh` + `gpu_scheduler*.sh` OR `run_pairwise_epoch0_1_5.sh`).
3. Post-process matrices (`aggregate_sft_sdft_diffs.sh`, `extract_top3_eigs_from_diffs.sh`, `analyze_pairwise_matrices*.py`).
4. Compute variance summaries (`run_all_variances.sh`).

## File-by-file details

| File | Main function | Key outputs / symbols | Dependency and order |
|---|---|---|---|
| `aggregate_sft_sdft_diffs.sh` | Compute matrix difference `D=(M_sft-M_sdft)*1e5` and eigendecompose it. | Saves `diff_matrix_*.npy/.csv`, `diff_eigenvalues_*.npy`, `diff_eigenvectors_*.npy`, summary txt under `results/<model>/<epoch>/sdft/`. | Needs `pairwise_matrix_..._sdft.npy` and `..._sft.npy` first. |
| `analyze_pairwise_matrices.py` | Load pairwise matrices, eigendecompose sdft/sft/diff, create readable logs and plots. | Outputs `analysis/analysis_log.txt`, `*_matrix.npy`, `*_eigvecs.npy`, optional `analysis_matrices.png`. | Needs pairwise matrices or `pairwise_results/sim_*.json` first. |
| `analyze_pairwise_matrices_write_txt_with_corrs.py` | Convert matrix to correlation form, eigendecompose, and log safely. | Correlation formula `corr_ij = cov_ij/(sqrt(cov_ii)*sqrt(cov_jj))`; outputs `analysis_safe/analysis_corr_safe*.txt`. | Needs pairwise matrices first. |
| `batch_run_grads_epoch0_sdft.sh` | Batch average-grad generation for epoch0 sdft. | Calls `src/save_avg_grad.py`; writes `.pt` to `output_grads/epoch_0/sdft/<model>/<dataset>.pt`. | First-stage script (no LoRA checkpoint required). |
| `batch_run_grads_epoch0_sft.sh` | Batch average-grad generation for epoch0 sft. | Writes `.pt` under `output_grads/epoch_0/sft/...`. | First-stage script for epoch0 sft. |
| `batch_run_grads_epoch1_sdft.sh` | Batch average-grad generation for epoch1 sdft checkpoints. | Writes `.pt` under `output_grads/epoch_1/sdft/...`. | Requires `epoch1_checkpoints/<model>/sdft`. |
| `batch_run_grads_epoch1_sft.sh` | Batch average-grad generation for epoch1 sft checkpoints. | Writes `.pt` under `output_grads/epoch_1/sft/...`. | Requires `epoch1_checkpoints/<model>/sft`. |
| `batch_run_grads_epoch5_sdft.sh` | Batch average-grad generation for epoch5 sdft checkpoints. | Writes `.pt` under `output_grads/epoch_5/sdft/...`. | Requires `checkpoints/<model>/sdft`. |
| `batch_run_grads_epoch5_sft.sh` | Batch average-grad generation for epoch5 sft checkpoints. | Writes `.pt` under `output_grads/epoch_5/sft/...`. | Requires `checkpoints/<model>/sft`. |
| `extract_top3_eigs_from_diffs.sh` | Extract top-left 3x3 block eigenpairs from each diff matrix. | Writes `top3_eigs/top3_submatrix_eigs_<model>_<epoch>_<ts>.txt`. | Requires diff matrices from `aggregate_sft_sdft_diffs.sh`. |
| `gpu_scheduler.sh` | Batch launcher for multiple `(model, epoch, method)` pairwise jobs. | Calls `pairwise_tasks.sh` and prints resulting directory. | Needs gradients generated first. |
| `gpu_scheduler_epoch_0.sh` | Epoch-0 batch launcher using epoch0 pairwise scheduler. | Calls `pairwise_tasks_epoch_0.sh` and returns results directory. | Needs epoch0 gradients first. |
| `pairwise_runner.sh` | Single fixed-combo runner with per-GPU occupancy control. | Produces `pairwise_results/sim_*.json`, then `pairwise_matrix_*.csv/.npy`. | Needs all input gradient `.pt` files first. |
| `pairwise_tasks.sh` | Generic pairwise scheduler with GPU-memory gating and fail-fast handling. | Calls `calc_dataset_similarity.py` per pair and `assemble_matrix.py` at end; outputs in `results/.../pairwise_results`. | Needs gradient vectors and required model/checkpoint/data paths. |
| `pairwise_tasks_epoch_0.sh` | Epoch-0 pairwise scheduler without `--lora_path`. | Same output layout as `pairwise_tasks.sh`, base-model-only mode. | Needs epoch0 gradients first. |
| `run_all_variances.sh` | Launch all `(epoch, method)` variance jobs in GPU batches. | Calls `compute_loss_variance.py`; saves `variance_results_<model>/<model_short>_<epoch>_<method>.txt` under `DataInf/results`. | Independent from pairwise matrix outputs. |
| `run_pairwise_epoch0_1_5.sh` | Sequential direct pairwise build for all model/epoch/method combos. | Calls `compute_pairwise_from_grads_tagged.py`; writes matrix/eigen files and summary JSON/TXT in `DataInf/results/...`. | Alternative pairwise path; needs gradients first. |

## Formula notes for complex scripts

### Difference matrix (`aggregate_sft_sdft_diffs.sh`)

\[
D = (M_{sft} - M_{sdft}) \times 10^5
\]

### Correlation conversion (`analyze_pairwise_matrices_write_txt_with_corrs.py`)

\[
\mathrm{corr}_{ij} = \frac{\mathrm{cov}_{ij}}{\sqrt{\mathrm{cov}_{ii}}\sqrt{\mathrm{cov}_{jj}}}
\]

with safe replacement for invalid divisions.

### What `compute_pairwise_from_grads_tagged.py` actually does

- Input: already-saved gradient vectors  
  - `output_grads/<epoch>/<method>/<model>/<dataset>.pt`
- Core math: stack vectors into `V`, then compute
  - `M = V^T V` (default), or cosine matrix when `--normalize` is set.
- Output: writes matrix/eigen files directly under
  - `results/<model>/<epoch>/<method>/`
- Important distinction:
  - It does **not** generate `sim_i_j.json`.
  - It does **not** call `assemble_matrix.py`.
  - It is the direct/offline path used by `run_pairwise_epoch0_1_5.sh`.

## Can scripts run independently?

- Gradient-generation scripts can run per epoch/method independently.
- Pairwise scripts require gradients first.
- Matrix analysis scripts require pairwise matrices first.
- Variance launcher can run independently of pairwise generation.
