# DataInf/script Script Catalog

This file documents all scripts under `DataInf/script`.

## Path convention used by this folder

Most scripts in this folder assume:
- gradient vectors under `DataInf/output_grad/...`
- pairwise and analysis results under `DataInf/result/...`

## End-to-end order (recommended)

1. Generate gradients (`batch_run_grads_epoch*.sh`).
2. Build pairwise outputs (`pairwise_tasks*.sh` + `gpu_scheduler_epoch_*.sh` OR `run_pairwise_epoch0_1_5.sh`).
3. Post-process pairwise matrices (`aggregate_sft_sdft_diffs.sh`, `extract_top3_eigs_from_diffs.sh`, `analyze_pairwise_matrices*.py`).
4. Compute variance summaries (`run_all_variances.sh`).

Notes:
- `run_pairwise_epoch0_1_5.sh` uses the direct matrix builder in `DataInf/src` and is an alternative to `pairwise_tasks*.sh`.
- `pairwise_runner.sh` is a fixed one-combo runner (not full-batch scheduler).
- Clarification:
  - The standard direct builder is `compute_pairwise_from_grads_tagged.py` (used in `DataInf/scripts` path).
  - This folder's batch script uses `compute_pairwise_from_grads_tagged_localresults.py`, which has the same matrix math but different output-location preference and optional missing-grad recomputation.

## File-by-file details

| File | Main function | Key outputs / symbols | Dependency and order |
|---|---|---|---|
| `aggregate_sft_sdft_diffs.sh` | Compute SFT-SDFT matrix difference for each model/epoch and eigendecompose it. | Uses `A=sdft`, `B=sft`, `D=(B-A)*1e5`; saves `diff_matrix_*.npy/.csv`, `diff_eigenvalues_*.npy`, `diff_eigenvectors_*.npy`, summary txt files in `result/<model>/<epoch>/sdft/`. | Requires both `pairwise_matrix_<model>_<epoch>_sdft.npy` and `..._sft.npy` first. |
| `analyze_pairwise_matrices.py` | Load sdft/sft pairwise matrices, compute eigensystems, and write readable logs + optional plot. | Saves `analysis/analysis_log.txt`, `sdft_matrix.npy`, `sft_matrix.npy`, `diff_matrix.npy`, and matching eigvec files; may create `analysis_matrices.png`. | Requires pairwise matrices (or `sim_*.json` fallback) under `result/...`. |
| `analyze_pairwise_matrices_write_txt_with_corrs.py` | Convert covariance-like pairwise matrices to correlation matrices, then eigendecompose and log. | Correlation conversion uses `corr_ij = cov_ij / (sqrt(cov_ii)*sqrt(cov_jj))`; outputs `analysis_safe/analysis_corr_safe*.txt`. | Requires pairwise matrices or pairwise JSONs first. |
| `batch_run_grads_epoch0_sdft.sh` | Batch compute average gradients for epoch 0 / sdft track. | Runs `save_avg_grad_with_integrated_templates.py`; writes `.pt` to `output_grad/epoch_0/sdft/<model>/<dataset>.pt`; writes task logs. | First-stage script; no checkpoint LoRA required (runtime LoRA init path). |
| `batch_run_grads_epoch0_sft.sh` | Batch compute average gradients for epoch 0 / sft track. | Same output pattern under `output_grad/epoch_0/sft/...`. | First-stage script; run before pairwise scripts. |
| `batch_run_grads_epoch1_sdft.sh` | Batch compute average gradients for epoch 1 / sdft checkpoints. | Writes `.pt` under `output_grad/epoch_1/sdft/...`; includes `--lora_path` from `epoch1_checkpoints`. | Requires epoch1 sdft checkpoints to exist. |
| `batch_run_grads_epoch1_sft.sh` | Batch compute average gradients for epoch 1 / sft checkpoints. | Writes `.pt` under `output_grad/epoch_1/sft/...`. | Requires epoch1 sft checkpoints. |
| `batch_run_grads_epoch5_sdft.sh` | Batch compute average gradients for epoch 5 / sdft checkpoints. | Writes `.pt` under `output_grad/epoch_5/sdft/...`. | Requires checkpoint directory `checkpoints/<model>/sdft`. |
| `batch_run_grads_epoch5_sft.sh` | Batch compute average gradients for epoch 5 / sft checkpoints. | Writes `.pt` under `output_grad/epoch_5/sft/...`. | Requires checkpoint directory `checkpoints/<model>/sft`. |
| `extract_top3_eigs_from_diffs.sh` | For each diff matrix, extract top-left `3x3`, compute eigenpairs, and log normalized eigenvectors. | Saves `top3_eigs/top3_submatrix_eigs_<model>_<epoch>_<ts>.txt`. | Requires `diff_matrix_<...>.npy` produced by `aggregate_sft_sdft_diffs.sh`. |
| `gpu_scheduler_epoch_0.sh` | Batch launcher for epoch 0 pairwise jobs. | Calls `pairwise_tasks_epoch_0.sh` per model/method, prints resulting directory. | Requires epoch0 gradient `.pt` files first. |
| `gpu_scheduler_epoch_1.sh` | Batch launcher for epoch 1 pairwise jobs. | Calls `pairwise_tasks.sh` with `epoch_1`. | Requires epoch1 gradient `.pt` files first. |
| `gpu_scheduler_epoch_5.sh` | Batch launcher for epoch 5 pairwise jobs. | Calls `pairwise_tasks.sh` with `epoch_5`. | Requires epoch5 gradient `.pt` files first. |
| `pairwise_runner.sh` | Single fixed configuration runner with per-GPU occupancy control. | Produces `sim_<di>_<dj>.json` logs and final `pairwise_matrix_*.csv/.npy` in `result/.../pairwise_result`. | Requires all input grad `.pt` for configured datasets first. |
| `pairwise_tasks.sh` | Generic pairwise scheduler: compute all pair similarities then assemble matrix. | Calls `calc_dataset_similarity.py` for each pair, then `assemble_matrix.py`; outputs in `result/<model>/<epoch>/<method>/pairwise_result`. | Requires gradient `.pt` vectors and base/train data/checkpoint paths. |
| `pairwise_tasks_epoch_0.sh` | Epoch-0 variant of pairwise scheduler (no LoRA checkpoint argument). | Same output structure as `pairwise_tasks.sh`, using base model only for Hessian-like loop. | Requires epoch0 gradient `.pt` vectors. |
| `run_all_variances.sh` | Launch full `(epoch, method)` variance jobs in multi-GPU batches. | Calls `compute_loss_variance_v2.py`; saves `variance_results_<model>/<model_short>_<epoch>_<method>.txt` under `DataInf/result`. | Can run after checkpoints/data exist; independent from pairwise matrix generation. |
| `run_pairwise_epoch0_1_5.sh` | Sequentially run direct matrix builder for all models/epochs/methods. | Calls `compute_pairwise_from_grads_tagged_localresults.py`; outputs matrix/eigen files and summary txt (localresults layout). | Alternative pairwise path; requires grad vectors first. |
| `schemeA_10_train_test_rect.py` | Build rectangular train-vs-test matrices (`7x5`) for SchemeA. | Per `(epoch,method)` outputs `T_train_test_*_7x5` and `C_train_test_*_7x5` (`.npy/.csv/.json`), plus row-level summary files and optional `sft_minus_sdft` diffs. | Reuses existing test-task grads; may optionally compute missing train-self grads. |
| `run_schemeA_train_test_rect_all_epochs.sh` | Launcher for Step10 with split controls. Defaults: all train datasets + all epochs + both methods; can pass positional args: `<epoch> <method> [train_dataset]`. | Produces `DataInf/results/schemeA/train_test_rect/...` (or `DataInf/result/...` fallback). | Useful for 6-way split by `(epoch,method)` across multiple servers. |
| `schemeA_11_train_test_rect_hessian.py` | Build rectangular train-vs-test matrices (`7x5`) using Hessian proxy (non-inverse). | Per `(epoch,method)` outputs `T_train_test_hessian_*_7x5` and `C_train_test_hessian_*_7x5` (`.npy/.csv/.json`), plus row-level summary and optional `sft_minus_sdft` diffs. | Uses empirical Hessian score script (`calc_dataset_hessian_score.py`), no damping term. |
| `run_schemeA_train_test_rect_hessian_all_epochs.sh` | Launcher for Step11 (Hessian) with split controls. Positional args: `<epoch> <method> [train_dataset]`. | Produces `DataInf/results/schemeA/train_test_rect_hessian/...` (or `DataInf/result/...` fallback). | Useful for 6-way split by `(epoch,method)` across multiple servers. |

## Formula notes for complex scripts

### Difference matrix and eigensystem (`aggregate_sft_sdft_diffs.sh`)

\[
D = (M_{sft} - M_{sdft}) \times 10^5
\]

Then compute eigenvalues/eigenvectors of `D`, and L2-normalize each eigenvector column before saving.

### Correlation conversion (`analyze_pairwise_matrices_write_txt_with_corrs.py`)

\[
\mathrm{corr}_{ij} = \frac{\mathrm{cov}_{ij}}{\sqrt{\mathrm{cov}_{ii}}\sqrt{\mathrm{cov}_{jj}}}
\]

with safe handling for zero/invalid diagonal entries.

## Can scripts run independently?

- Gradient scripts (`batch_run_grads_*`) can run independently per epoch/method, but pairwise scripts depend on their `.pt` outputs.
- Pairwise post-processing scripts (`aggregate_*`, `extract_top3_*`, `analyze_*`) must run after pairwise matrices are available.
- Variance script launcher (`run_all_variances.sh`) is independent of pairwise matrices.
