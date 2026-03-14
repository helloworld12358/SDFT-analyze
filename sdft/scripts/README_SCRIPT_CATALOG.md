# sdft/scripts Script Catalog

This document covers all `.py` and `.sh` files under `sdft/scripts`.

## 1) High-level dependency map

Common end-to-end paths in this folder:

1. Train/evaluate base SFT/SDFT models (dataset folders: `sdft.sh`, `sft.sh`, `*_1epoch.sh`, `*_copy.sh`).
2. Generate gradient analyses (`compute_gradient*.sh`, `gradient/*.sh`, `gradient_lora/*.sh`).
3. Merge per-rank gradients (`merge_gradients.py` called by wrapper scripts).
4. Compare average-gradient directions (`gradient*/compute_sim.sh` calling `compute_avg_grad_cosines.py`).
5. Run theorem-style checkpoint analysis (`run_adaptive*.sh` calling `theorem_experiment*.py`).
6. Aggregate experiment JSONs into tables (`aggregate*.py`).

Can scripts run independently?
- Utility scripts (`run_all_token_stats_aligned.sh`, `test_seed_LM*.sh`, `utils.sh`) can run independently with required inputs.
- Aggregation scripts require prior `experiment_results/*.json`.
- Cosine scripts require prior `analysis/*/merged/avg_grad.pt` files.

## 2) Important math formulas used by key scripts

### Theorem experiment family (`theorem_experiment*.py`)

Core variables:
- `delta = theta_end - theta_0`
- Per-sample gradient list before/after finetune: `g_i^{before}`, `g_i^{after}`

Quadratic form estimator implemented in code:

\[
v^T H v \approx \frac{1}{n}\sum_{i=1}^{n}(v^T g_i)^2
\]

Metrics saved by these scripts include:

\[
C_{start}=\frac{1}{2}\cdot \frac{1}{n}\sum_i (\delta^T g_i^{before})^2,
\quad
C_{end}=\frac{1}{2}\cdot \frac{1}{n}\sum_i (\delta^T g_i^{after})^2
\]

\[
\texttt{dot\_avggrad\_delta}=\langle \overline g_{after}, \delta\rangle,
\quad
V_{align}=-\texttt{dot\_avggrad\_delta}
\]

Some variants also save:
- `dot_avggrad_delta_before`
- `V_align_before`

### Aggregation family (`aggregate*.py`)

`aggregate_results.py` score:

\[
\texttt{total} = V_{align} + w_{start} C_{start} + C_{end}
\]

`aggregate_results_modified.py` score (as coded):

\[
\texttt{total} = w_{dot}\,\texttt{dot\_avggrad\_delta\_before} - \texttt{dot\_avggrad\_delta} + w_{start} C_{start} + C_{end}
\]

`aggregate_and_fit.py` / `aggregate_fit.py` use sign-prediction objectives on:

\[
\texttt{score} = w_{dot}\,\Delta x_1 + w_{start}\,\Delta x_2 + \Delta b
\]

and compare predicted sign of `(sft - sdft)` against label signs.

## 3) Versioned script families (same goal, different behavior)

This section clarifies groups that look similar but are not interchangeable.

### A) Filename suffix semantics used across this folder

| Suffix / naming | Typical meaning | What usually changes |
|---|---|---|
| `_copy` | Alternate launcher implementation for same task | Usually switches to `torchrun`/multi-GPU scheduling or different orchestration/retry logic. |
| `_1epoch` | 1-epoch training/eval variant | Uses `epoch1_*` output/checkpoint/result dirs and `num_train_epochs=1` (often with denser logging). |
| `_base` | Base-model-only gradient analysis | Leaves `ADAPTER_DIR` empty; script builds in-memory LoRA for analysis path. |
| `_gsm8ktest` | Gradient analysis against gsm8k-test style evaluation split | Changes split/output target naming for test-style comparison files. |
| `_modified` | Metric-extended variant | Adds extra saved fields (commonly `dot_avggrad_delta_before`, `V_align_before`). |
| `_lasttry` | Strict-output safety variant | Enforces strict `--output_path` file behavior and no implicit fallback output dir. |
| `_random` | Repeated-run averaging variant | Runs `n_runs` times and saves averaged theorem metrics, with optional DDP/disk-streamed grads. |

### B) `theorem_experiment*.py` family: exact differences

| Script | Warm-up behavior | Output policy | Extra metrics | Parallel/compute behavior |
|---|---|---|---|---|
| `theorem_experiment.py` | Performs warm-up updates (default one step). | If `--output_path` is a dir, auto-generates `<ckpt>__<data>.json`; also accepts file path. | Base fields: `delta_norm`, `dot_avggrad_delta`, `V_align`, `C_start`, `C_end`. | Single run per invocation. |
| `theorem_experiment_modified.py` | Same warm-up pattern as base. | Same flexible output-path behavior as base. | Adds `dot_avggrad_delta_before`, `V_align_before`. | Single run per invocation. |
| `theorem_experiment_lasttry.py` | Warm-up constants kept but warm-up path disabled in this strict variant. | Requires explicit file `--output_path`; refuses directory path and refuses overwrite of non-empty existing file. | Includes before/after dot and align fields. | Single run; designed for safer batch dispatch. |
| `theorem_experiment_random.py` | Uses warm-up each run, then aggregates over repeated runs. | Accepts file or dir and resolves output path; also manages per-task `tmp_dir`. | Saves averaged metrics and `n_runs` (including before/after align terms). | Supports DDP paths, disk streaming of per-sample grads, optional `--reuse_base_model`. |

### C) `run_adaptive.sh` vs `run_adaptive_copy.sh`

| Script | Backend python | Scheduler style | Skip/retry policy | Output root |
|---|---|---|---|---|
| `run_adaptive.sh` | `theorem_experiment_lasttry.py` | Memory-aware launcher based on free GPU memory and per-GPU slots. | No built-in “valid JSON skip”; dispatches full generated task grid. | `experiment_results/` |
| `run_adaptive_copy.sh` | `theorem_experiment_random.py` | Per-GPU queue (one active task per listed GPU slot, refill on completion). | Skips already-valid JSON results (`delta_norm` present and no `error` flag); aborts all on task failure. | `experiment_results_random/` |

### D) `gradient/compute_sim.sh` vs `gradient_lora/compute_sim.sh`

| Script | Expected base analysis path family | Comparator name pattern |
|---|---|---|
| `gradient/compute_sim.sh` | `analysis/<dataset>/gradient_analysis_original_<method>/...` | `gradient_analysis_original_METHOD_*` |
| `gradient_lora/compute_sim.sh` | `analysis/<dataset>/gradient_analysis_<method>/...` | `gradient_analysis_METHOD_*` |

Operationally both call `compute_avg_grad_cosines.py`; the key difference is which analysis directory naming convention they assume.

### E) Dataset-folder script families: where differences matter

| Family | Common shared logic | Key differences you should care about |
|---|---|---|
| `sdft.sh` vs `sft.sh` | Same downstream eval suite (math/openfunction/humaneval/alpaca_eval/safety). | `sdft.sh` adds a distillation stage (`*_distill` prediction + `gen_distilled_data.py`) before LoRA training; `sft.sh` trains directly on `<dataset>_train`. |
| `*_1epoch.sh` vs non-`1epoch` | Same task graph and eval tasks. | Uses `epoch1_predictions/`, `epoch1_results/`, `epoch1_checkpoints/`; training epochs are set to 1; often uses `torchrun` with multi-GPU defaults. |
| `*_copy.sh` vs non-`copy` | Intends same logical pipeline outcome. | Usually replaces plain `python main.py` calls with distributed `torchrun` commands and explicit NCCL-related environment setup. |
| `compute_gradient.sh` vs `compute_gradient_base.sh` | Both run `analyze_gradients_llama_factory.py` then `merge_gradients.py`. | `compute_gradient.sh` points to a real adapter checkpoint; `compute_gradient_base.sh` keeps adapter empty and performs base-model-only gradient analysis. |
| `compute_gradient_gsm8ktest.sh` | Same analyzer/merge pipeline. | Changes split/output target for gsm8k-test-style comparison runs. |
| `alpaca/test.sh` | Same analyzer and merge components. | Debug-oriented single-process/single-GPU run (`python`), usually with reduced steps for quick sanity checks. |

## 4) Root-level files in `sdft/scripts`

| File | Main function | Key outputs | Dependency |
|---|---|---|---|
| `aggregate_and_fit.py` | Massive log-grid search for fitted `(w_start, w_dot)` and aggregated tables. | `experiment_results/aggregate_fitted_*.txt` | Requires theorem experiment JSON files in `experiment_results/`. |
| `aggregate_fit.py` | Aggregation with user-specified fixed weights (`SPECIFIED_W_START`, `SPECIFIED_W_DOT`), plus numeric diff table. | `experiment_results/aggregate_fitted_*.txt` | Requires theorem experiment JSON files. |
| `aggregate_results.py` | Build single summary table from `V_align`, `C_start`, `C_end`. | `experiment_results/aggregate_single_txt_w*.txt` | Requires theorem experiment JSON files. |
| `aggregate_results_modified.py` | Modified weighted aggregation including `dot_avggrad_delta_before` and `dot_avggrad_delta`. | `experiment_results/aggregate_single_txt_w*_d*.txt` | Requires theorem experiment JSON files with dot fields. |
| `run_adaptive.sh` | Dispatch theorem experiment jobs across GPUs using checkpoint grid (domain x ckpt_type x test set). | JSON outputs in `experiment_results/`; log files in `tmp_logs/` | Requires checkpoints, datasets, and `theorem_experiment_lasttry.py`. |
| `run_adaptive_copy.sh` | Similar dispatcher with per-GPU queue + valid-result skip logic, calling `theorem_experiment_random.py`. | JSON outputs in `experiment_results_random/` | Requires checkpoints, datasets, theorem random script. |
| `run_all_token_stats_aligned.sh` | Batch token statistics for train + distilled files of each dataset. | `results/<dataset>/*_log.txt` and `*_token_topM.png` | Requires `batch_token_stats_aligned.py`, model tokenizer path, dataset files. |
| `test_seed_LM.sh` | Evaluate seed model (single GPU) across math/openfunction/humaneval/alpaca_eval/safety tasks. | `results/seed_LM.log` and per-task prediction artifacts | Requires `main.py`, eval scripts, seed model path. |
| `test_seed_LM_copy.sh` | Multi-GPU (`torchrun`) variant of seed evaluation pipeline. | Same output family as above | Requires same inputs plus multi-GPU runtime. |
| `theorem_experiment.py` | Base theorem experiment variant with warm-up step before `C_start` computation. | One JSON result file per run (`...__...json`) | Requires base model, checkpoint, dataset. |
| `theorem_experiment_lasttry.py` | Strict-output-path theorem variant (requires file path, no fallback dir). | JSON result at required `--output_path` | Requires base model, checkpoint, dataset. |
| `theorem_experiment_modified.py` | Warm-up variant adding `dot_avggrad_delta_before` and `V_align_before`. | JSON with extended fields | Requires base model, checkpoint, dataset. |
| `theorem_experiment_random.py` | Multi-run averaged theorem variant with optional base-model reuse and disk-streamed gradients. | Aggregated JSON including `n_runs` and averaged metrics | Requires base model, checkpoint, dataset; optional DDP environment. |
| `utils.sh` | Helper shell utilities (currently `create_empty_file`). | No standalone artifact; used by other shell scripts | Sourced by eval/train scripts. |

## 5) `gradient/` folder (all files)

| File | Main function | Key outputs | Dependency |
|---|---|---|---|
| `gradient/compute_sim.sh` | Run cosine comparisons across datasets/methods using "original" gradient-analysis outputs. | Log files in configured `results_dir` | Requires merged `avg_grad.pt` files and `compute_avg_grad_cosines.py`. |
| `gradient/sdft_alpaca_eval_gradient.sh` | Compute gradient analysis for sdft adapter on alpaca-eval split. | `analysis/<adapter>/gradient_analysis_original_sdft_alpacaeval/` + `merged/` | Requires adapter checkpoint and analyzer scripts. |
| `gradient/sdft_gsm8k_testgradient.sh` | Compute sdft gradient analysis on gsm8k test. | `analysis/<adapter>/gradient_analysis_original_sdft_gsm8ktest/` + `merged/` | Requires adapter checkpoint and analyzer scripts. |
| `gradient/sdft_openfunction_testgradient.sh` | Compute sdft gradient analysis on openfunction test. | `analysis/<adapter>/gradient_analysis_original_sdft_openfunctiontest/` + `merged/` | Same dependency as above. |
| `gradient/sft_alpaca_eval_gradient.sh` | Compute sft gradient analysis on alpaca-eval split. | `analysis/<adapter>/gradient_analysis_original_sft_alpacaeval/` + `merged/` | Requires sft adapter checkpoint. |
| `gradient/sft_gsm8k_testgradient.sh` | Compute sft gradient analysis on gsm8k test. | `analysis/<adapter>/gradient_analysis_original_sft_gsm8ktest/` + `merged/` | Requires sft adapter checkpoint. |
| `gradient/sft_openfunction_testgradient.sh` | Compute sft gradient analysis on openfunction test. | `analysis/<adapter>/gradient_analysis_original_sft_openfunctiontest/` + `merged/` | Requires sft adapter checkpoint. |
| `gradient/train_sdft_gradient.sh` | Loop over all training domains and run original-sdft gradient analysis + merge. | `analysis/<dataset>/gradient_analysis_original_sdft/merged/avg_grad.pt` | Requires data/model and analyzer scripts. |
| `gradient/train_sft_gradient.sh` | Loop over all training domains and run original-sft gradient analysis + merge. | `analysis/<dataset>/gradient_analysis_original_sft/merged/avg_grad.pt` | Requires data/model and analyzer scripts. |

## 6) `gradient_lora/` folder (all files)

| File | Main function | Key outputs | Dependency |
|---|---|---|---|
| `gradient_lora/compute_sim.sh` | Cosine comparison over lora-based gradient analyses. | Log files in configured result dir | Requires merged `avg_grad.pt` files from lora runs. |
| `gradient_lora/sdft_alpaca_eval_gradient.sh` | sdft lora gradient analysis on alpaca-eval split. | `analysis/<adapter>/gradient_analysis_sdft_alpacaeval/` + `merged/` | Requires `epoch1_checkpoints/<adapter>/sdft`. |
| `gradient_lora/sdft_gsm8k_testgradient.sh` | sdft lora gradient analysis on gsm8k test. | `analysis/<adapter>/gradient_analysis_sdft_gsm8ktest/` + `merged/` | Same as above. |
| `gradient_lora/sdft_openfunction_testgradient.sh` | sdft lora gradient analysis on openfunction test. | `analysis/<adapter>/gradient_analysis_sdft_openfunctiontest/` + `merged/` | Same as above. |
| `gradient_lora/sft_alpaca_eval_gradient.sh` | sft lora gradient analysis on alpaca-eval split. | `analysis/<adapter>/gradient_analysis_sft_alpacaeval/` + `merged/` | Requires `epoch1_checkpoints/<adapter>/sft`. |
| `gradient_lora/sft_gsm8k_testgradient.sh` | sft lora gradient analysis on gsm8k test. | `analysis/<adapter>/gradient_analysis_sft_gsm8ktest/` + `merged/` | Same as above. |
| `gradient_lora/sft_openfunction_testgradient.sh` | sft lora gradient analysis on openfunction test. | `analysis/<adapter>/gradient_analysis_sft_openfunctiontest/` + `merged/` | Same as above. |
| `gradient_lora/train_sdft_gradient.sh` | Loop over all domains for lora-sdft gradient analysis + merge. | `analysis/<dataset>/gradient_analysis_sdft/merged/avg_grad.pt` | Requires `epoch1_checkpoints/<dataset>/sdft`. |
| `gradient_lora/train_sft_gradient.sh` | Loop over all domains for lora-sft gradient analysis + merge. | `analysis/<dataset>/gradient_analysis_sft/merged/avg_grad.pt` | Requires `epoch1_checkpoints/<dataset>/sft`. |

## 7) Dataset folders under `sdft/scripts/*` (all files)

General patterns:
- `compute_gradient*.sh`: run gradient analysis wrappers (usually `torchrun analyze_gradients_llama_factory.py`) then merge.
- `sdft*.sh`: run distillation-based training/eval pipeline (`main.py` + eval scripts).
- `sft*.sh`: run direct SFT training/eval pipeline (`main.py` + eval scripts).
- `*_1epoch.sh`: epoch1 checkpoint naming variants.
- `*_copy.sh`: alternative launcher variants with near-identical logic.

### `alpaca/`

| File | Main function | Dependency |
|---|---|---|
| `alpaca/compute_gradient.sh` | Gradient analysis for `alpaca` train with adapter path. | Needs analyzer + adapter checkpoint. |
| `alpaca/compute_gradient_base.sh` | Gradient analysis for `alpaca` train with base-only (no adapter dir). | Needs analyzer + base model. |
| `alpaca/compute_gradient_gsm8ktest.sh` | Gradient analysis on gsm8k test-style data using alpaca adapter. | Needs analyzer + alpaca adapter checkpoint. |
| `alpaca/sdft.sh` | Distill (`tmp_predictions`) then finetune and run eval suites. | Needs `main.py`, eval scripts, data/model paths. |
| `alpaca/sdft_1epoch.sh` | Epoch1 variant of `sdft.sh` (distributed). | Same dependencies with epoch1 output dirs. |
| `alpaca/sdft_copy.sh` | Alternate distributed variant of `sdft.sh`. | Same dependency as `sdft.sh`. |
| `alpaca/sft.sh` | Standard SFT train + eval pipeline. | Needs `main.py`, eval scripts, data/model paths. |
| `alpaca/sft_1epoch.sh` | Epoch1 distributed SFT variant. | Same dependency with epoch1 output dirs. |
| `alpaca/test.sh` | Single-process gradient-analysis test wrapper. | Needs analyzer scripts + adapter checkpoint. |

### `dolly/`

| File | Main function | Dependency |
|---|---|---|
| `dolly/compute_gradient.sh` | Gradient analysis for `dolly` train with adapter. | Analyzer + dolly adapter checkpoint. |
| `dolly/compute_gradient_base.sh` | Base-only gradient analysis for `dolly`. | Analyzer + base model. |
| `dolly/compute_gradient_gsm8ktest.sh` | Gradient analysis on gsm8k test-style data with dolly adapter. | Analyzer + dolly adapter checkpoint. |
| `dolly/sdft.sh` | Distill-then-finetune + evaluation for dolly domain. | `main.py`, eval scripts, data/model paths. |
| `dolly/sdft_1epoch.sh` | Epoch1 distributed sdft variant for dolly. | Same dependency with epoch1 outputs. |
| `dolly/sdft_copy.sh` | Alternate distributed sdft variant for dolly. | Same dependency as above. |
| `dolly/sft.sh` | Standard dolly SFT + evaluation pipeline. | `main.py`, eval scripts. |
| `dolly/sft_1epoch.sh` | Epoch1 distributed SFT pipeline for dolly. | Same dependency as `sft.sh`. |

### `gsm8k/`

| File | Main function | Dependency |
|---|---|---|
| `gsm8k/compute_gradient.sh` | Gradient analysis for gsm8k train with adapter. | Analyzer + gsm8k adapter checkpoint. |
| `gsm8k/compute_gradient_base.sh` | Base-only gradient analysis for gsm8k. | Analyzer + base model. |
| `gsm8k/compute_gradient_gsm8ktest.sh` | Gradient analysis on gsm8k test-style dataset with gsm8k adapter. | Analyzer + adapter checkpoint. |
| `gsm8k/sdft.sh` | Distill-then-finetune + evaluation for gsm8k. | `main.py`, eval scripts, data/model paths. |
| `gsm8k/sdft_1epoch.sh` | Epoch1 distributed sdft variant for gsm8k. | Same dependency with epoch1 dirs. |
| `gsm8k/sdft_copy.sh` | Alternate distributed sdft launcher for gsm8k. | Same dependency as above. |
| `gsm8k/sft.sh` | Standard gsm8k SFT + evaluation. | `main.py`, eval scripts. |
| `gsm8k/sft_1epoch.sh` | Epoch1 distributed SFT for gsm8k. | Same dependency as above. |
| `gsm8k/sft_copy.sh` | Alternate SFT launcher variant for gsm8k. | Same dependency as `sft.sh`. |

### `lima/`

| File | Main function | Dependency |
|---|---|---|
| `lima/compute_gradient.sh` | Gradient analysis for lima train with adapter. | Analyzer + lima adapter checkpoint. |
| `lima/compute_gradient_base.sh` | Base-only gradient analysis for lima. | Analyzer + base model. |
| `lima/compute_gradient_gsm8ktest.sh` | Gradient analysis on gsm8k test-style data with lima adapter. | Analyzer + adapter checkpoint. |
| `lima/sdft.sh` | Distill-then-finetune + evaluation for lima. | `main.py`, eval scripts. |
| `lima/sdft_1epoch.sh` | Epoch1 distributed sdft variant for lima. | Same dependency with epoch1 dirs. |
| `lima/sdft_copy.sh` | Alternate distributed sdft launcher for lima. | Same dependency as above. |
| `lima/sft.sh` | Standard lima SFT + evaluation. | `main.py`, eval scripts. |
| `lima/sft_1epoch.sh` | Epoch1 distributed SFT for lima. | Same dependency as above. |
| `lima/sft_copy.sh` | Alternate SFT launcher variant for lima. | Same dependency as `sft.sh`. |

### `magicoder/`

| File | Main function | Dependency |
|---|---|---|
| `magicoder/compute_gradient.sh` | Gradient analysis for magicoder train with adapter. | Analyzer + magicoder adapter checkpoint. |
| `magicoder/compute_gradient_base.sh` | Base-only gradient analysis for magicoder. | Analyzer + base model. |
| `magicoder/compute_gradient_gsm8ktest.sh` | Gradient analysis on gsm8k test-style data with magicoder adapter. | Analyzer + adapter checkpoint. |
| `magicoder/sdft.sh` | Distill-then-finetune + evaluation for magicoder. | `main.py`, eval scripts. |
| `magicoder/sdft_1epoch.sh` | Epoch1 distributed sdft variant for magicoder. | Same dependency with epoch1 dirs. |
| `magicoder/sdft_copy.sh` | Alternate distributed sdft launcher for magicoder. | Same dependency as above. |
| `magicoder/sft.sh` | Standard magicoder SFT + evaluation. | `main.py`, eval scripts. |
| `magicoder/sft_1epoch.sh` | Epoch1 distributed SFT for magicoder. | Same dependency as above. |
| `magicoder/sft_copy.sh` | Alternate SFT launcher variant for magicoder. | Same dependency as `sft.sh`. |

### `openfunction/`

| File | Main function | Dependency |
|---|---|---|
| `openfunction/compute_gradient.sh` | Gradient analysis for openfunction train with adapter. | Analyzer + openfunction adapter checkpoint. |
| `openfunction/compute_gradient_base.sh` | Base-only gradient analysis for openfunction. | Analyzer + base model. |
| `openfunction/compute_gradient_gsm8ktest.sh` | Gradient analysis on gsm8k test-style data with openfunction adapter. | Analyzer + adapter checkpoint. |
| `openfunction/sdft.sh` | Distill-then-finetune + evaluation for openfunction. | `main.py`, eval scripts. |
| `openfunction/sdft_1epoch.sh` | Epoch1 distributed sdft variant for openfunction. | Same dependency with epoch1 dirs. |
| `openfunction/sdft_copy.sh` | Alternate distributed sdft launcher for openfunction. | Same dependency as above. |
| `openfunction/sft.sh` | Standard openfunction SFT + evaluation. | `main.py`, eval scripts. |
| `openfunction/sft_1epoch.sh` | Epoch1 distributed SFT for openfunction. | Same dependency as above. |

### `openhermes/`

| File | Main function | Dependency |
|---|---|---|
| `openhermes/compute_gradient.sh` | Gradient analysis for openhermes train with adapter. | Analyzer + openhermes adapter checkpoint. |
| `openhermes/compute_gradient_base.sh` | Base-only gradient analysis for openhermes. | Analyzer + base model. |
| `openhermes/compute_gradient_gsm8ktest.sh` | Gradient analysis on gsm8k test-style data with openhermes adapter. | Analyzer + adapter checkpoint. |
| `openhermes/sdft.sh` | Distill-then-finetune + evaluation for openhermes. | `main.py`, eval scripts. |
| `openhermes/sdft_1epoch.sh` | Epoch1 distributed sdft variant for openhermes. | Same dependency with epoch1 dirs. |
| `openhermes/sdft_copy.sh` | Alternate distributed sdft launcher for openhermes. | Same dependency as above. |
| `openhermes/sft.sh` | Standard openhermes SFT + evaluation. | `main.py`, eval scripts. |
| `openhermes/sft_1epoch.sh` | Epoch1 distributed SFT for openhermes. | Same dependency as above. |
| `openhermes/sft_copy.sh` | Alternate SFT launcher variant for openhermes. | Same dependency as `sft.sh`. |

## 8) Practical run-order recommendations

- If your goal is checkpoint-level theorem metrics (`C_start`, `C_end`, `V_align`):
  1. Produce checkpoints (`sdft.sh` / `sft.sh` families).
  2. Run `run_adaptive.sh` or `run_adaptive_copy.sh`.
  3. Run `aggregate*.py` for summary tables.

- If your goal is gradient-direction comparison:
  1. Run `gradient/*.sh` or `gradient_lora/*.sh` to generate merged `avg_grad.pt`.
  2. Run `gradient/compute_sim.sh` or `gradient_lora/compute_sim.sh`.

- Token stats and seed-model evaluation can be run independently whenever the required data/model paths exist.
