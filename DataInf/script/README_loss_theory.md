# Loss Theory Suite (Forward-Only)

This suite runs theory-oriented diagnostics using only frozen-LLM forward passes.

## Scope

- No training / finetuning
- No backward / gradients / Hessian / MI
- No parameter perturbation and re-evaluation
- Epoch scope: `epoch_1, epoch_5` (epoch_0 intentionally excluded)

## Scripts

- `loss_theory_01_forward_collect.py`
  - Streaming dataset read
  - Batched forward pass
  - Resume via `state.json`
  - Random token subsampling (no full token dump)
  - Outputs per combo:
    - `sample_stats.csv|parquet`
    - `token_subsample_stats.csv|parquet`
    - `token_sequence_probe_stats.csv|parquet`
- `loss_theory_02_tail_shape.py`
- `loss_theory_03_mgf_check.py`
- `loss_theory_04_emp_bernstein.py`
- `loss_theory_05_robust_mean.py`
- `loss_theory_06_conditional.py`
- `loss_theory_07_len_ablation.py`
- `loss_theory_08_dependence.py`
- `loss_theory_09_final_report.py`
- `loss_theory_10_combo_matrix_tables.py`
  - Build combo-level tree tables in order: `metric -> epoch -> method`
  - Each table is `7 (train datasets) x 5 (tasks)`
  - Outputs:
    - `analysis/combo_matrix/combo_metric_long.csv|json`
    - `analysis/combo_matrix/combo_metric_tree_tables.json|txt`
    - `analysis/combo_matrix/unavailable_combo_metric_tables.json`

## Run on 7 machines

On each machine:

```bash
cd /inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/SDFT-analysis
export LOSS_THEORY_DATAINF_ROOT=/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/SDFT-analysis/DataInf
export LOSS_THEORY_PYTHON=/opt/conda/bin/python
export LOSS_THEORY_GPU_IDS=0,1,2,3
export LOSS_THEORY_SHARD_COUNT=7
export LOSS_THEORY_SHARD_INDEX=<0~6>
bash DataInf/script/run_loss_theory_collect_shard.sh
```

Then on one machine only:

```bash
bash DataInf/script/run_loss_theory_analyze.sh
```

## Output root

Default output root:

- `DataInf/results/loss_theory` (fallback: `DataInf/result/loss_theory`)

Key artifacts:

- `by_combo/.../sample_stats.csv|parquet`
- `by_combo/.../token_subsample_stats.csv|parquet`
- `analysis/tail_shape/*`
- `analysis/mgf_check/*`
- `analysis/emp_bernstein/*`
- `analysis/robust_mean/*`
- `analysis/conditional/*`
- `analysis/len_ablation/*`
- `analysis/dependence/*`
- `analysis/final_report/loss_theory_final_report.json|md`
- `analysis/combo_matrix/combo_metric_tree_tables.txt|json`
