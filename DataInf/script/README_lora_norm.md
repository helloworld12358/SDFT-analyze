# LoRA Norm Drift Analysis

This mini-pipeline measures parameter drift from the base model using LoRA weights.

## Metrics

- `BA_norm_global`:
  - per module: `||(alpha/r) * B @ A||_F`
  - global: `sqrt(sum_module norm^2)`
- `A_norm_global`:
  - global: `sqrt(sum_module ||A||_F^2)`
- `B_norm_global`:
  - global: `sqrt(sum_module ||B||_F^2)`

`epoch_0` is fixed to zero (base model without adapter).

## Scripts

- `lora_norm_01_collect.py`
  - collects norms for `(train_dataset, method, epoch)`
- `lora_norm_02_summary_plot.py`
  - builds txt/csv/json summaries and plots
- `run_lora_norm_full.sh`
  - one-command runner

## Run

```bash
cd /inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/SDFT-analysis
export LORA_NORM_DATAINF_ROOT=/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/SDFT-analysis/DataInf
export LORA_NORM_PYTHON=/opt/conda/bin/python
bash DataInf/script/run_lora_norm_full.sh
```

Optional:

```bash
export LORA_NORM_OUTPUT_ROOT=/inspire/.../DataInf/results/lora_norm
```

## Outputs

Default output root:

- `DataInf/results/lora_norm`

Key files:

- `lora_norm_rows.csv`
- `lora_norm_rows.json`
- `lora_norm_wide.csv`
- `lora_norm_wide.json`
- `lora_norm_summary.txt`
- `plots/lora_norm_ba_curves.png`
- `plots/lora_norm_a_curves.png`
- `plots/lora_norm_b_curves.png`
- `unavailable_lora_norm.json`

## Runtime

- GPU is **not required**.
- CPU-only is enough (this pipeline only loads adapter weights and computes norms).

