# Scheme A 并行运行说明（A1 不重算版）

## 1. 总原则
- A1（own-H task Gram 矩阵本体）默认不重算。
- A1 优先读取已有结果：
  - `DataInf/result/<train_dataset>/<epoch>/analysis/analysis_log.txt`
  - `DataInf/result/<train_dataset>/<epoch>/analysis_safe/analysis_corr_safe*.txt`
- 只有当某组缺失且显式开启 `--allow_recompute_missing` 时，才最小范围补算。
- task-level 固定 5 个测试任务：`alpaca_eval,gsm8k,humaneval,multiarith,openfunction`，矩阵始终应为 `5x5`。

## 2. 维度定义
- 训练集（7）：`gsm8k,openfunction,magicoder,alpaca,dolly,lima,openhermes`
- 方法（2）：`sft,sdft`
- epoch（3）：`epoch_0,epoch_1,epoch_5`

## 3. 推荐并行脚本
- `run_schemeA_ownH_collect.sh`
- `run_schemeA_crossH_epoch0.sh`
- `run_schemeA_crossH_epoch1.sh`
- `run_schemeA_crossH_epoch5.sh`
- `run_schemeA_mixedH_epoch0.sh`
- `run_schemeA_mixedH_epoch1.sh`
- `run_schemeA_mixedH_epoch5.sh`
- `run_schemeA_recover_all.sh`
- `run_schemeA_raw_rewrite_epoch5.sh`
- `run_schemeA_score_bridge.sh`
- `run_schemeA_final_summary.sh`

## 4. 环境变量
- `SCHEMEA_DATAINF_ROOT`：DataInf 根目录（默认脚本自动推导）
- `SCHEMEA_PYTHON`：Python 可执行文件（默认 `python`）
- `SCHEMEA_EXISTING_RESULT_ROOTS`：A1 读取已有结果的根目录（逗号分隔）
- `SCHEMEA_ALLOW_RECOMPUTE_MISSING`：A1 缺失时是否最小补算（`1` 开启，`0` 关闭，默认 `1`）
- `SCHEMEA_GPU_IDS`：并行 pair 任务可用 GPU 列表（例如 `0,1,2,3`）
- `SCHEMEA_PAIRWISE_WORKERS`：pair 并行工作进程数（`0` 表示自动按 GPU 数）
- `SCHEMEA_PAIR_TIMEOUT_SEC`：单个 pair 超时（秒，`0` 不限）

示例：
```bash
export SCHEMEA_DATAINF_ROOT=/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/SDFT-analysis/DataInf
export SCHEMEA_EXISTING_RESULT_ROOTS=/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/SDFT-analysis/DataInf/result
export SCHEMEA_GPU_IDS=0,1,2,3
export SCHEMEA_PAIRWISE_WORKERS=4
```

## 5. 建议提交顺序
1. `run_schemeA_ownH_collect.sh`
2. `run_schemeA_crossH_epoch0.sh` / `run_schemeA_crossH_epoch1.sh` / `run_schemeA_crossH_epoch5.sh`（并行）
3. `run_schemeA_mixedH_epoch0.sh` / `run_schemeA_mixedH_epoch1.sh` / `run_schemeA_mixedH_epoch5.sh`（并行）
4. `run_schemeA_raw_rewrite_epoch5.sh`
5. `run_schemeA_score_bridge.sh`
6. `run_schemeA_recover_all.sh`
7. `run_schemeA_final_summary.sh`

## 6. 关键输出目录
- own-H：`DataInf/results/schemeA/ownH/`
- cross-H：`DataInf/results/schemeA/crossH/`
- mixed-H：`DataInf/results/schemeA/mixedH/`
- raw/rewrite：`DataInf/results/schemeA/raw_rewrite/`
- score CKA/HSIC：`DataInf/results/schemeA/score_hsic/`
- 最终汇总：`DataInf/results/schemeA/final_summary/`

## 7. 失败处理
- 任一步骤缺缓存或路径不明确时，会生成 `unavailable_*.json`，不会静默跳过。
- mixed/common-H 如果无法构造，不会替换成其他 common-H 版本。
