# sdft/scripts 脚本总览（中文）

本文档对应 `sdft/scripts/README_SCRIPT_CATALOG.md` 的中文版。

## 一、总体依赖链

常见流程：
1. 训练/蒸馏得到 checkpoint（各数据集目录下 `sdft.sh` / `sft.sh` / `*_1epoch.sh`）。
2. 梯度分析（`compute_gradient*.sh`、`gradient/*.sh`、`gradient_lora/*.sh`）。
3. 合并 rank 梯度（由包装脚本调用 `merge_gradients.py`）。
4. 余弦比较（`gradient*/compute_sim.sh` 调 `compute_avg_grad_cosines.py`）。
5. 定理实验批量调度（`run_adaptive*.sh` 调 `theorem_experiment*.py`）。
6. 聚合实验结果（`aggregate*.py`）。

## 二、核心公式（关键脚本）

### 1) theorem_experiment 家族

- `delta = theta_end - theta_0`
- 近似二次型：
\[
v^THv\approx \frac{1}{n}\sum_i(v^Tg_i)^2
\]

- 指标：
\[
C_{start}=\frac12\cdot\frac1n\sum_i(\delta^Tg_i^{before})^2,
\quad
C_{end}=\frac12\cdot\frac1n\sum_i(\delta^Tg_i^{after})^2
\]
\[
\texttt{dot\_avggrad\_delta}=\langle\bar g_{after},\delta\rangle,
\quad V_{align}=-\texttt{dot\_avggrad\_delta}
\]

部分版本还输出 `dot_avggrad_delta_before` 与 `V_align_before`。

### 2) aggregate 家族

- `aggregate_results.py`：
\[
\texttt{total}=V_{align}+w_{start}C_{start}+C_{end}
\]

- `aggregate_results_modified.py`：
\[
\texttt{total}=w_{dot}\cdot\texttt{dot\_avggrad\_delta\_before}-\texttt{dot\_avggrad\_delta}+w_{start}C_{start}+C_{end}
\]

## 三、同类脚本的版本差异说明（重点）

下面这部分专门解释“看起来很像、但不能完全互换”的脚本族。

### 1) 常见后缀含义

| 后缀/命名 | 常见语义 | 典型变化点 |
|---|---|---|
| `_copy` | 同任务的替代启动器版本 | 常见变化是改为 `torchrun`/多卡调度，或作业编排与重试逻辑不同。 |
| `_1epoch` | 只跑 1 个 epoch 的版本 | 通常改到 `epoch1_*` 目录，并设置 `num_train_epochs=1`（日志更密）。 |
| `_base` | 仅 base model 的梯度分析版本 | `ADAPTER_DIR` 留空，脚本在分析路径中构造内存 LoRA。 |
| `_gsm8ktest` | 面向 gsm8k test 风格的梯度分析版本 | 主要改 split 与输出目录命名。 |
| `_modified` | 指标扩展版本 | 增加 `dot_avggrad_delta_before`、`V_align_before` 等字段。 |
| `_lasttry` | 严格输出路径版本 | 强制 `--output_path` 为文件路径，不允许隐式 fallback。 |
| `_random` | 多次随机重复并求均值版本 | 运行 `n_runs` 次并输出均值，可结合 DDP/磁盘流式梯度。 |

### 2) `theorem_experiment*.py` 细分对比

| 脚本 | warm-up 行为 | 输出策略 | 额外指标 | 并行/计算特性 |
|---|---|---|---|---|
| `theorem_experiment.py` | 执行 warm-up（默认 1 step）。 | `--output_path` 可传目录或文件。 | 基础字段：`delta_norm`、`dot_avggrad_delta`、`V_align`、`C_start`、`C_end`。 | 单次运行。 |
| `theorem_experiment_modified.py` | 与基础版 warm-up 逻辑一致。 | 与基础版一致（目录/文件都可）。 | 新增 `dot_avggrad_delta_before`、`V_align_before`。 | 单次运行。 |
| `theorem_experiment_lasttry.py` | 保留常量但走严格输出路径逻辑。 | 必须传文件级 `--output_path`；目录与非空文件覆盖都会报错。 | 包含 before/after 对齐相关字段。 | 单次运行，适合安全批调度。 |
| `theorem_experiment_random.py` | 每次 run 执行后再做多次均值聚合。 | 支持目录/文件并解析输出；管理任务临时目录 `tmp_dir`。 | 输出均值指标与 `n_runs`。 | 支持 DDP、梯度落盘流式处理、`--reuse_base_model`。 |

### 3) `run_adaptive.sh` vs `run_adaptive_copy.sh`

| 脚本 | 后端 Python | 调度方式 | 跳过/重试策略 | 输出目录 |
|---|---|---|---|---|
| `run_adaptive.sh` | `theorem_experiment_lasttry.py` | 按显存余量和槽位发任务（memory-aware）。 | 不做“已有有效结果自动跳过”。 | `experiment_results/` |
| `run_adaptive_copy.sh` | `theorem_experiment_random.py` | 每卡队列式调度（任务完成后补位）。 | 会检查 JSON 是否有效并自动跳过；任务失败会整体中止。 | `experiment_results_random/` |

### 4) `gradient/compute_sim.sh` 与 `gradient_lora/compute_sim.sh`

| 脚本 | 预期输入分析目录命名 | 对比项命名模式 |
|---|---|---|
| `gradient/compute_sim.sh` | `analysis/<dataset>/gradient_analysis_original_<method>/...` | `gradient_analysis_original_METHOD_*` |
| `gradient_lora/compute_sim.sh` | `analysis/<dataset>/gradient_analysis_<method>/...` | `gradient_analysis_METHOD_*` |

两者都调用 `compute_avg_grad_cosines.py`，核心差异是它们假定的上游目录命名约定不同。

### 5) 数据集目录中最容易混淆的脚本族

| 脚本族 | 共性 | 关键差异 |
|---|---|---|
| `sdft.sh` vs `sft.sh` | 下游评测任务族基本一致（math/openfunction/humaneval/alpaca_eval/safety）。 | `sdft.sh` 先做蒸馏数据生成（`*_distill` + `gen_distilled_data.py`）再训练；`sft.sh` 直接在 `<dataset>_train` 训练。 |
| `*_1epoch.sh` vs 非 `1epoch` | 流程图基本一致。 | 使用 `epoch1_predictions/`、`epoch1_results/`、`epoch1_checkpoints/`，并把训练轮数改为 1。 |
| `*_copy.sh` vs 非 `copy` | 目标流程一致。 | 常见差异是把 `python main.py` 改为 `torchrun` 多卡执行，并显式设置 NCCL/DDP 相关环境。 |
| `compute_gradient.sh` vs `compute_gradient_base.sh` | 都是 `analyze_gradients_llama_factory.py` + `merge_gradients.py`。 | 前者使用真实 adapter checkpoint，后者 adapter 留空做 base-only 分析。 |
| `compute_gradient_gsm8ktest.sh` | 仍是分析+合并流程。 | 主要用于 gsm8k-test 风格 split 的对比输出。 |
| `alpaca/test.sh` | 同样走分析+合并组件。 | 更偏调试：单进程单卡运行，通常步数更小，便于快速验证。 |

## 四、根目录脚本（`sdft/scripts` 直接子文件）

- `aggregate_and_fit.py`：大规模网格搜索拟合权重并输出聚合表。
- `aggregate_fit.py`：使用手工指定权重聚合（含数值差值表）。
- `aggregate_results.py`：基础聚合表输出。
- `aggregate_results_modified.py`：带 dot 项的改进聚合。
- `run_adaptive.sh`：按 checkpoint × 测试集批量跑 `theorem_experiment_lasttry.py`。
- `run_adaptive_copy.sh`：批量跑 `theorem_experiment_random.py`，并带“有效结果自动跳过”的队列调度逻辑。
- `run_all_token_stats_aligned.sh`：批量 token 统计。
- `test_seed_LM.sh`：seed 模型单卡评测流程。
- `test_seed_LM_copy.sh`：seed 模型多卡评测流程。
- `theorem_experiment.py`：带 warm-up 的定理实验版本。
- `theorem_experiment_lasttry.py`：严格 `--output_path` 的定理实验版本。
- `theorem_experiment_modified.py`：含 `*_before` 指标的 warm-up 版本。
- `theorem_experiment_random.py`：多次随机运行并聚合均值。
- `utils.sh`：工具函数（如 `create_empty_file`）。

## 五、`gradient/` 全部脚本说明

- `compute_sim.sh`：对“original”梯度分析结果做余弦比较。
- `sdft_alpaca_eval_gradient.sh`：sdft 在 alpaca_eval 的梯度分析。
- `sdft_gsm8k_testgradient.sh`：sdft 在 gsm8k_test 的梯度分析。
- `sdft_openfunction_testgradient.sh`：sdft 在 openfunction_test 的梯度分析。
- `sft_alpaca_eval_gradient.sh`：sft 在 alpaca_eval 的梯度分析。
- `sft_gsm8k_testgradient.sh`：sft 在 gsm8k_test 的梯度分析。
- `sft_openfunction_testgradient.sh`：sft 在 openfunction_test 的梯度分析。
- `train_sdft_gradient.sh`：遍历全部训练域做 original sdft 梯度分析。
- `train_sft_gradient.sh`：遍历全部训练域做 original sft 梯度分析。

共同依赖：`analyze_gradients_llama_factory.py` 与 `merge_gradients.py`，并需要对应模型/数据/checkpoint。

## 六、`gradient_lora/` 全部脚本说明

- `compute_sim.sh`：对 lora 梯度分析结果做余弦比较。
- `sdft_alpaca_eval_gradient.sh`
- `sdft_gsm8k_testgradient.sh`
- `sdft_openfunction_testgradient.sh`
- `sft_alpaca_eval_gradient.sh`
- `sft_gsm8k_testgradient.sh`
- `sft_openfunction_testgradient.sh`
- `train_sdft_gradient.sh`
- `train_sft_gradient.sh`

与 `gradient/` 的主要区别：路径约定更偏向 epoch1 LoRA checkpoint。

## 七、数据集子目录脚本（全部覆盖）

以下 7 个目录：
- `alpaca/`
- `dolly/`
- `gsm8k/`
- `lima/`
- `magicoder/`
- `openfunction/`
- `openhermes/`

每个目录中的脚本类型与作用一致：
- `compute_gradient.sh`：带 adapter 的梯度分析。
- `compute_gradient_base.sh`：base-only 梯度分析。
- `compute_gradient_gsm8ktest.sh`：面向 gsm8k test 形式的梯度分析。
- `sdft.sh`：蒸馏后训练 + 多任务评估流程。
- `sdft_1epoch.sh`：1 epoch 命名/路径版本（多为 torchrun）。
- `sdft_copy.sh`：同流程替代版本。
- `sft.sh`：直接 SFT 训练 + 评估流程。
- `sft_1epoch.sh`：1 epoch SFT 版本。
- （部分目录还有）`sft_copy.sh` 或 `test.sh`：替代/调试版本。

共同依赖：`main.py`、`eval/*` 脚本、对应数据文件、模型路径、checkpoint 路径。

## 八、运行顺序建议

- 若目标是 theorem 指标（`C_start/C_end/V_align`）：
  1. 先有 checkpoint；
  2. 跑 `run_adaptive*.sh`；
  3. 再跑 `aggregate*.py`。

- 若目标是 avg_grad 方向比较：
  1. 跑 `gradient/*` 或 `gradient_lora/*` 产出 `merged/avg_grad.pt`；
  2. 跑对应 `compute_sim.sh`。

- token 统计与 seed 评测可独立运行（满足输入条件即可）。
