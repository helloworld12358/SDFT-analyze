# Epoch0 QA-A最后Token 聚类实验说明

## 1. 目标
- 只使用 `epoch_0`（base模型，无LoRA）。
- 使用训练语料（不是测试集）做7类聚类分析。
- 每条样本输入为 `Q+A` 拼接文本。
- 每条样本提取 `A` 最后一个 token 的逐层表示。
- 比较 `SFT family` 与 `SDFT family` 的聚类质量差异。

## 2. 数据与类别
- family=`sft` 时，读取 `DataInf/data/<dataset>/<dataset>_train.json`
- family=`sdft` 时，读取 `DataInf/data/<dataset>/distilled_<dataset>.json`
- 默认7个类别：`gsm8k,openfunction,magicoder,alpaca,dolly,lima,openhermes`

## 3. 主要脚本
- `embedding_cluster_epoch0_qa_01_run_job.py`
  - 单个作业：一个 `family + seed`
  - 产出：sample pool、逐层高维指标、可选t-SNE
- `embedding_cluster_epoch0_qa_02_summary.py`
  - 汇总所有作业结果
  - 产出：跨seed/跨family聚合表、sdft-sft差值表、反例表、中文汇总md
- `run_embedding_cluster_epoch0_qa_answer_lasttok_single_node_4gpu.sh`
  - 单机4GPU调度器
  - 自动并行运行 `family × seed` 所有作业
  - 默认跳过已完成作业（可续跑）
- `embedding_cluster_epoch0_qa_02b_pick_shared_layers.py`
  - 从已有汇总中自动挑选“跨family同层可比”的投影层
  - 支持策略：`common_quality` / `delta_focus` / `balanced`
- `run_embedding_cluster_epoch0_qa_answer_lasttok_shared_tsne.sh`
  - 基于上一步挑选的共享层，重跑同层 t-SNE（SFT/SDFT 用同一批层）
  - 默认写到新目录，不覆盖旧结果

## 4. 输出目录（默认）
- `DataInf/results/embedding_cluster_epoch0_qa_answer_lasttok`

关键文件：
- `jobs/<family>/seed_<seed>/layer_metrics_<family>_seed<seed>.csv`
- `jobs/<family>/seed_<seed>/tsne/tsne_plot_*.png`
- `summary/layer_delta_sdft_minus_sft_by_seed.csv`
- `summary/layer_delta_sdft_minus_sft_agg.csv`
- `summary/counterexamples_orig_by_seed_layer.csv`
- `summary/epoch0_qa_answer_lasttok_summary_zh.md`

## 5. 推荐运行
见 `run_embedding_cluster_epoch0_qa_answer_lasttok_single_node_4gpu.sh`，常用环境变量：
- `EMBED_QA0_LOCAL_GPUS=0,1,2,3`
- `EMBED_QA0_SAMPLES_PER_CLASS=500`
- `EMBED_QA0_SEEDS=42,43,44`
- `EMBED_QA0_FAMILIES=sft,sdft`
- `EMBED_QA0_SKIP_DONE=1`
- `EMBED_QA0_DISABLE_TSNE=0`（若先快跑统计可设为1）

## 6. 同层可比 t-SNE（推荐用于对比图）
如果已经有一版全层统计结果，建议改用：
- `run_embedding_cluster_epoch0_qa_answer_lasttok_shared_tsne.sh`

示例（自动挑 3 层，共享于 SFT/SDFT）：
```bash
export EMBED_QA0_BASE_OUTPUT_ROOT=/inspire/hdd/project/.../DataInf/results/embedding_cluster_epoch0_qa_answer_lasttok
export EMBED_QA0_OUTPUT_ROOT=/inspire/hdd/project/.../DataInf/results/embedding_cluster_epoch0_qa_answer_lasttok_shared_tsne
export EMBED_QA0_SHARED_POLICY=common_quality
export EMBED_QA0_SHARED_TOP_K=3
bash DataInf/script/run_embedding_cluster_epoch0_qa_answer_lasttok_shared_tsne.sh
```

若你想固定层号（例如 `16,17,20`）：
```bash
export EMBED_QA0_SHARED_LAYERS=16,17,20
bash DataInf/script/run_embedding_cluster_epoch0_qa_answer_lasttok_shared_tsne.sh
```
