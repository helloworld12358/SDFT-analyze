# Embedding 聚类实验（t-SNE 版）

本实验目标：
- 先用 `epoch_5` 的 `sft/sdft` 模型做逐层扫描，找出 **SDFT 与 SFT 聚类差异最明显** 的层；
- 再对每个训练集分片，绘制故事线模型（`epoch_1/epoch_5` 的 `sft/sdft` + 可选 `base`）在选定层上的 2D t-SNE 图；
- 每张图默认 500 个点（5 个测试集 × 每集 100 条问题）。

## 新增脚本

- `embedding_cluster_01_epoch5_layer_scan.py`
  - 阶段 1：逐层扫描（默认全层）
  - 产出：`stage1/by_train_dataset/*/epoch5_layer_scan_*.csv`
- `embedding_cluster_02_select_layers.py`
  - 阶段 2：合并阶段 1 结果并选层
  - 产出：`stage2/recommended_layers.json`
- `embedding_cluster_03_plot_selected_layers_tsne.py`
  - 阶段 3：按选中层画 t-SNE（默认读阶段 2 的推荐层）
  - 产出：`stage3/by_train_dataset/*/tsne_plot_*.png`

## 运行入口（bash）

- `run_embedding_cluster_stage1_scan.sh`
- `run_embedding_cluster_stage2_select_layers.sh`
- `run_embedding_cluster_stage3_tsne.sh`
- `run_embedding_cluster_pipeline.sh`（按 `EMBED_CLUSTER_PHASE` 调度）
- `run_embedding_cluster_7nodes_shared_auto.sh`（7 台共享存储全自动总控，推荐）

## 建议执行顺序

1. 阶段 1：7 台机器并行分片（按训练集）
2. 阶段 2：任意 1 台机器汇总选层
3. 阶段 3：7 台机器并行分片（按训练集）

## 7 台机器并行（推荐）

公共环境变量（每台都执行）：

```bash
cd /inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/SDFT-analysis
export EMBED_CLUSTER_DATAINF_ROOT=/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/SDFT-analysis/DataInf
export EMBED_CLUSTER_PYTHON=/opt/conda/bin/python
export EMBED_CLUSTER_DEVICE=auto
export EMBED_CLUSTER_SAMPLES_PER_TASK=100
export EMBED_CLUSTER_SEED=42
export EMBED_CLUSTER_BATCH_SIZE=8
```

### 阶段 1（7 台并行）

每台机器只改 `EMBED_CLUSTER_TRAIN_DATASET`：

```bash
export EMBED_CLUSTER_TRAIN_DATASET=alpaca   # 每台换成不同训练集
bash DataInf/script/run_embedding_cluster_stage1_scan.sh
```

7 个训练集：
- `alpaca`
- `dolly`
- `gsm8k`
- `lima`
- `magicoder`
- `openfunction`
- `openhermes`

### 阶段 2（1 台汇总）

```bash
unset EMBED_CLUSTER_TRAIN_DATASET
export EMBED_CLUSTER_TOP_K_LAYERS=3
bash DataInf/script/run_embedding_cluster_stage2_select_layers.sh
```

### 阶段 3（7 台并行）

每台机器只改 `EMBED_CLUSTER_TRAIN_DATASET`：

```bash
export EMBED_CLUSTER_TRAIN_DATASET=alpaca   # 每台换成不同训练集
export EMBED_CLUSTER_INCLUDE_BASE=1
bash DataInf/script/run_embedding_cluster_stage3_tsne.sh
```

## 一次挂起到跑完（7台共享存储自动协同）

这个模式下，每台机器都跑同一个脚本，只改 `EMBED_CLUSTER_WORKER_ID` 即可。  
脚本会自动做：
- stage1 分片跑
- 等待其他机器
- 仅一台机器自动执行 stage2
- 等待 stage2 完成
- stage3 分片跑
- 最终等待全部完成

每台机器执行（示例 worker 0）：

```bash
cd /inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/SDFT-analysis
export EMBED_CLUSTER_DATAINF_ROOT=/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/SDFT-analysis/DataInf
export EMBED_CLUSTER_PYTHON=/opt/conda/bin/python
export EMBED_CLUSTER_DEVICE=auto
export EMBED_CLUSTER_WORKER_ID=0   # 0~6，每台不同
export EMBED_CLUSTER_NUM_WORKERS=7
bash DataInf/script/run_embedding_cluster_7nodes_shared_auto.sh
```

建议 7 台对应：
- 机器1: `EMBED_CLUSTER_WORKER_ID=0`
- 机器2: `EMBED_CLUSTER_WORKER_ID=1`
- 机器3: `EMBED_CLUSTER_WORKER_ID=2`
- 机器4: `EMBED_CLUSTER_WORKER_ID=3`
- 机器5: `EMBED_CLUSTER_WORKER_ID=4`
- 机器6: `EMBED_CLUSTER_WORKER_ID=5`
- 机器7: `EMBED_CLUSTER_WORKER_ID=6`

协同状态文件在：
- `DataInf/results/embedding_cluster/_coord/`

## 重要参数

- `EMBED_CLUSTER_DEVICE=auto`
  - 推荐值。会优先用 `device_map=auto`，允许多卡切分模型。
- `EMBED_CLUSTER_LAYERS`
  - 阶段 1 使用，默认 `all`
- `EMBED_CLUSTER_PLOT_LAYERS`
  - 阶段 3 可手动覆盖推荐层，例如 `8,16,24`
- `EMBED_CLUSTER_TOP_K_LAYERS`
  - 阶段 2 选层数，阶段 3（未手工指定层时）读取同数量推荐层
- `EMBED_CLUSTER_MAX_LENGTH`
  - 默认 `1024`。如果你想更保守防 OOM，可调小。

## 输出目录

默认输出根目录：
- `DataInf/results/embedding_cluster/`

关键文件：
- `stage1/epoch5_layer_scan_all.csv`
- `stage2/layer_rank_summary.csv`
- `stage2/recommended_layers.json`
- `stage3/story_tsne_summary_all.csv`
- `stage3/by_train_dataset/<train_dataset>/tsne_plot_*.png`
