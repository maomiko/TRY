# L2Seg 论文复现（仓库实现版，CVRP100）

论文：`/home/runner/work/TRY/TRY/2507.01037v2.pdf`

## 1) 复现目标与成功标准

本仓库复现目标定义为：

1. 可稳定生成监督数据：`results/l2seg_dataset/l2seg_training_data.pt`
2. 可完成训练并输出 checkpoint：`results/l2seg_dataset/checkpoints/checkpoint-*.pt`
3. 可切换到 AI 模式推理并完成评测，输出 `results.csv / solutions.csv`
4. 产出 baseline（专家）与 AI（模型）两组 cost/runtime，可做对齐分析

## 2) 论文设定与当前仓库对齐说明

基于论文附录 D.3 / D.4，已在复现配置中对齐以下关键参数：

- `η (NAR threshold) = 0.6`
- `nKMEANS = 3`
- 训练数据生成迭代步：`TIS = 40`（在本仓库映射为 `nb_iterations=40`）
- 训练优化器：`ADAM`
- 小规模 CVRP 学习率：`1e-4`
- 训练 epoch：`200`

说明：论文主实验以 1k/2k/5k 规模为主；本仓库已有稳定的 CVRP100 数据与实现，因此复现流程采用 CVRP100 实现路径，目标是“流程与关键机制可复现”。

## 3) 一键分阶段复现实验

### 阶段 A：专家模式生成监督数据

```bash
python /home/runner/work/TRY/TRY/eval.py --config /home/runner/work/TRY/TRY/configs/reproduce/label_gen_cvrp100.yaml --seed 1234
```

期望输出：

- `results/l2seg_dataset/l2seg_training_data.pt`

### 阶段 B：训练 L2Seg 模型

```bash
python /home/runner/work/TRY/TRY/train.py --config /home/runner/work/TRY/TRY/configs/reproduce/train_cvrp100.yaml --seed 1234
```

期望输出：

- `results/l2seg_dataset/checkpoints/checkpoint-199.pt`（若 200 epochs）
- `results/l2seg_dataset/train_metrics.csv`

### 阶段 C：AI 模式推理评测

```bash
python /home/runner/work/TRY/TRY/eval.py --config /home/runner/work/TRY/TRY/configs/reproduce/eval_ai_cvrp100.yaml --seed 1234
```

期望输出：

- `results/l2seg_dataset/results.csv`
- `results/l2seg_dataset/solutions.csv`

## 4) 结果对齐模板（手工填写）

| 模式 | 数据集 | 平均 Cost | 平均 Runtime(s) | 备注 |
|---|---|---:|---:|---|
| Baseline Destroy | CVRP100 test |  |  | label_gen 配置 |
| L2Seg AI Destroy | CVRP100 test |  |  | eval_ai 配置 |

建议同时记录：

- 随机种子
- checkpoint epoch
- 配置文件路径
- GPU/CPU 环境

## 5) 本次实现改动摘要

1. 新增复现配置目录：`configs/reproduce/`
   - `label_gen_cvrp100.yaml`
   - `train_cvrp100.yaml`
   - `eval_ai_cvrp100.yaml`
2. 训练入口 `train.py` 配置化，去除关键硬编码（数据路径、epoch、lr、checkpoint 路径、loss 权重等）
3. `search_sa.py` 支持通过配置覆盖 `nar_threshold` 和 `n_kmeans`，并支持自定义训练数据保存路径

