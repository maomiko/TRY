# L2Seg 论文复现（仓库实现版，CVRP100）

论文：`2507.01037v2.pdf`

## 1) 复现目标与成功标准

本仓库复现目标定义为：

1. 可稳定生成监督数据：`results/l2seg_dataset/l2seg_training_data.pt`
2. 可完成训练并输出 checkpoint：`results/l2seg_dataset/checkpoints/checkpoint-*.pt`
3. 可切换到 AI 模式推理并完成评测，输出 `results.csv / solutions.csv`
4. 产出 baseline（专家）与 AI（模型）两组 cost/runtime，可做对齐分析

## 2) 论文设定与当前仓库对齐说明

基于论文附录 D.3 / D.4，已在复现配置中对齐以下关键参数：

- `η (NAR threshold) = 0.6`
- `n_kmeans = 3`
- 训练数据生成迭代步：`TIS = 40`（在本仓库映射为 `nb_iterations=40`）
- 训练优化器：`ADAM`
- 小规模 CVRP 学习率：`1e-4`
- 训练 epoch：`200`
- 训练 batch size：`128`
- AR 序列权重：`wdelete=0.2`、`winsert=0.8`
- NAR 正样本权重：`wpos=9`

说明（对齐范围）：论文主实验以 1k/2k/5k 规模为主；本仓库复现实验固定为 **CVRP100 可运行链路**（`configs/reproduce/*.yaml`），目标是“关键机制与核心超参对齐 + 端到端流程可复现”。

### 2.1 论文参数 ↔ 仓库参数映射表（CVRP100 路径）

| 论文项 | 论文值 | 仓库映射 | 配置位置 |
|---|---:|---|---|
| NAR threshold `η` | 0.6 | `nar_threshold=0.6` | `label_gen_cvrp100.yaml` / `eval_ai_cvrp100.yaml` |
| `nKMEANS` | 3 | `n_kmeans=3` | `label_gen_cvrp100.yaml` / `eval_ai_cvrp100.yaml` |
| TIS | 40 | `nb_iterations=40` | `label_gen_cvrp100.yaml` / `eval_ai_cvrp100.yaml` |
| Optimizer | ADAM | `optim.Adam` | `train.py` |
| Epochs | 200 | `epochs=200` | `train_cvrp100.yaml` |
| Learning rate（small CVRP） | 1e-4 | `learning_rate=1e-4` | `train_cvrp100.yaml` |
| Batch size | 128 | `batch_size=128` | `train_cvrp100.yaml` |
| `wpos` | 9 | `nar_pos_weight=9` | `train_cvrp100.yaml` |
| `wdelete` | 0.2 | `ar_delete_weight=0.2` | `train_cvrp100.yaml` |
| `winsert` | 0.8 | `ar_insert_weight=0.8` | `train_cvrp100.yaml` |

时间限制说明：论文给出 1k/2k/5k 规模限时；本复现链路为 CVRP100，本仓库默认 `max_runtime=0`（不限时），以保证可稳定完成数据生成、训练与 AI 推理流程。

## 3) 一键分阶段复现实验

### 阶段 A：专家模式生成监督数据

```bash
python eval.py --config configs/reproduce/label_gen_cvrp100.yaml --seed 1234
```

说明：`eval.py` 默认配置已对齐为 `configs/reproduce/label_gen_cvrp100.yaml`。

LKH 说明（Linux/macOS）：

- 复现配置默认使用 `tester_params.lkh_path: "./LKH-3"`。
- 若根目录仅有 `LKH-3.0.14.tgz`，代码会自动尝试编译并生成 `./LKH-3`。
- 也可手动编译：

```bash
tar -xzf LKH-3.0.14.tgz
make -C LKH-3.0.14 -j
cp LKH-3.0.14/LKH ./LKH-3
chmod +x ./LKH-3
```

Windows 排查（LKH 崩溃 vs 数据参数崩溃）请参考：

- `WINDOWS_LKH_TROUBLESHOOTING.md`

期望输出：

- `results/l2seg_dataset/l2seg_training_data.pt`

### 阶段 B：训练 L2Seg 模型

```bash
python train.py --config configs/reproduce/train_cvrp100.yaml --seed 1234
```

期望输出：

- `results/l2seg_dataset/checkpoints/checkpoint-199.pt`（若 200 epochs）
- `results/l2seg_dataset/train_metrics.csv`

### 阶段 C：AI 模式推理评测

```bash
python eval.py --config configs/reproduce/eval_ai_cvrp100.yaml --seed 1234
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

## 6) 最小复现实例脚本（固定种子 + 单实例）

用于快速自动验证“不会出现 0.00 幽灵解，并且异常不会冒泡导致主流程崩溃”：

```bash
python repro_min_case.py \
  --config configs/reproduce/label_gen_cvrp100.yaml \
  --seed 1234 \
  --instance-idx 0 \
  --nb-iterations 5
```

脚本行为：

- 强制单实例、单进程、固定种子、CPU、基线破坏策略（不依赖模型 checkpoint）
- 运行一次最小 SA 求解流程
- 自动校验：
  - `cost` 为有限正数（拒绝 `0.00` 幽灵解）
  - 解不为空
  - 客户节点无重复、无缺失、全集覆盖正确

返回码约定：

- `0`：通过
- `1`：求解返回了无效/幽灵解
- `2`：脚本自身或运行流程崩溃
