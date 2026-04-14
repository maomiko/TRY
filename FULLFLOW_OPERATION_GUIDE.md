# TRY 全流程操作文档（依赖安装 + 验证）

## 1. 环境与目录

- 仓库目录：`/home/runner/work/TRY/TRY`
- Python：`3.12+`

先进入仓库：

```bash
cd /home/runner/work/TRY/TRY
```

## 2. 安装依赖

```bash
python -m pip install --upgrade pip
python -m pip install numpy networkx scikit-learn cppimport pybind11 pyyaml wandb torch
```

安装后可快速自检：

```bash
python - <<'PY'
import torch, yaml, numpy, sklearn, networkx, cppimport, wandb
print("deps ok")
PY
```

## 3. 基线检查（改动前/运行前）

```bash
python -m py_compile \
  /home/runner/work/TRY/TRY/eval.py \
  /home/runner/work/TRY/TRY/train.py \
  /home/runner/work/TRY/TRY/repro_min_case.py \
  /home/runner/work/TRY/TRY/src/search_sa.py
```

## 4. 全流程验证（推荐先跑 smoke）

### 4.1 生成 smoke 配置（临时，不入库）

```bash
mkdir -p /tmp/try-validation
python - <<'PY'
import yaml, pathlib
root=pathlib.Path('/home/runner/work/TRY/TRY')
out=pathlib.Path('/tmp/try-validation')

cfg=yaml.safe_load((root/'configs/reproduce/label_gen_cvrp100.yaml').read_text(encoding='utf-8'))
t=cfg['tester_params']
t['nb_instances']=2; t['nb_iterations']=2; t['rollout_size']=1
t['lkh_path']=str(root/'LKH-3')
t['l2s_data_save_path']=str(root/'results/l2seg_dataset/smoke_training_data.pt')
(out/'label_smoke.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')

cfg=yaml.safe_load((root/'configs/reproduce/train_cvrp100.yaml').read_text(encoding='utf-8'))
tr=cfg['trainer_params']
tr['train_data_path']=str(root/'results/l2seg_dataset/smoke_training_data.pt')
tr['checkpoint_dir']=str(root/'results/l2seg_dataset/smoke_checkpoints')
tr['metrics_csv']=str(root/'results/l2seg_dataset/smoke_train_metrics.csv')
tr['epochs']=1; tr['batch_size']=2; tr['checkpoint_every']=1
(out/'train_smoke.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')

cfg=yaml.safe_load((root/'configs/reproduce/eval_ai_cvrp100.yaml').read_text(encoding='utf-8'))
t=cfg['tester_params']
t['nb_instances']=2; t['nb_iterations']=2; t['rollout_size']=1
t['lkh_path']=str(root/'LKH-3')
t['model_load']=[{'path': str(root/'results/l2seg_dataset/smoke_checkpoints'), 'epoch':0, 'node_to_remove':15}]
(out/'eval_ai_smoke.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
print("smoke configs ready")
PY
```

### 4.2 阶段 A：生成训练数据

```bash
python /home/runner/work/TRY/TRY/eval.py --config /tmp/try-validation/label_smoke.yaml --seed 1234
```

### 4.3 阶段 B：训练模型

```bash
python /home/runner/work/TRY/TRY/train.py --config /tmp/try-validation/train_smoke.yaml --seed 1234
```

### 4.4 阶段 C：AI 推理评测

```bash
python /home/runner/work/TRY/TRY/eval.py --config /tmp/try-validation/eval_ai_smoke.yaml --seed 1234
```

## 5. 验证成功标准（smoke）

以下文件存在即代表链路打通：

- `/home/runner/work/TRY/TRY/results/l2seg_dataset/smoke_training_data.pt`
- `/home/runner/work/TRY/TRY/results/l2seg_dataset/smoke_checkpoints/checkpoint-0.pt`
- `/home/runner/work/TRY/TRY/results/l2seg_dataset/smoke_train_metrics.csv`
- `/home/runner/work/TRY/TRY/results/l2seg_dataset/results.csv`
- `/home/runner/work/TRY/TRY/results/l2seg_dataset/solutions.csv`

## 6. 正式全量运行（论文口径）

> 参数：`1000 cases × 25 iterations`，耗时显著高于 smoke。

```bash
python /home/runner/work/TRY/TRY/eval.py --config /home/runner/work/TRY/TRY/configs/reproduce/label_gen_cvrp100.yaml --seed 1234
python /home/runner/work/TRY/TRY/train.py --config /home/runner/work/TRY/TRY/configs/reproduce/train_cvrp100.yaml --seed 1234
python /home/runner/work/TRY/TRY/eval.py --config /home/runner/work/TRY/TRY/configs/reproduce/eval_ai_cvrp100.yaml --seed 1234
```

运行前请先把 YAML 中 `tester_params.lkh_path` 改成你机器上的可执行路径（Linux 建议 `/home/runner/work/TRY/TRY/LKH-3`）。
