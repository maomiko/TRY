# TRY 操作手册（L2Seg CVRP100）

> 适用目录：`/home/runner/work/TRY/TRY`

## 1. 目标与入口

标准三阶段链路：

1. 阶段 A：生成训练标签数据  
   `generate_l2seg_training_data.py`
2. 阶段 B：训练模型  
   `train.py`
3. 阶段 C：AI 推理评测  
   `eval.py`

推荐配置：

- `configs/reproduce/label_gen_cvrp100.yaml`
- `configs/reproduce/train_cvrp100.yaml`
- `configs/reproduce/eval_ai_cvrp100.yaml`

---

## 2. 环境准备

```bash
cd /home/runner/work/TRY/TRY
python -m pip install --upgrade pip
python -m pip install pytest numpy torch pyyaml scikit-learn cppimport pybind11 networkx wandb
```

---

## 3. 冒烟验证（建议先跑）

### 3.1 生成临时 smoke 配置

```bash
mkdir -p /tmp/try-validation
python - <<'PY'
import yaml, pathlib
root=pathlib.Path('/home/runner/work/TRY/TRY')
out=pathlib.Path('/tmp/try-validation')

cfg=yaml.safe_load((root/'configs/reproduce/label_gen_cvrp100.yaml').read_text(encoding='utf-8'))
t=cfg['tester_params']
t['nb_instances']=2; t['nb_iterations']=2; t['rollout_size']=1
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
t['model_load']=[{'path': str(root/'results/l2seg_dataset/smoke_checkpoints'), 'epoch':0, 'node_to_remove':15}]
(out/'eval_ai_smoke.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
print('smoke configs ready')
PY
```

### 3.2 依次执行 A/B/C

```bash
python /home/runner/work/TRY/TRY/generate_l2seg_training_data.py --config /tmp/try-validation/label_smoke.yaml --seed 1234
python /home/runner/work/TRY/TRY/train.py --config /tmp/try-validation/train_smoke.yaml --seed 1234
python /home/runner/work/TRY/TRY/eval.py --config /tmp/try-validation/eval_ai_smoke.yaml --seed 1234
```

---

## 4. 正式运行（复现实验）

```bash
python /home/runner/work/TRY/TRY/generate_l2seg_training_data.py --config /home/runner/work/TRY/TRY/configs/reproduce/label_gen_cvrp100.yaml --seed 1234
python /home/runner/work/TRY/TRY/train.py --config /home/runner/work/TRY/TRY/configs/reproduce/train_cvrp100.yaml --seed 1234
python /home/runner/work/TRY/TRY/eval.py --config /home/runner/work/TRY/TRY/configs/reproduce/eval_ai_cvrp100.yaml --seed 1234
```

---

## 5. 产物检查

关键输出文件：

- `results/l2seg_dataset/l2seg_training_data.pt`
- `results/l2seg_dataset/checkpoints/checkpoint-*.pt`
- `results/l2seg_dataset/train_metrics.csv`
- `results/l2seg_dataset/results.csv`
- `results/l2seg_dataset/solutions.csv`

---

## 6. 各步骤代码层面拦截（失败前置）

### 阶段 A：`generate_l2seg_training_data.py`

- 配置文件存在性检查
- `env_params.problem_size` 正整数检查
- `tester_params.lkh_path` 存在性检查
- `test_data_load.enable=true` 时数据文件存在性检查
- 数据文件名 `vrpXXX` 与 `problem_size` 一致性检查

### 阶段 B：`train.py`

- 配置节缺失检查（`model_params`、`env_params`）
- 训练数据路径存在性检查
- 关键参数合法性检查（`epochs/batch_size/lr/grad_clip/checkpoint_every`）

### 阶段 C：`eval.py`

- 配置文件存在性检查
- `problem_size` / `lkh_path` / `test_data_load` 拦截同阶段 A
- AI 模式下 `model_load[0].path + epoch` 对应 checkpoint 存在性检查

> 三个入口脚本均在异常路径返回非 0 退出码，便于流水线快速失败。

---

## 7. 常见问题

1. **LKH 不存在**  
   检查 `tester_params.lkh_path`，Linux 推荐 `./LKH-3`。

2. **CVRP100/1000 混配**  
   确认 `problem_size` 与 `test_data_load.filename` 的 `vrpXXX` 一致。

3. **AI 评测找不到权重**  
   确认 `model_load[0]` 中 `path` 与 `epoch` 对应的 `checkpoint-{epoch}.pt` 存在。

---

## 8. 关联文档

- 复现说明：`REPRODUCE_L2SEG_CVRP100.md`
- 细化流程：`FULLFLOW_OPERATION_GUIDE.md`
- 文件作用说明：`FILE_ROLE_GUIDE.md`
- Windows LKH 排查：`WINDOWS_LKH_TROUBLESHOOTING.md`
