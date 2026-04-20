# TRY 文件作用文档

> 根目录：`/home/runner/work/TRY/TRY`

## 1. 顶层入口脚本

- `generate_l2seg_training_data.py`  
  阶段 A 入口：专家模式生成 L2Seg 训练数据（`.pt`）。

- `train.py`  
  阶段 B 入口：读取训练数据并训练 25 维模型，输出 checkpoint 与训练指标。

- `eval.py`  
  阶段 C 入口：加载 baseline 或 AI destroy，执行评测并写入结果。

- `repro_min_case.py`  
  最小复现脚本：单实例、固定种子，用于快速排查“崩溃/幽灵解”问题。

- `generate_dataset.py`  
  通用数据集生成工具（CVRP/VRPTW/PCVRP 随机数据）。

---

## 2. 配置文件（`configs/reproduce/`）

- `label_gen_cvrp100.yaml`  
  阶段 A 配置：专家数据生成参数。

- `train_cvrp100.yaml`  
  阶段 B 配置：训练超参、数据路径、checkpoint 路径。

- `eval_ai_cvrp100.yaml`  
  阶段 C 配置：AI 评测参数与模型加载路径。

---

## 3. 核心源码（`src/`）

- `search_sa.py`  
  主搜索引擎与 A/B/C 共用执行核心：实例加载、destroy/repair、结果写出、模型装载。

- `model.py`  
  L2Seg 模型实现（Encoder/Decoder + NAR/AR 分支）。

- `fsta_core.py`  
  FSTA 压缩与 LKH 调用封装（含 LKH 自动解析/构建、崩溃分类、fallback）。

- `expert_dataset_collector.py`  
  专家标签收集器，负责将中间步骤样本落盘为训练数据。

- `label_generator.py`  
  标签构造逻辑（NAR/AR 监督信号相关流程）。

- `utils.py`  
  通用工具函数（含 L2Seg 特征计算、数据装载辅助）。

- `trainer.py`  
  训练器实现（历史/通用训练流程），并包含 `l2s_collate_fn` 等训练辅助函数。

- `env.py`  
  环境与状态管理，供训练/搜索过程使用。

- `instance_set.py`  
  测试数据集加载与管理。

- `logging_utils.py`  
  日志目录、计时器、统计器等基础设施。

- `seed_sampler.py`  
  训练/推理时 seed vector 采样。

---

## 4. 问题与数据生成模块

- `problem.py` / `problem_cvrp.py` / `problem_vrptw.py` / `problem_pcvrp.py`  
  不同问题定义与约束处理。

- `generator_cvrp.py` / `generator_vrptw.py` / `generator_pcvrp.py`  
  对应问题的随机实例生成器。

- `validator.py`  
  解有效性校验。

---

## 5. 数据、结果与模型目录

- `data/`  
  输入数据集（如 `data/cvrp/vrp100_test_seed1234.pkl`）。

- `results/l2seg_dataset/`  
  主要输出目录（训练数据、checkpoint、metrics、results、solutions）。

- `models/`  
  模型相关资源目录（如已有预置文件时使用）。

---

## 6. 测试与文档

- `tests/test_l2seg_regression.py`  
  回归测试：AR token remap、loss 加权、starting node clamp 等关键行为。

- `FULLFLOW_OPERATION_GUIDE.md`  
  全流程执行说明（依赖安装与 smoke/full run）。

- `REPRODUCE_L2SEG_CVRP100.md`  
  论文复现目标、参数映射与结果对齐说明。

- `WINDOWS_LKH_TROUBLESHOOTING.md`  
  Windows 下 LKH 崩溃定位手册。

- `OPERATION_MANUAL.md`  
  本次新增的统一操作手册（面向执行与排错）。
