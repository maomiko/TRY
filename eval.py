import argparse
import yaml
import os
import re
import sys
import torch
import random
import numpy as np
from src.search_sa import Search

def load_config(yaml_path):
    """加载 YAML 配置文件"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def _infer_problem_size_from_filename(path: str):
    match = re.search(r"vrp(\d+)", os.path.basename(str(path)).lower())
    return int(match.group(1)) if match else None


def _validate_preflight(env_params, tester_params):
    problem_size = int(env_params.get("problem_size", 0) or 0)
    if problem_size <= 0:
        raise ValueError("env_params.problem_size 必须是正整数。")

    lkh_path = str(tester_params.get("lkh_path", "")).strip()
    if lkh_path and not os.path.exists(lkh_path):
        raise FileNotFoundError(f"找不到 LKH 可执行文件: {lkh_path}")

    test_data_load = tester_params.get("test_data_load", {}) or {}
    if bool(test_data_load.get("enable", False)):
        filename = str(test_data_load.get("filename", "")).strip()
        if filename and not os.path.exists(filename):
            raise FileNotFoundError(f"找不到测试数据文件: {filename}")
        inferred = _infer_problem_size_from_filename(filename) if filename else None
        if inferred is not None and inferred != problem_size:
            raise ValueError(
                f"配置不一致: env_params.problem_size={problem_size}, "
                f"但 test_data_load.filename 指向 vrp{inferred} 数据。"
            )

    if not tester_params.get("use_baseline_destroy", True):
        model_load = tester_params.get("model_load", []) or []
        if not model_load:
            raise ValueError("AI 评测模式下 tester_params.model_load 不能为空。")
        first = model_load[0]
        ckpt_dir = str(first.get("path", "")).strip()
        epoch = first.get("epoch", None)
        if not ckpt_dir or epoch is None:
            raise ValueError("model_load[0] 需要同时包含 path 和 epoch。")
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint-{int(epoch)}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"找不到模型权重: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="L2Seg 25-Dim Pure Evaluation Entry")
    # 允许通过命令行指定配置文件
    parser.add_argument('--config', type=str, default='configs/reproduce/eval_ai_cvrp100.yaml', help='Path to yaml config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
    args = parser.parse_args()

    print(f"🚀 [1/3] 加载评测配置: {args.config}")
    if not os.path.exists(args.config):
        print(f"❌ 错误: 找不到配置文件 {args.config}")
        return 1

    # 读取 YAML
    config = load_config(args.config)
    env_params = config.get("env_params", {})
    tester_params = config.get("tester_params", {})

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tester_params.setdefault("seed", args.seed)
    if tester_params.get("expert_data_mode", False):
        print("⚠️ 检测到 expert_data_mode=true，eval.py 作为纯评测入口将强制关闭该模式。")
        tester_params["expert_data_mode"] = False

    # ==========================================
    # 👑 动态防御逻辑 (同步 dry_run 的成功经验)
    # ==========================================
    # 确保 C++ 算子加载时有基本的进程数配置
    if "num_processes" not in env_params:
        env_params["num_processes"] = 1
    
    # 注入模型参数：Search 类在实例化模型时需要根据 model_params 组装 25 维架构
    # 我们把 model_params 整体塞进 env_params，确保 Search 类能一站式读到
    env_params["model_params"] = config.get("model_params", {})
    try:
        _validate_preflight(env_params, tester_params)
    except Exception as e:
        print(f"❌ 配置预检失败: {e}")
        return 1

    print(f"⚙️ [2/3] 初始化 Search 引擎（纯评测模式，AI 模式: {not tester_params.get('use_baseline_destroy', True)}）")
    
    try:
        # Search 类现在会根据 YAML 驱动决定是去加载 LKH 专家还是 AI 权重
        search = Search(env_params, tester_params)
    except Exception as e:
        print(f"❌ 引擎初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("🔥 [3/3] 开始评测运行...")
    try:
        # 启动主循环，遍历数据集并记录 Cost/Runtime
        search.run()
        print("\n🎉 评测结束！结果已保存至 results 文件夹。")
    except Exception as e:
        print(f"\n❌ 运行时崩溃: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
