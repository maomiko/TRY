import argparse
import copy
import os
import random
import re
import sys

import numpy as np
import torch
import yaml

from src.search_sa import Search


def load_config(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def main():
    parser = argparse.ArgumentParser(
        description="Standalone entry for L2Seg expert training-dataset generation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/reproduce/label_gen_cvrp100.yaml",
        help="Path to yaml config file",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    print(f"🚀 [1/3] 加载数据集生成配置: {args.config}")
    if not os.path.exists(args.config):
        print(f"❌ 错误: 找不到配置文件 {args.config}")
        return 1

    config = load_config(args.config)
    env_params = copy.deepcopy(config.get("env_params", {}))
    tester_params = copy.deepcopy(config.get("tester_params", {}))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tester_params.setdefault("seed", args.seed)
    tester_params["expert_data_mode"] = True

    if "num_processes" not in env_params:
        env_params["num_processes"] = 1
    env_params["model_params"] = config.get("model_params", {})
    try:
        _validate_preflight(env_params, tester_params)
    except Exception as e:
        print(f"❌ 配置预检失败: {e}")
        return 1

    print("⚙️ [2/3] 初始化 Search 引擎（独立数据集生成模式）")
    try:
        search = Search(env_params, tester_params)
    except Exception as e:
        print(f"❌ 引擎初始化失败: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("🔥 [3/3] 开始生成训练数据...")
    try:
        search.run()
        print("\n🎉 数据集生成结束！标签文件已保存。")
    except Exception as e:
        print(f"\n❌ 运行时崩溃: {e}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
