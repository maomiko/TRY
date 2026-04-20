import argparse
import copy
import os
import random

import numpy as np
import torch
import yaml

from src.search_sa import Search


def load_config(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
        return

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

    print("⚙️ [2/3] 初始化 Search 引擎（独立数据集生成模式）")
    try:
        search = Search(env_params, tester_params)
    except Exception as e:
        print(f"❌ 引擎初始化失败: {e}")
        import traceback

        traceback.print_exc()
        return

    print("🔥 [3/3] 开始生成训练数据...")
    try:
        search.run()
        print("\n🎉 数据集生成结束！标签文件已保存。")
    except Exception as e:
        print(f"\n❌ 运行时崩溃: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
