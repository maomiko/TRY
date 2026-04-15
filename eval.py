import argparse
import yaml
import os
import torch
import random
import numpy as np
from src.search_sa import Search


def _normalize_optional_path(path_value):
    if not path_value:
        return None
    return os.path.normcase(os.path.normpath(os.path.abspath(os.path.expandvars(os.path.expanduser(str(path_value))))))


def _startup_check_data_splits(config, tester_params):
    split_meta = config.get("data_splits", {})
    train_file = (split_meta.get("train", {}) if isinstance(split_meta.get("train", {}), dict) else {}).get("filename")
    valid_file = (split_meta.get("valid", {}) if isinstance(split_meta.get("valid", {}), dict) else {}).get("filename")
    test_file_meta = (split_meta.get("test", {}) if isinstance(split_meta.get("test", {}), dict) else {}).get("filename")

    cfg_data_file = (tester_params.get("test_data_load", {}) if isinstance(tester_params.get("test_data_load", {}), dict) else {}).get("filename")
    if tester_params.get("expert_data_mode", False):
        train_file = cfg_data_file or train_file
    else:
        test_file_meta = cfg_data_file or test_file_meta

    train_seed = (split_meta.get("train", {}) if isinstance(split_meta.get("train", {}), dict) else {}).get("seed")
    valid_seed = (split_meta.get("valid", {}) if isinstance(split_meta.get("valid", {}), dict) else {}).get("seed")
    test_seed = (split_meta.get("test", {}) if isinstance(split_meta.get("test", {}), dict) else {}).get("seed")

    print("🔎 数据切分检查（train/valid/test）:")
    print(f"   train: {train_file} (seed={train_seed})")
    print(f"   valid: {valid_file} (seed={valid_seed})")
    print(f"   test : {test_file_meta} (seed={test_seed})")

    if train_file and valid_file and test_file_meta:
        basenames = [os.path.basename(str(train_file)), os.path.basename(str(valid_file)), os.path.basename(str(test_file_meta))]
        if len(set(basenames)) != 3:
            raise ValueError(
                f"train/valid/test 文件名存在重名: {basenames}；请使用互斥数据集。"
            )

        normalized = [_normalize_optional_path(train_file), _normalize_optional_path(valid_file), _normalize_optional_path(test_file_meta)]
        if len(set(normalized)) != 3:
            raise ValueError(
                "train/valid/test 文件路径存在重叠（同一路径），请改为互斥数据源。"
            )

def load_config(yaml_path):
    """加载 YAML 配置文件"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="L2Seg 25-Dim Evaluation & Data Collection Entry")
    # 允许通过命令行指定配置文件
    parser.add_argument('--config', type=str, default='configs/reproduce/label_gen_cvrp100.yaml', help='Path to yaml config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
    args = parser.parse_args()

    print(f"🚀 [1/3] 加载驱动配置: {args.config}")
    if not os.path.exists(args.config):
        print(f"❌ 错误: 找不到配置文件 {args.config}")
        return

    # 读取 YAML
    config = load_config(args.config)
    env_params = config.get("env_params", {})
    tester_params = config.get("tester_params", {})
    _startup_check_data_splits(config, tester_params)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tester_params.setdefault("seed", args.seed)

    # ==========================================
    # 👑 动态防御逻辑 (同步 dry_run 的成功经验)
    # ==========================================
    # 确保 C++ 算子加载时有基本的进程数配置
    if "num_processes" not in env_params:
        env_params["num_processes"] = 1
    
    # 注入模型参数：Search 类在实例化模型时需要根据 model_params 组装 25 维架构
    # 我们把 model_params 整体塞进 env_params，确保 Search 类能一站式读到
    env_params["model_params"] = config.get("model_params", {})

    print(f"⚙️ [2/3] 初始化 Search 引擎 (AI 模式: {not tester_params.get('use_baseline_destroy', True)})")
    
    try:
        # Search 类现在会根据 YAML 驱动决定是去加载 LKH 专家还是 AI 权重
        search = Search(env_params, tester_params)
    except Exception as e:
        print(f"❌ 引擎初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("🔥 [3/3] 开始正式运行...")
    try:
        # 启动主循环，遍历数据集并记录 Cost/Runtime
        search.run()
        print("\n🎉 运行结束！结果已保存至 results 文件夹。")
    except Exception as e:
        print(f"\n❌ 运行时崩溃: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
