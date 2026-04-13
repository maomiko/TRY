import argparse
import yaml
import os
import torch
from src.search_sa import Search

def load_config(yaml_path):
    """加载 YAML 配置文件"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="L2Seg 25-Dim Evaluation & Data Collection Entry")
    # 允许通过命令行指定配置文件
    parser.add_argument('--config', type=str, default='configs/eval/cvrp100.yaml', help='Path to yaml config file')
    args = parser.parse_args()

    print(f"🚀 [1/3] 加载驱动配置: {args.config}")
    if not os.path.exists(args.config):
        print(f"❌ 错误: 找不到配置文件 {args.config}")
        return

    # 读取 YAML
    config = load_config(args.config)
    env_params = config.get("env_params", {})
    tester_params = config.get("tester_params", {})

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