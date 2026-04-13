import torch
import numpy as np
import os
import sys

# 如果你把文件放在了项目根目录，这两行可以注释掉；如果还在 src 里，保留这两行
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import Env
from src.model import Model
from src.search_sa import Search

def run_dry_test():
    print("🚀 [1/5] 初始化测试环境...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   -> 使用设备: {device}")

    # ==========================================
    # 1. 完美对齐你的老版本 YAML 配置
    # ==========================================
    env_params = {
        "num_processes": 1,
        "problem": "cvrp",
        "problem_size": 100,
        "num_nodes_to_remove": 15,    # 每次破坏 15 个点
        "recreate_n": 5,
        "beta": 0,
        "insert_in_new_tours_only": True,
        "generator_params": None,     # 因为有 PKL 文件，所以这里是 None
        "starting_solution_params": {
            "nb_iterations": 50,
            "nb_nodes_ratio": 0.15
        }
    }

    tester_params = {
        "aug_factor": 8,              # 注意！你老配置里增强因子是 8
        "use_cuda": True,
        "cuda_device_num": 0,
        "use_baseline_destroy": False,
        "model_load": [],
        "test_data_load": {
            "enable": True,
            "use_pkl_file": True,
            "filename": "data/cvrp/vrp100_test_seed1234.pkl"
        }
        # ⚠️ 注意：这里故意注释掉了 model_load！
        # 因为我们修改了25维输入，现在如果强行加载老权重 checkpoint-15.pt 必定报错。
        # Dry Run 阶段我们只用随机权重测试张量维度是否跑通！
    }
    
    # 必须保留完整的模型结构参数，否则 Model 初始化会报错
    model_params = {
        "problem": "cvrp",
        "problem_size": 100,
        "embedding_dim": 128,         
        "encoder_layer_num": 3,
        "head_num": 8,
        "qkv_dim": 16,
        "decoder_layer_num": 4, 
        "ff_hidden_dim": 512,
        
        # --- 根据源码找出的隐藏参数 ---
        "tour_layer": True,               # 源码是 if model_params["tour_layer"]:
        "message_passing_layer_num": 3,
        "encoder_layer_num_2": 3,
        "poly_embedding_dim": 128,
        "z_dim": 16,                      # 潜在向量维度
        "sqrt_embedding_dim": 128 ** 0.5, # 缩放因子，128的平方根 (约11.31)
        "logit_clipping": 10.0,           # Tanh 裁剪阈值
        "eval_type": "argmax"             # 评测时的解码策略
    }

    # ==========================================
    # 2. 实例化组件 (增加对 PKL 文件的智能读取)
    # ==========================================
    print("📦 [2/5] 实例化 环境与模型 (读取 PKL 数据)...")
    try:
        env = Env(**env_params)
        
        # --- 智能加载测试集逻辑 ---
        pkl_file = tester_params["test_data_load"]["filename"]
        if os.path.exists(pkl_file):
            print(f"   -> 成功找到测试集: {pkl_file}")
            # Dry Run 为了速度，只加载前 2 个图
            env.load_problem_dataset_pkl(pkl_file, num_problems=2) 
        else:
            print(f"   ⚠️ 警告: 找不到 {pkl_file}！将临时回退到随机生成模式...")
            env.problem.generator_params = {
                "use_X_generator": False, "rootPos": 0, "custPos": 0, "demandType": 0, "avgRouteSize": 5
            }
        
        # 初始化实例 (启用 aug_factor=8)
        env.init_instances(nb_instances=2, rollout_size=1, device=device, aug_factor=tester_params["aug_factor"])
        
        model = Model(**model_params).to(device)
        model.eval() 
    except Exception as e:
        print(f"❌ 实例化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==========================================
    # 3. 测试特征提取
    # ==========================================
    print("🧬 [3/5] 测试 25维特征 提取与拼接 (结合 Data Augmentation)...")
    try:
        reset_state = env.get_model_input(device)
        print("   -> 静态特征 Shape:", reset_state.l2seg_static_feats.shape)
        print("   -> 动态特征 Shape:", reset_state.l2seg_dynamic_feats.shape)
    except Exception as e:
        print(f"❌ 特征提取报错: {e}")
        import traceback
        traceback.print_exc()
        return

# ==========================================
    # 4. 测试网络前向传播
    # ==========================================
    print("🧠 [4/5] 测试 网络前向传播 (含深浅层交替 Masking)...")
    try:
        model.pre_forward(reset_state)
        
        # 👇 修改这里：使用你最新的方法名，并传入 starting_node=0 作为假定起点
        selected_nodes = model.generate_sequence_from_node(
            reset_state, 
            starting_node=0, 
            max_steps=env_params["num_nodes_to_remove"]
        )
        print("   -> 生成的破坏点 Shape:", selected_nodes.shape)
    except Exception as e:
        print(f"❌ 网络前向传播报错: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==========================================
    # 5. 测试 FSTA 压缩与 LKH-3 求解
    # ==========================================
    print("⚡ [5/5] 测试 FSTA 拓扑降维与 LKH-3 (处理 15 个不稳定点)...")
    try:
        search = Search(env_params, tester_params)
        
        from src.fsta_core import FSTA_Compressor
        
        depot_xy = reset_state.problem_feat.depot_xy[0].cpu().numpy()
        node_xy = reset_state.problem_feat.node_xy[0].cpu().numpy()

        # 确保 depot_xy 是二维的 (1, 2)，以便与 (100, 2) 的 node_xy 拼接
        if depot_xy.ndim == 1:
            depot_xy = np.expand_dims(depot_xy, axis=0)
            
        full_node_xy = np.concatenate((depot_xy, node_xy), axis=0)
        full_node_demand = np.concatenate(([0.0], reset_state.problem_feat.node_demand[0].cpu().numpy()))
        
        search.fsta_compressor = FSTA_Compressor(
            node_xy=full_node_xy,          # <--- 修复: 长度101的坐标
            node_demand=full_node_demand,  # <--- 修复: 长度101的需求
            capacity=50,
            lkh_path="./LKH-3.exe" 
        )
        
        
        # 取第一张图测试
        current_solution = env.instanceSet.get_solution(0) 
        tours_before = current_solution.getTourList()


        
        # ==========================================
        # 取出整个序列，并过滤掉车场(0)和虚拟结束符(>100)
        # ==========================================
        raw_ai_selected = selected_nodes[0].cpu().numpy().tolist()
        # 确保传给 FSTA 的只有 1~100 范围内的真实物理客户
        ai_selected = [x for x in raw_ai_selected if 0 < x <= env_params["problem_size"]]
        
        print(f"   -> AI 决定破坏以下节点: {ai_selected}")
        
        # 扔给 FSTA 压缩重构
        recovered_flat_tour = search.fsta_compressor.run_fsta_reoptimization(tours_before, set(ai_selected))
        
        print(f"   -> FSTA + LKH-3 完美融合！返回序列: {recovered_flat_tour[:20]}...")
    except Exception as e:
        print(f"❌ FSTA 或 LKH-3 报错: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n🎉🎉🎉 DRY RUN 测试完美通关！真实数据流水线彻底打通！ 🎉🎉🎉")

if __name__ == "__main__":
    with torch.no_grad(): 
        run_dry_test()