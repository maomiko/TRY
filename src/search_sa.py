"""Simulated Annealing search for solving VRP instances one at a time."""

import os
import time
import random
import itertools
import copy
import csv
from logging import getLogger
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

from .utils import compute_original_l2seg_features
from .utils import load_cvrplib_instance
from sklearn.cluster import KMeans

from .model import Model
from .logging_utils import (
    get_result_folder,
    TimeEstimator,
    AverageMeter,
)
from .seed_sampler import SeedVectorSampler
from .label_generator import L2SegLabelGenerator

from .fsta_core import FSTA_Compressor


class MockProblemFeat:
    def __init__(self, depot_xy, node_xy, node_demand):
        self.depot_xy = depot_xy.unsqueeze(0)       # 增加 Batch 维度 [1, 1, 2]
        self.node_xy = node_xy.unsqueeze(0)         # [1, 100, 2]
        self.node_demand = node_demand.unsqueeze(0) # [1, 100]

class MockState:
    def __init__(self, problem_feat, neighbours, tour_index):
        self.problem_feat = problem_feat
        self.neighbours = neighbours.unsqueeze(0)   # [1, 100, K]
        self.tour_index = tour_index.unsqueeze(0)   # [1, 100]

def create_l2seg_input(nn_d_xy, nn_n_xy, nn_n_dem, device, k=20):
    d_tensor = torch.tensor(nn_d_xy, dtype=torch.float32, device=device)
    n_tensor = torch.tensor(nn_n_xy, dtype=torch.float32, device=device)
    dem_tensor = torch.tensor(nn_n_dem, dtype=torch.float32, device=device)
    
    feat = MockProblemFeat(d_tensor, n_tensor, dem_tensor)
    tour_index = torch.zeros_like(dem_tensor, dtype=torch.long)
    
    # 修复：原版特征需要邻居形状严格为 [1, N, 2] (即前、后两个相连节点)
    neighbours = torch.zeros((1, n_tensor.size(0), 2), dtype=torch.long, device=device)
    
    # 👑 计算 100% 还原的论文特征
    static_feats, dynamic_feats = compute_original_l2seg_features(
        feat.depot_xy, 
        feat.node_xy, 
        feat.node_demand, 
        tour_index.unsqueeze(0),
        neighbours
    )
    
    state = MockState(feat, neighbours, tour_index)
    state.l2seg_static_feats = static_feats
    state.l2seg_dynamic_feats = dynamic_feats
    
    return state


class PurePythonSolution:
    def __init__(self, tours, full_node_xy):
        # 确保 tours 格式干净：去掉头尾多余的车场 0
        self.tours = [[n for n in tour if n != 0] for tour in tours if any(n != 0 for n in tour)]
        self.node_xy = full_node_xy
        self.totalCosts = self._calculate_cost()

    def _calculate_cost(self):
        """自己动手，丰衣足食：精准计算二维欧几里得距离作为 Cost"""
        cost = 0.0
        for tour in self.tours:
            if not tour: continue
            # 补齐起点和终点的车场 (0)
            full_tour = [0] + tour + [0]
            for i in range(len(full_tour) - 1):
                n1, n2 = full_tour[i], full_tour[i+1]
                # 计算两点之间的直线距离
                cost += np.linalg.norm(self.node_xy[n1] - self.node_xy[n2])
        return cost

    def getTourList(self):
        return copy.deepcopy(self.tours)


class Search:
    """Simulated Annealing search that solves one VRP instance at a time."""

    def __init__(
        self,
        env_params: Dict[str, Any],
        tester_params: Dict[str, Any],
    ):
        """Initialize search with configuration parameters."""
        self.env_params = env_params
        self.tester_params = tester_params

        # Setup logging
        self.logger = getLogger(name="tester")
        self.result_folder = get_result_folder()

        # --- 新增防错逻辑：新建一个专门的子文件夹来保存数据集 ---
        if "{desc}" in self.result_folder:
            # 专门新建一个名为 l2seg_dataset 的文件夹
            self.result_folder = "results/l2seg_dataset" 
        os.makedirs(self.result_folder, exist_ok=True)
        # -------------------------------------------------------------

        self.time_estimator = TimeEstimator()

        # Setup device
        self.device = self._setup_device()

        
        # Load trained models for learned destroy operations
        self.destroy_operators = self._load_destroy_operators()

    
        # 初始化 L2Seg 标签生成器
        self.label_generator = L2SegLabelGenerator(self.env_params["problem_size"])
        
        # 创建一个列表用于在内存中暂存生成的训练数据
        self.training_data_buffer = []

        
       

        

    def _setup_device(self) -> torch.device:
        """Setup and return the compute device (CPU or CUDA)."""
        use_cuda = self.tester_params["use_cuda"]
        if use_cuda:
            cuda_device_num = self.tester_params["cuda_device_num"]
            torch.cuda.set_device(cuda_device_num)
            return torch.device("cuda", cuda_device_num)
        return torch.device("cpu")

    

    def _load_destroy_operators(self) -> List[Dict[str, Any]]:
        """Load trained neural network models for destroy operations."""
        if self.tester_params["use_baseline_destroy"]:
            return []

        operators = []
        for model_config in self.tester_params["model_load"]:
            checkpoint_path = "{path}/checkpoint-{epoch}.pt".format(**model_config)
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            model_params = checkpoint["model_params"]

            # Create and load model
            model = Model(**model_params).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            # Create seed vector sampler
            seed_sampler = SeedVectorSampler(model_params["z_dim"], self.device)

            # Verify configuration matches
            assert (
                checkpoint["env_params"]["num_nodes_to_remove"]
                == model_config["node_to_remove"]
            ), f"Model trained with different num_nodes_to_remove: {checkpoint['env_params']['num_nodes_to_remove']} vs {model_config['node_to_remove']}"

            operators.append(
                {"model": model, "seed_sampler": seed_sampler, **model_config}
            )

            self.logger.info(f"Loaded deconstruction policy from {checkpoint_path}")

        return operators

    def run(self) -> None:
        """Run search on all test instances and save results."""
        self.time_estimator.reset()

        # Initialize metrics
        metrics = {
            "costs": AverageMeter(),
            "runtime": AverageMeter(),
            "iterations": AverageMeter(),
        }

        # Load test dataset if specified
        self._load_test_dataset()

        # Process test instances
        total_instances = self.tester_params.get("nb_instances", 1)
        self.logger.info("=" * 80)
        self.logger.info(f"Starting search on {total_instances} instances")
        self.logger.info("=" * 80)

        # ======= 替换为多核并发代码 =======
        # 从配置中读取需要开启的进程数 (你在 yaml 里写的 6 或者 8)
        num_processes = self.env_params.get("num_processes", 1)
        
        self.logger.info("=" * 80)
        self.logger.info(f" 多核模式: {num_processes} 个 CPU 核心")
        self.logger.info("=" * 80)

        if num_processes > 1:
            # 开启进程池并发
            with ThreadPoolExecutor(max_workers=num_processes) as executor:
                # 将所有图的任务同时扔进池子里
                futures = {
                    executor.submit(self._solve_one_instance, idx): idx 
                    for idx in range(total_instances)
                }
                
                # 哪个核心先跑完，就先处理哪个的结果
                for future in as_completed(futures):
                    instance_idx = futures[future]
                    try:
                        result = future.result()
                        # 更新指标
                        metrics["costs"].update(result["cost"], 1)
                        metrics["runtime"].update(result["runtime"], 1)
                        metrics["iterations"].update(result["nb_iterations"], 1)
                        
                        self._log_instance_progress(instance_idx, total_instances, result, metrics)
                        self._save_instance_results(instance_idx, result)
                    except Exception as e:
                        self.logger.error(f"实例 {instance_idx} 发生崩溃: {str(e)}")
        else:
            # 降级方案：如果是单核配置，依然串行跑
            for instance_idx in range(total_instances):
                result = self._solve_one_instance(instance_idx)
                metrics["costs"].update(result["cost"], 1)
                metrics["runtime"].update(result["runtime"], 1)
                metrics["iterations"].update(result["nb_iterations"], 1)
                self._log_instance_progress(instance_idx, total_instances, result, metrics)
                self._save_instance_results(instance_idx, result)

        # Log final summary
        self._log_final_summary(metrics)

        # 保存生成的标签数据
        self._save_training_labels()

    def _save_training_labels(self):
        """将收集到的 L2Seg 标签保存为 PyTorch 数据文件"""
        if not self.training_data_buffer:
            self.logger.info("未收集到任何有效的标签数据。")
            return
            
        save_path = os.path.join(self.result_folder, "l2seg_training_data.pt")
        torch.save(self.training_data_buffer, save_path)
        self.logger.info(f"成功保存 {len(self.training_data_buffer)} 条训练数据至 {save_path}")

    def _load_test_dataset(self) -> None:
        return

    def _solve_one_instance(self, instance_idx: int) -> Dict[str, Any]:
        """
        Solve a single VRP instance using Simulated Annealing with learned destroy operations.

        Returns dictionary with 'cost', 'runtime', 'nb_iterations', 'solution'.
        """
        aug_factor = self.tester_params["aug_factor"]
        max_iterations = self.tester_params["nb_iterations"]
        rollout_size = self.tester_params["rollout_size"]

        # Initialize SA parameters
        sa_config = self._init_simulated_annealing()

        start_time = time.time()

        # ==========================================
        # [纯 Python 重构] 1. 加载并拆分数据
        # ==========================================
        # 动态从 yaml 中读取路径，而不是写死！
        filepath = self.tester_params.get("official_vrp_path", "./data/X-n101-k25.vrp")
        
        raw_d_xy, raw_n_xy, raw_n_dem, raw_capacity = load_cvrplib_instance(filepath)
        num_customers = len(raw_n_xy)

        self.full_node_xy = np.concatenate((raw_d_xy, raw_n_xy), axis=0).astype(np.float32)

        # ==========================================
        # [纯 Python 重构] 2. 初始化 LKH-3
        # ==========================================
        lkh_node_xy = self.full_node_xy.astype(np.int64)
        lkh_demand = np.concatenate(([0], raw_n_dem)).astype(np.int64)

        self.fsta_compressor = FSTA_Compressor(
            node_xy=lkh_node_xy,
            node_demand=lkh_demand,
            capacity=int(raw_capacity),
            lkh_path="./LKH-3.exe" ,
            max_vehicles=self.env_params.get("max_vehicles", 50)
        )

        # ==========================================
        # [纯 Python 重构] 3. 构建深度学习状态
        # ==========================================
        self.max_coord = np.max(self.full_node_xy)
        self.nn_node_xy = self.full_node_xy / self.max_coord
        self.nn_node_demand = raw_n_dem / raw_capacity

        self.current_reset_state = create_l2seg_input(
            self.nn_node_xy[0:1], self.nn_node_xy[1:], self.nn_node_demand, self.device
        )
        self.cpu_reset_state = create_l2seg_input(
            self.nn_node_xy[0:1], self.nn_node_xy[1:], self.nn_node_demand, torch.device("cpu")
        )

        # ==========================================
        # [纯 Python 重构] 4. Bootstrapping：让 LKH-3 生成初始解
        # ==========================================
        # 我们不再依赖 C++ 给初始解，而是构造一个包含所有客户的列表，
        # 直接让 LKH-3 全局跑一次，给出一个极好的起点！
        naive_tour = [list(range(1, num_customers + 1))] 
        all_nodes_set = set(range(1, num_customers + 1))
        
        if instance_idx == 0:
            self.logger.info("正在为所有图生成 LKH 初始解")
        initial_flat_tour = self.fsta_compressor.run_fsta_reoptimization(naive_tour, all_nodes_set)
        
        # 提取初始路径
        init_tours = []
        temp_route = []
        for node in initial_flat_tour:
            if node == 0:
                if temp_route: init_tours.append(temp_route); temp_route = []
            else:
                temp_route.append(node)
        if temp_route: init_tours.append(temp_route)
        
        # 将这个极好的初始解复制 aug_factor 份
        base_solution = PurePythonSolution(init_tours, self.full_node_xy)
        self.my_python_solutions = [copy.deepcopy(base_solution) for _ in range(aug_factor)]
        print(f"✅ 初始解生成完毕！初始 Cost: {base_solution.totalCosts:.2f}")

        # Track best solution
        incumbent_cost = np.inf

        incumbent_solution = None

        # Simulated Annealing loop
        iteration = 0
        while iteration < max_iterations:
            
            if iteration % 5 == 0:
                print(f" 实例 {instance_idx} | SA 退火迭代: {iteration}/{max_iterations} | 当前最佳 Cost: {incumbent_cost:.2f}")
            
            # Perform one SA iteration
            new_solutions = self._perform_sa_iteration(
                aug_factor, rollout_size, sa_config["T"]
            )

            # Update incumbent
            for sol in new_solutions:
                if sol.totalCosts < incumbent_cost:
                    incumbent_cost = sol.totalCosts
                    incumbent_solution = sol

            # Synchronize augmented solutions (only when aug_factor > 1)
            if aug_factor > 1:
                self._synchronize_augmented_solutions(
                    new_solutions, sa_config["T"], sa_config["delta"]
                )

            # Update solutions in environment
            self.my_python_solutions = new_solutions

            # Update temperature
            sa_config["T"] = self._update_temperature(
                sa_config, iteration, start_time, max_iterations
            )

            iteration += 1

            # Check runtime limit
            if (
                sa_config["runtime_limited"]
                and (time.time() - start_time) > sa_config["max_runtime"]
            ):
                break

        runtime = time.time() - start_time

        # Verify runtime limit was enforced correctly
        if sa_config["runtime_limited"]:
            assert (
                runtime > sa_config["max_runtime"]
            ), "Runtime limit was set, but search terminated based on iteration count"

        return {
            "cost": incumbent_cost,
            "runtime": runtime,
            "nb_iterations": iteration,
            "solution": incumbent_solution,
        }

    def _init_simulated_annealing(self) -> Dict[str, Any]:
        """Initialize Simulated Annealing parameters."""
        max_runtime = self.tester_params["max_runtime"]
        runtime_limited = max_runtime > 0

        T_0 = self.tester_params["SA_start_T"]
        T_f = self.tester_params["SA_final_T"]

        config = {
            "T": T_0,
            "T_0": T_0,
            "T_f": T_f,
            "delta": self.tester_params["SA_delta"],
            "max_runtime": max_runtime,
            "runtime_limited": runtime_limited,
        }

        # Compute cooling rate if not runtime-limited
        if not runtime_limited:
            nb_iterations = self.tester_params["nb_iterations"]
            config["cooling_rate"] = (T_f / T_0) ** (1 / nb_iterations)

        return config

    def _perform_sa_iteration(
        self, aug_factor: int, rollout_size: int, temperature: float
    ) -> List[Any]:
        use_ai_accelerator = (len(self.destroy_operators) > 0) and (not self.tester_params.get("use_baseline_destroy", True))

        # =======================================================
        # 👑 动态构建纯 Python 状态 (Tour Index & Neighbours)
        # =======================================================
        raw_tours = self.my_python_solutions[0].getTourList()
        
        # 👑 1. 终极降维清洗：不管 raw_tours 被套娃了多少层，统统拍平！
        def flatten(lst):
            result = []
            for item in lst:
                if isinstance(item, list):
                    result.extend(flatten(item))
                else:
                    result.append(item)
            return result
            
        flat_tour = flatten(raw_tours)
        
        num_customers = len(self.full_node_xy) - 1
        
        # 👑 2. 洗牌重组：剔除 0 和 LKH 产生的假车场，还原出完美的 2D 路线！
        clean_tours = []
        current_route = []
        for node in flat_tour:
            if node == 0 or node > num_customers:
                # 遇到车场，切割出一条独立的车队路线
                if len(current_route) > 0:
                    clean_tours.append(current_route)
                    current_route = []
            else:
                current_route.append(node)
        if len(current_route) > 0:
            clean_tours.append(current_route)
            
        # 将绝对纯净的 List[List[int]] 赋值回去
        tours = clean_tours
        
        # 👑 3. 提取特征 (这下绝对不会再有 list - 1 的报错了)
        tour_idx_array = np.zeros(num_customers, dtype=np.int64)
        neighbours_array = np.zeros((num_customers, 2), dtype=np.int64)

        for t_idx, tour in enumerate(tours):
            full_tour = [0] + tour + [0] # 补齐起终点车场
            for i in range(1, len(full_tour) - 1):
                node = full_tour[i]
                left_node = full_tour[i-1]
                right_node = full_tour[i+1]
                
                neighbours_array[node - 1, 0] = left_node
                neighbours_array[node - 1, 1] = right_node
                tour_idx_array[node - 1] = t_idx + 1

        # 转换为张量
        t_tensor = torch.tensor(tour_idx_array, dtype=torch.long, device=self.device)
        nb_tensor = torch.tensor(neighbours_array, dtype=torch.long, device=self.device)
        
        t_tensor_cpu = torch.tensor(tour_idx_array, dtype=torch.long, device=torch.device("cpu"))
        nb_tensor_cpu = torch.tensor(neighbours_array, dtype=torch.long, device=torch.device("cpu"))

        # 覆盖更新全局的状态 (使用 unsqueeze 满足 MockState 的批次维度要求)
        self.current_reset_state.tour_index = t_tensor.unsqueeze(0)
        self.current_reset_state.neighbours = nb_tensor.unsqueeze(0)
        
        self.cpu_reset_state.tour_index = t_tensor_cpu.unsqueeze(0)
        self.cpu_reset_state.neighbours = nb_tensor_cpu.unsqueeze(0)

        # ==========================================
        # AI 选点
        # ==========================================
        all_ai_selected_nodes = None
        if use_ai_accelerator:
            all_ai_selected_nodes = self._select_nodes_with_model(aug_factor, self.current_reset_state)

        # ==========================================
        # 1. 提取环境特征快照备份 (转回 CPU 供标签提取用)
        # ==========================================
        state_snapshot = {
            "depot_xy": self.cpu_reset_state.problem_feat.depot_xy.reshape(-1, 2),  # 强制拍平 [1, 2]
            "node_xy": self.cpu_reset_state.problem_feat.node_xy.reshape(-1, 2),    # 强制拍平 [N, 2]
            "node_demand": self.cpu_reset_state.problem_feat.node_demand.reshape(-1), # 强制拍平 [N]
            "tour_index": self.cpu_reset_state.tour_index.reshape(-1),              # 强制拍平 [N]
            "neighbours": self.cpu_reset_state.neighbours.reshape(-1, 2),           # 终极修复：强制拍平为 [N, 2]！
        }

        # 初始化用于返回的空列表
        new_solutions = []
        # ==========================================
        # 2. 执行 Destroy-Repair 循环
        # ==========================================
        for aug_idx in range(aug_factor):
            current_solution = self.my_python_solutions[aug_idx]

            # 提取优化前的路径
            tours_before = current_solution.getTourList()

            # 决定被破坏的节点
            # =======================================================
            # 【修改点 2】：正式切换选点决策权 (体力活 vs 脑力活)
            # =======================================================
            if use_ai_accelerator and all_ai_selected_nodes is not None:
                # 路径 A [AI加速组]：瞬间拿到 GPU 刚才推理出的“不稳定节点”列表
                selected_nodes = all_ai_selected_nodes[aug_idx]
            else:
                # ==========================================
                # 👑 [纯 Python 选点基准]：彻底甩掉 C++，自己写启发式破坏！
                # ==========================================
                import random
                # 1. 摊平当前路径，找出所有真实的客户点 (排除车场 0)
                all_customers = [node for tour in current_solution.tours for node in tour if node != 0]
                
                # 2. 确定要删几个点 (通常是 15 个)
                num_to_remove = min(self.env_params["num_nodes_to_remove"], len(all_customers))
                
                # 3. 按照 rollout_size 生成选点列表
                selected_nodes = []
                for _ in range(rollout_size):
                    # 随机挑选 15 个点作为被破坏的节点
                    selected = random.sample(all_customers, num_to_remove)
                    selected_nodes.append(selected)

            # 调用我们在 fsta_core.py 中定义的压缩与求解逻辑
            if torch.is_tensor(selected_nodes):
                nodes_array = selected_nodes.cpu().detach().numpy().flatten()
            else:
                nodes_array = np.array(selected_nodes).flatten()

            # 过滤非法点并转为 int 列表，确保 set() 可哈希
            flattened_nodes = [int(x) for x in nodes_array if 0 < x <= self.env_params["problem_size"]]

            recovered_flat_tour = self.fsta_compressor.run_fsta_reoptimization(
                tours_before, 
                set(flattened_nodes)
            )
            

            # --- 将平铺数组重新切分为二维路径列表 ---
            new_tours = []
            temp_route = []
            for node in recovered_flat_tour:
                if node == 0:
                    if temp_route:
                        new_tours.append(temp_route)
                        temp_route = []
                else:
                    temp_route.append(node)
            # 处理最后一个未闭合的路径
            if temp_route:
                new_tours.append(temp_route)

            # --- 更新解对象 ---
            # 这里非常关键：你需要使用你项目中更新路径的方法
            # 如果你使用的是 InstanceSet 维护的解，可能需要类似下面的操作：
            
            # ==========================================
            # 👑 [终极修复：通过 C++ 绑定的 Getter 方法提取 Instance]
            # ==========================================
            tours_list = current_solution.getTourList()
            
            
            new_solution = PurePythonSolution(new_tours, self.full_node_xy)


            
            
            # 提取优化后的路径
            tours_after = new_solution.getTourList()

            # ==========================================
            # 3. 生成标签并打包存入 Buffer
            # ==========================================
            # 只有当路径成本实质性下降时，才认为这是一次成功的专家操作
            if current_solution.totalCosts > new_solution.totalCosts:
                # 名字没变！直接调用，但返回的是一个包含了多个局部标签的列表
                subproblem_labels = self.label_generator.generate_labels(
                    tours_before, tours_after
                )
                
                # 遍历这一次迭代产生的“多个”局部修补操作
                for sub_label in subproblem_labels:
                    involved_nodes = sub_label["involved_nodes"]
                    
                    if len(involved_nodes) == 0:
                        continue
                        
                    # 【核心修复】：分离 Depot，把 1~100 的客户 ID 转为 0~99 的张量索引
                    customer_nodes = [x for x in involved_nodes if x != 0]
                    # 把 involved_nodes 转换为 tensor，方便对 state_snapshot 进行切片
                    idx_tensor = torch.tensor([x - 1 for x in customer_nodes], dtype=torch.long)


                    # === 新增：将全局 neighbours 映射为局部 neighbours ===
                    # 建立映射表，Depot 依然是 0
                    g2l_map = {g_id: l_idx + 1 for l_idx, g_id in enumerate(customer_nodes)}
                    g2l_map[0] = 0  
                    
                    raw_neighbours = state_snapshot["neighbours"][idx_tensor].tolist()
                    local_neighbours = []
                    for left, right in raw_neighbours:
                        # 如果邻居不在这个切片里，就当它连着车场 (0)
                        local_neighbours.append([g2l_map.get(left, 0), g2l_map.get(right, 0)])
                    local_neighbours_tensor = torch.tensor(local_neighbours, dtype=torch.long)
                    
                    # 【核心】：从全图快照中，只切出这两条路线的特征！
                    local_state_dict = {
                        "depot_xy": state_snapshot["depot_xy"], # 车场坐标不变
                        "node_xy": state_snapshot["node_xy"][idx_tensor], 
                        "node_demand": state_snapshot["node_demand"][idx_tensor],
                        "tour_index": state_snapshot["tour_index"][idx_tensor],
                        "neighbours": local_neighbours_tensor, # <--- 加回邻居
                        "global_node_indices": customer_nodes # 注意：这里只存客户 ID 了！  
                    }

                    # 把这个“微型子问题”存入 buffer
                    self.training_data_buffer.append({
                        "nar_labels": sub_label["nar_labels"],       # 局部 NAR 标签
                        "ar_sequences": sub_label["ar_sequence"],    # 局部 AR 序列
                        "state_dict": local_state_dict               # 纯局部的特征快照
                    })

            new_solutions.append(new_solution)

        return new_solutions

    def _select_nodes_with_model(
        self, aug_factor: int, reset_state
    ) -> List[List[int]]:
        """
        L2Seg-SYN 协同推理逻辑：NAR 全局圈定 -> KMeans 聚类 -> AR 局部深挖
        严格复刻论文 Algorithm 3
        """
        operator = random.choice(self.destroy_operators)
        model = operator["model"]

        
        # 直接使用传入的 reset_state，保护 SA 现场不被破坏！

        # 论文推荐的超参数
        eta = 0.6            # NAR 判定不稳定的阈值
        n_kmeans = 3         # 聚类簇的数量
        
        with torch.no_grad():

            static_feats, dynamic_feats = compute_original_l2seg_features(
                reset_state.problem_feat.depot_xy,
                reset_state.problem_feat.node_xy,
                reset_state.problem_feat.node_demand,
                reset_state.tour_index,
                reset_state.neighbours,
                pad_mask=None # 推理时是全局图，没有 padding
            )
            reset_state.l2seg_static_feats = static_feats
            reset_state.l2seg_dynamic_feats = dynamic_feats



            # 阶段一：NAR 全局探路 (Global Unstable Node Detection)
            # 注意：此处假设你的 model 内部已实现 pre_forward 和 nar_forward
            model.pre_forward(reset_state)
            nar_logits = model.nar_forward()
            
            # 获取每个节点被判定为不稳定的概率 (假设提取单图概率)
            nar_probs = torch.sigmoid(nar_logits).squeeze(0) 
            nar_probs[0] = -1e9 # 永远不破坏车场 (Depot)
            
            # 找出所有大于等于阈值 η 的候选节点
            unstable_candidates = torch.where(nar_probs >= eta)[0]
            
            # 如果当前图极其稳定，直接返回空列表
            if len(unstable_candidates) == 0:
                # 返回 aug_factor 份空列表，以对齐外层循环
                return [[] for _ in range(aug_factor)]
                
            # 阶段二：KMeans 聚类与寻找引爆点 (Clustering & Initial Node Identification)
            # 提取候选节点的真实二维坐标用于空间聚类
            all_coords = reset_state.problem_feat.node_xy.squeeze(0).cpu().numpy()
            candidate_coords = all_coords[unstable_candidates.cpu().numpy()]
            
            # 防止候选点数量比预设的 K 还少
            actual_k = min(n_kmeans, len(unstable_candidates))
            kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(candidate_coords)
            
            initial_nodes = []
            candidate_probs = nar_probs[unstable_candidates].cpu().numpy()
            
            # 遍历每个簇，寻找 NAR 概率最高的那个节点作为 AR 的起点
            for k in range(actual_k):
                cluster_mask = (cluster_labels == k)
                best_local_idx = np.argmax(candidate_probs[cluster_mask])
                global_idx = unstable_candidates[cluster_mask][best_local_idx].item()
                initial_nodes.append(global_idx)
                
            # 阶段三：AR 局部精准爆破 (Local Unstable Edge Detection via AR)
            final_destroyed_nodes = set()
            end_token_idx = model.vocab_size - 1
            
            for init_node in initial_nodes:
                with torch.amp.autocast(device_type=self.device.type):
                    # 【核心接力】：将 NAR 找出的 init_node 传递给 AR 作为解码起点
                    seq_tensor = model.generate_sequence_from_node(reset_state, starting_node=init_node)
                
                seq_np = seq_tensor.cpu().numpy().flatten()
                
                # 遇到 EOS 就截断，动态决定破坏规模
                for node in seq_np:
                    if node == end_token_idx:
                        break
                    if node != 0:
                        final_destroyed_nodes.add(node)

        # 把最终确定的不稳定节点列表，复制 aug_factor 份返回
        # 这样能无缝对齐你 _perform_sa_iteration 中 selected_nodes = all_selected_nodes[aug_idx] 的格式
        clean_selected_nodes = [list(final_destroyed_nodes) for _ in range(aug_factor)]

        return clean_selected_nodes
    def _synchronize_augmented_solutions(
        self, solutions: List[Any], temperature: float, delta: float
    ) -> None:
        """
        Synchronize solutions across augmentations by replacing poor solutions
        with good candidates (within temperature threshold).
        """
        costs = np.array([sol.totalCosts for sol in solutions])
        min_cost = np.min(costs)

        # Find candidate solutions (within threshold)
        threshold = min_cost + temperature * delta
        candidate_indices = np.where(costs < threshold)[0]

        if len(candidate_indices) == 0:
            return

        # Replace solutions that are too expensive
        for idx in range(len(solutions)):
            if costs[idx] > threshold:
                replacement_idx = np.random.choice(candidate_indices)
                solutions[idx] = solutions[replacement_idx]

    def _update_temperature(
        self,
        sa_config: Dict[str, Any],
        iteration: int,
        start_time: float,
        max_iterations: int,
    ) -> float:
        """Update and return new temperature for Simulated Annealing."""
        if sa_config["runtime_limited"]:
            # Runtime-based cooling schedule
            elapsed = time.time() - start_time
            max_runtime = sa_config["max_runtime"]
            progress = min(1.0, elapsed / max_runtime)
            T = sa_config["T_f"] * (sa_config["T_0"] / sa_config["T_f"]) ** (
                1 - progress
            )
        else:
            # Iteration-based geometric cooling
            T = sa_config["T"] * sa_config["cooling_rate"]

        return T

    def _log_instance_progress(
        self,
        instance_idx: int,
        total: int,
        result: Dict[str, Any],
        metrics: Dict[str, AverageMeter],
    ) -> None:
        """Log progress for current instance."""
        elapsed, remaining = self.time_estimator.get_est_string(instance_idx + 1, total)
        self.logger.info(
            f"Instance {instance_idx + 1:3d}/{total:3d}  |  "
            f"Elapsed: {elapsed}  |  Remain: {remaining}  |  "
            f"Cost: {result['cost']:7.2f}  |  Avg: {metrics['costs'].avg:7.3f}  |  "
            f"Iters: {result['nb_iterations']:4.0f}"
        )

    def _log_final_summary(self, metrics: Dict[str, AverageMeter]) -> None:
        """Log final summary statistics."""
        self.logger.info("=" * 80)
        self.logger.info("Search Complete")
        self.logger.info("=" * 80)
        self.logger.info(f"Average Cost:       {metrics['costs'].avg:7.3f}")
        self.logger.info(f"Average Runtime:    {metrics['runtime'].avg:7.2f} seconds")
        self.logger.info(f"Average Iterations: {metrics['iterations'].avg:7.1f}")
        self.logger.info("=" * 80)

    def _save_instance_results(self, instance_idx: int, result: Dict[str, Any]) -> None:
        """Save instance results to CSV files."""
        # Save cost and runtime statistics
        results_path = os.path.join(self.result_folder, "results.csv")
        with open(results_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    instance_idx + 1,
                    result["cost"],
                    result["runtime"],
                    result["nb_iterations"],
                ]
            )

        # Save solution tours
        if result["solution"] is not None:
            solutions_path = os.path.join(self.result_folder, "solutions.csv")
            tours = result["solution"].getTourList()
            # Add depot (node 0) at start and end of each tour
            tours_with_depot = [[0, *tour, 0] for tour in tours]

            with open(solutions_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([instance_idx + 1, tours_with_depot])
