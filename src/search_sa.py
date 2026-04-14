"""Simulated Annealing search for solving VRP instances one at a time."""

import os
import glob
import time
import random
import hashlib
import itertools
import copy
import csv
import pickle
from logging import getLogger
from typing import List, Dict, Any, Tuple, Optional, Set

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

MIN_SA_TEMPERATURE = 1e-12


class MockProblemFeat:
    def __init__(self, depot_xy, node_xy, node_demand):
        self.depot_xy = depot_xy.unsqueeze(0)       # 增加 Batch 维度 [1, 1, 2]
        self.node_xy = node_xy.unsqueeze(0)         # [1, 100, 2]
        self.node_demand = node_demand.unsqueeze(0) # [1, 100]

class MockState:
    def __init__(self, problem_feat, neighbours, tour_index):
        self.problem_feat = problem_feat
        self.neighbours = neighbours.unsqueeze(0) if neighbours.dim() == 2 else neighbours   # accepts [N, 2] or [B, N, 2]
        self.tour_index = tour_index.unsqueeze(0)   # [1, 100]

def create_l2seg_input(nn_d_xy, nn_n_xy, nn_n_dem, device, k=20):
    d_tensor = torch.tensor(nn_d_xy, dtype=torch.float32, device=device)
    n_tensor = torch.tensor(nn_n_xy, dtype=torch.float32, device=device)
    dem_tensor = torch.tensor(nn_n_dem, dtype=torch.float32, device=device)
    
    feat = MockProblemFeat(d_tensor, n_tensor, dem_tensor)
    tour_index = torch.zeros_like(dem_tensor, dtype=torch.long)
    
    # Original features require neighbours with shape [B, N, 2].
    batched_neighbours = torch.zeros((1, n_tensor.size(0), 2), dtype=torch.long, device=device)
    
    # Compute original L2Seg feature tensors.
    static_feats, dynamic_feats = compute_original_l2seg_features(
        feat.depot_xy, 
        feat.node_xy, 
        feat.node_demand, 
        tour_index.unsqueeze(0),
        batched_neighbours
    )
    
    state = MockState(feat, batched_neighbours, tour_index)
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

        # Reproducibility
        if "seed" in self.tester_params:
            self.seed = int(self.tester_params["seed"])
        elif "seed" in self.env_params:
            self.seed = int(self.env_params["seed"])
        else:
            self.seed = 1234
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        
        # Load trained models for learned destroy operations
        self.destroy_operators = self._load_destroy_operators()

    
        # 初始化 L2Seg 标签生成器
        self.label_generator = L2SegLabelGenerator(self.env_params["problem_size"])
        
        # 创建一个列表用于在内存中暂存生成的训练数据
        self.training_data_buffer = []

        self.test_dataset = None
        self.test_dataset_size = None

        self.lkh_path = self._resolve_lkh_path()

        
       

        

    def _resolve_lkh_path(self) -> str:
        """Resolve lkh_path with env expansion and safe fallback."""
        if "lkh_path" in self.tester_params:
            raw_path = self.tester_params["lkh_path"]
        elif "lkh_path" in self.env_params:
            raw_path = self.env_params["lkh_path"]
        else:
            raw_path = "./LKH-3"
        lkh_path = str(raw_path).strip()
        lkh_path = os.path.expanduser(os.path.expandvars(lkh_path))

        if os.path.exists(lkh_path):
            return lkh_path

        fallback_candidates = [os.path.abspath("./LKH-3"), os.path.abspath("./LKH-3.exe")]
        for fallback in fallback_candidates:
            if os.path.exists(fallback):
                self.logger.warning(
                    f"Configured lkh_path not found: {lkh_path}. Falling back to {fallback}. "
                    "If this is not expected, set tester_params.lkh_path to a valid executable path."
                )
                return fallback

        raise FileNotFoundError(
            f"LKH executable not found: configured={lkh_path}, "
            f"fallbacks={fallback_candidates}. "
            "Please set tester_params.lkh_path to a valid executable path."
        )

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
            checkpoint_path = self._resolve_checkpoint_path(model_config)
            if checkpoint_path is None:
                self.logger.warning(
                    "No checkpoint found for model config %s, skipping AI destroy operator.",
                    model_config,
                )
                continue

            checkpoint = self._load_checkpoint_compat(checkpoint_path)
            if checkpoint is None:
                self.logger.warning(
                    "Failed to load checkpoint %s, skipping AI destroy operator.",
                    checkpoint_path,
                )
                continue
            model_params = dict(checkpoint.get("model_params", {}))
            model_params.setdefault("problem", self.env_params.get("problem", "cvrp"))
            model_params.setdefault(
                "problem_size", int(self.env_params.get("problem_size", 100))
            )

            # Create and load model
            model = Model(**model_params).to(self.device)
            ckpt_state = checkpoint.get("model_state_dict", {})
            model_state = model.state_dict()
            compatible_state = {
                k: v
                for k, v in ckpt_state.items()
                if (k in model_state and model_state[k].shape == v.shape)
            }
            missing, unexpected = model.load_state_dict(
                compatible_state, strict=False
            )
            model.eval()
            if len(compatible_state) != len(ckpt_state):
                self.logger.warning(
                    "Checkpoint %s partially compatible (%d/%d tensors matched by shape).",
                    checkpoint_path,
                    len(compatible_state),
                    len(ckpt_state),
                )
            if len(missing) > 0 or len(unexpected) > 0:
                self.logger.warning(
                    "Checkpoint %s loaded with non-strict mode (missing=%d, unexpected=%d).",
                    checkpoint_path,
                    len(missing),
                    len(unexpected),
                )

            # Create seed vector sampler
            seed_sampler = SeedVectorSampler(int(model_params.get("z_dim", 16)), self.device)

            # Verify configuration matches
            checkpoint_remove = (
                checkpoint.get("env_params", {}).get("num_nodes_to_remove", None)
            )
            if checkpoint_remove is not None and checkpoint_remove != model_config.get(
                "node_to_remove"
            ):
                self.logger.warning(
                    "Model remove-count mismatch (%s vs %s), skipping operator %s.",
                    checkpoint_remove,
                    model_config.get("node_to_remove"),
                    checkpoint_path,
                )
                continue

            operators.append(
                {"model": model, "seed_sampler": seed_sampler, **model_config}
            )

            self.logger.info(f"Loaded deconstruction policy from {checkpoint_path}")

        if len(operators) == 0:
            self.logger.warning(
                "No valid AI destroy operator loaded; evaluation will fall back to baseline destroy."
            )
        return operators

    def _resolve_checkpoint_path(self, model_config: Dict[str, Any]) -> Optional[str]:
        """Resolve checkpoint path from config, with safe fallbacks."""
        base_path = str(model_config.get("path", ""))
        epoch = model_config.get("epoch", None)
        if base_path and epoch is not None:
            direct = os.path.join(base_path, f"checkpoint-{epoch}.pt")
            if os.path.exists(direct):
                return direct

        if base_path:
            discovered = sorted(glob.glob(os.path.join(base_path, "checkpoint-*.pt")))
            if len(discovered) > 0:
                return discovered[-1]

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        fallback = os.path.join(
            repo_root,
            "models",
            f"{self.env_params.get('problem', 'cvrp')}_{int(self.env_params.get('problem_size', 100))}",
            "checkpoint-2000.pt",
        )
        if os.path.exists(fallback):
            return fallback
        return None

    def _load_checkpoint_compat(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint with compatibility for newer torch defaults."""
        try:
            return torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
        except Exception as e:
            self.logger.warning(
                "weights_only=True failed for %s (%s), retrying with weights_only=False.",
                checkpoint_path,
                str(e).splitlines()[0] if str(e) else type(e).__name__,
            )
        try:
            return torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
        except Exception as e:
            self.logger.error(
                "Checkpoint load failed for %s: %s",
                checkpoint_path,
                str(e).splitlines()[0] if str(e) else type(e).__name__,
            )
            return None

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
        total_instances = self.tester_params.get("nb_instances", None)
        if total_instances is None:
            total_instances = self.test_dataset_size if self.test_dataset_size is not None else 1
        if self.test_dataset_size is not None:
            total_instances = min(total_instances, self.test_dataset_size)
        self.logger.info("=" * 80)
        self.logger.info(f"Starting search on {total_instances} instances")
        self.logger.info("=" * 80)

        # ======= 替换为多核并发代码 =======
        # 从配置中读取需要开启的进程数 (你在 yaml 里写的 6 或者 8)
        num_processes = self.env_params.get("num_processes", 1)
        
        self.logger.info("=" * 80)
        self.logger.info(f" 多核模式: {num_processes} 个 CPU 核心")
        self.logger.info("=" * 80)

        # Set tester_params.deterministic=True to force reproducible single-process execution.
        deterministic = self.tester_params.get("deterministic", False)
        if deterministic and num_processes > 1:
            # Deterministic replay requires a fixed execution order, so disable multi-process mode.
            self.logger.info("Deterministic mode enabled, forcing single-process execution.")
            num_processes = 1

        if num_processes > 1:
            self.logger.warning(
                "num_processes > 1 is not supported safely in current Search implementation "
                "because _solve_one_instance mutates shared instance state; forcing single-process mode."
            )
            num_processes = 1

        # 单核串行执行（避免共享状态竞态）
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

        save_path = self.tester_params.get(
            "l2s_data_save_path",
            os.path.join(self.result_folder, "l2seg_training_data.pt"),
        )
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(self.training_data_buffer, save_path)
        self.logger.info(f"成功保存 {len(self.training_data_buffer)} 条训练数据至 {save_path}")

    def _load_test_dataset(self) -> None:
        cfg = self.tester_params.get("test_data_load", {})
        if not cfg.get("enable", False):
            self.test_dataset = None
            self.test_dataset_size = None
            return

        filename = cfg.get("filename")
        if filename is None:
            raise ValueError("tester_params.test_data_load.filename is required when test_data_load.enable=True")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Test dataset file not found: {filename}")

        use_pkl = cfg.get("use_pkl_file", filename.endswith(".pkl"))
        if use_pkl:
            with open(filename, "rb") as f:
                self.test_dataset = pickle.load(f)
            self.test_dataset_size = len(self.test_dataset)
        else:
            loaded = torch.load(filename, map_location="cpu", weights_only=True)
            required = {"depot_xy", "node_xy", "node_demand", "capacity"}
            if not isinstance(loaded, dict) or not required.issubset(set(loaded.keys())):
                raise ValueError(f"Unsupported dataset format in {filename}; required keys: {sorted(required)}")
            self.test_dataset = loaded
            self.test_dataset_size = int(loaded["depot_xy"].shape[0])

        self.logger.info(f"Loaded test dataset from {filename} ({self.test_dataset_size} instances)")

    def _get_instance_raw_data(self, instance_idx: int):
        if self.test_dataset is None:
            filepath = self.tester_params.get("official_vrp_path", "./data/X-n101-k25.vrp")
            return load_cvrplib_instance(filepath)

        if isinstance(self.test_dataset, list):
            depot_xy, node_xy, node_demand, capacity = self.test_dataset[instance_idx]
            return (
                np.asarray(depot_xy, dtype=np.float32).reshape(1, 2),
                np.asarray(node_xy, dtype=np.float32),
                np.asarray(node_demand, dtype=np.float32),
                float(capacity),
            )

        depot_xy = self.test_dataset["depot_xy"][instance_idx].numpy().reshape(1, 2).astype(np.float32)
        node_xy = self.test_dataset["node_xy"][instance_idx].numpy().astype(np.float32)
        node_demand = self.test_dataset["node_demand"][instance_idx].numpy().astype(np.float32)
        capacity = float(self.test_dataset["capacity"][instance_idx].item())
        return depot_xy, node_xy, node_demand, capacity

    def _decode_and_validate_tours(
        self,
        flat_tour: List[int],
        num_customers: int,
        expected_customers: Optional[List[int]] = None,
    ) -> Optional[List[List[int]]]:
        if expected_customers is None:
            expected_set = set(range(1, num_customers + 1))
        else:
            expected_set = {
                int(n)
                for n in expected_customers
                if 0 < int(n) <= num_customers
            }
        if not expected_set:
            return None

        tours = []
        route = []
        seen = set()
        for raw in flat_tour:
            try:
                node = int(raw)
            except (TypeError, ValueError):
                continue

            if node <= 0 or node > num_customers:
                if route:
                    tours.append(route)
                    route = []
                continue

            if node in seen:
                return None
            seen.add(node)
            route.append(node)

        if route:
            tours.append(route)

        if seen != expected_set:
            return None
        return tours if tours else None

    @staticmethod
    def _flatten_tours(tours: List[List[int]]) -> List[int]:
        return [int(node) for route in tours for node in route]

    @staticmethod
    def _rebuild_tours_with_template(flat_nodes: List[int], template_tours: List[List[int]]) -> List[List[int]]:
        rebuilt = []
        cursor = 0
        for route in template_tours:
            route_len = len(route)
            rebuilt.append(flat_nodes[cursor : cursor + route_len])
            cursor += route_len
        return rebuilt

    def _route_cost(self, route: List[int]) -> float:
        if not route:
            return 0.0
        full = [0] + [int(n) for n in route] + [0]
        cost = 0.0
        for i in range(len(full) - 1):
            cost += float(np.linalg.norm(self.full_node_xy[full[i]] - self.full_node_xy[full[i + 1]]))
        return cost

    def _two_opt_improve_route(self, route: List[int], max_passes: int = 2) -> List[int]:
        if len(route) < 4:
            return list(route)
        best = list(route)
        best_cost = self._route_cost(best)

        for _ in range(max_passes):
            improved = False
            n = len(best)
            for i in range(0, n - 2):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    candidate = best[: i + 1] + best[i + 1 : j + 1][::-1] + best[j + 1 :]
                    candidate_cost = self._route_cost(candidate)
                    if candidate_cost + 1e-9 < best_cost:
                        best = candidate
                        best_cost = candidate_cost
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best

    def _python_repair_fallback(
        self,
        tours_before: List[List[int]],
        destroyed_nodes: Set[int],
        rng: random.Random,
    ) -> List[List[int]]:
        repaired = [list(route) for route in tours_before]
        changed = False

        for idx, route in enumerate(repaired):
            if not route:
                continue
            if destroyed_nodes and not any(int(n) in destroyed_nodes for n in route):
                continue
            improved_route = self._two_opt_improve_route(route)
            if improved_route != route:
                repaired[idx] = improved_route
                changed = True

        if changed:
            return repaired

        flat_nodes = self._flatten_tours(tours_before)
        if len(flat_nodes) < 2:
            return repaired

        candidate_positions = [i for i, node in enumerate(flat_nodes) if int(node) in destroyed_nodes]
        if len(candidate_positions) < 2:
            candidate_positions = list(range(len(flat_nodes)))

        i, j = rng.sample(candidate_positions, 2)
        flat_nodes[i], flat_nodes[j] = flat_nodes[j], flat_nodes[i]
        return self._rebuild_tours_with_template(flat_nodes, tours_before)

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
        expert_data_mode = bool(self.tester_params.get("expert_data_mode", False))

        start_time = time.time()

        # ==========================================
        # [纯 Python 重构] 1. 加载并拆分数据
        # ==========================================
        raw_d_xy, raw_n_xy, raw_n_dem, raw_capacity = self._get_instance_raw_data(instance_idx)
        num_customers = len(raw_n_xy)

        self.full_node_xy = np.concatenate((raw_d_xy, raw_n_xy), axis=0).astype(np.float32)

        # ==========================================
        # [纯 Python 重构] 2. 初始化 LKH-3
        # ==========================================
        lkh_node_xy = self.full_node_xy.astype(np.float32)
        lkh_demand = np.concatenate(([0], raw_n_dem)).astype(np.int64)
        lkh_timeout_sec = int(self.tester_params.get("lkh_timeout_sec", 60))
        lkh_path = self.lkh_path

        self.fsta_compressor = FSTA_Compressor(
            node_xy=lkh_node_xy,
            node_demand=lkh_demand,
            capacity=int(raw_capacity),
            lkh_path=lkh_path,
            timeout_sec=lkh_timeout_sec,
            max_vehicles=self.env_params.get("max_vehicles", 50)
        )

        # ==========================================
        # [纯 Python 重构] 3. 构建深度学习状态
        # ==========================================
        self.max_coord = np.max(self.full_node_xy)
        if self.max_coord <= 0:
            self.nn_node_xy = self.full_node_xy.copy()
        else:
            self.nn_node_xy = self.full_node_xy / self.max_coord
        demand_scale = raw_capacity if raw_capacity > 0 else 1.0
        self.nn_node_demand = raw_n_dem / demand_scale

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

        init_tours = self._decode_and_validate_tours(initial_flat_tour, num_customers)
        if init_tours is None:
            self.logger.warning(
                "LKH initial tour invalid/empty; falling back to naive tour for this instance."
            )
            init_tours = copy.deepcopy(naive_tour)
        
        # 将这个极好的初始解复制 aug_factor 份
        base_solution = PurePythonSolution(init_tours, self.full_node_xy)
        if not np.isfinite(base_solution.totalCosts):
            raise RuntimeError(f"Initial solution for instance {instance_idx} has non-finite cost.")
        if base_solution.totalCosts <= 0:
            raise RuntimeError(f"Initial solution for instance {instance_idx} has non-positive cost.")
        self.my_python_solutions = [copy.deepcopy(base_solution) for _ in range(aug_factor)]
        print(f"✅ 初始解生成完毕！初始 Cost: {base_solution.totalCosts:.2f}")

        # Track best solution
        incumbent_cost = base_solution.totalCosts
        incumbent_solution = copy.deepcopy(base_solution)

        # Simulated Annealing loop
        seed_input = f"{self.seed}:{int(instance_idx)}".encode("utf-8")
        local_seed = int(hashlib.sha256(seed_input).hexdigest()[:16], 16)
        local_rng = random.Random(local_seed)
        iteration = 0
        while iteration < max_iterations:

            if iteration % 5 == 0:
                loop_name = "专家迭代" if expert_data_mode else "SA 退火迭代"
                print(f" 实例 {instance_idx} | {loop_name}: {iteration}/{max_iterations} | 当前最佳 Cost: {incumbent_cost:.2f}")

            if expert_data_mode:
                # 论文 Algorithm 2 风格：R <- R+，独立于退火接受逻辑
                new_solutions = self._perform_sa_iteration(
                    aug_factor, rollout_size, sa_config["T"], local_rng
                )
                self.my_python_solutions = new_solutions

                for sol in new_solutions:
                    if sol.totalCosts < incumbent_cost:
                        incumbent_cost = sol.totalCosts
                        incumbent_solution = sol
            else:
                # Perform one SA iteration
                new_solutions = self._perform_sa_iteration(
                    aug_factor, rollout_size, sa_config["T"], local_rng
                )

                # SA 接受/拒绝：Metropolis 准则
                current_temp = max(float(sa_config["T"]), MIN_SA_TEMPERATURE)
                accepted_solutions = []
                for old_sol, new_sol in zip(self.my_python_solutions, new_solutions):
                    old_cost = old_sol.totalCosts
                    new_cost = new_sol.totalCosts
                    delta_cost = new_cost - old_cost

                    accept = False
                    if delta_cost <= 0:
                        accept = True
                    else:
                        accept_prob = float(np.exp(-delta_cost / current_temp))
                        accept = local_rng.random() < accept_prob

                    accepted_solutions.append(new_sol if accept else old_sol)

                # Update incumbent
                for sol in accepted_solutions:
                    if sol.totalCosts < incumbent_cost:
                        incumbent_cost = sol.totalCosts
                        incumbent_solution = sol

                # Synchronize augmented solutions (only when aug_factor > 1)
                if aug_factor > 1:
                    self._synchronize_augmented_solutions(
                        accepted_solutions, sa_config["T"], sa_config["delta"]
                    )

                # Update solutions in environment
                self.my_python_solutions = accepted_solutions

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

        return {
            "cost": incumbent_cost,
            "runtime": runtime,
            "nb_iterations": iteration,
            "solution": incumbent_solution,
        }

    def _should_accept_expert_label(self, improvement: float, rng: random.Random) -> bool:
        """Algorithm 2 gating: improvement threshold η_improv + stochastic acceptance α_AC."""
        min_improvement = float(self.tester_params.get("eta_improv", 0.0))
        if improvement < min_improvement:
            return False

        alpha_ac = float(self.tester_params.get("alpha_ac", 0.0))
        # 论文 D.3 中，small-capacity（如 CVRP/VRPTW 小容量设定）常配 α_AC=0。
        # Implementation: alpha_ac <= 0 means no extra random downsampling;
        # all labels passing eta_improv are retained.
        if alpha_ac <= 0.0:
            return True
        alpha_ac = min(alpha_ac, 1.0)
        return rng.random() <= alpha_ac

    def _collect_expert_training_labels(
        self,
        tours_before: List[List[int]],
        current_cost: float,
        new_solution: "PurePythonSolution",
        state_snapshot: Dict[str, torch.Tensor],
        rng: random.Random,
    ) -> None:
        improvement = float(current_cost - new_solution.totalCosts)
        if improvement <= 0:
            return

        subproblem_labels = self.label_generator.generate_labels(tours_before, new_solution.getTourList())
        for sub_label in subproblem_labels:
            if not self._should_accept_expert_label(improvement, rng):
                continue

            involved_nodes = sub_label["involved_nodes"]
            if len(involved_nodes) == 0:
                continue

            customer_nodes = []
            problem_size = int(self.env_params["problem_size"])
            for x in involved_nodes:
                x_int = int(x)
                if 0 < x_int <= problem_size:
                    customer_nodes.append(x_int)
            if len(customer_nodes) == 0:
                continue
            # customer ID is 1..N; feature tensors are 0..N-1.
            idx_tensor = torch.tensor([x - 1 for x in customer_nodes], dtype=torch.long)

            g2l_map = {g_id: l_idx + 1 for l_idx, g_id in enumerate(customer_nodes)}
            g2l_map[0] = 0

            raw_neighbours = state_snapshot["neighbours"][idx_tensor].tolist()
            local_neighbours = []
            for left, right in raw_neighbours:
                local_neighbours.append([g2l_map.get(left, 0), g2l_map.get(right, 0)])
            local_neighbours_tensor = torch.tensor(local_neighbours, dtype=torch.long)

            local_state_dict = {
                "depot_xy": state_snapshot["depot_xy"],
                "node_xy": state_snapshot["node_xy"][idx_tensor],
                "node_demand": state_snapshot["node_demand"][idx_tensor],
                "tour_index": state_snapshot["tour_index"][idx_tensor],
                "neighbours": local_neighbours_tensor,
                "global_node_indices": customer_nodes,
            }

            self.training_data_buffer.append(
                {
                    "nar_labels": sub_label["nar_labels"],
                    "ar_sequences": sub_label["ar_sequence"],
                    "state_dict": local_state_dict,
                }
            )

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
        self, aug_factor: int, rollout_size: int, temperature: float, rng: random.Random
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
                # 1. 摊平当前路径，找出所有真实的客户点 (排除车场 0)
                all_customers = [node for tour in current_solution.tours for node in tour if node != 0]
                
                # 2. 确定要删几个点 (通常是 15 个)
                num_to_remove = min(self.env_params["num_nodes_to_remove"], len(all_customers))
                
                # 3. 按照 rollout_size 生成选点列表
                selected_nodes = []
                for _ in range(rollout_size):
                    # 随机挑选 15 个点作为被破坏的节点
                    selected = rng.sample(all_customers, num_to_remove)
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

            expected_nodes = [n for t in tours_before for n in t if 0 < n <= num_customers]
            new_tours = self._decode_and_validate_tours(
                recovered_flat_tour,
                num_customers,
                expected_customers=expected_nodes,
            )
            if new_tours is None:
                self.logger.warning(
                    "Repaired tour invalid/empty; keeping previous feasible solution."
                )
                new_tours = copy.deepcopy(tours_before)

            if new_tours == tours_before:
                new_tours = self._python_repair_fallback(
                    tours_before=tours_before,
                    destroyed_nodes=set(flattened_nodes),
                    rng=rng,
                )

            # --- 更新解对象 ---
            # 这里非常关键：你需要使用你项目中更新路径的方法
            # 如果你使用的是 InstanceSet 维护的解，可能需要类似下面的操作：
            
            new_solution = PurePythonSolution(new_tours, self.full_node_xy)

            # ==========================================
            # 3. 生成论文 Algorithm 2 风格专家标签并存入 Buffer
            # ==========================================
            self._collect_expert_training_labels(
                tours_before=tours_before,
                current_cost=current_solution.totalCosts,
                new_solution=new_solution,
                state_snapshot=state_snapshot,
                rng=rng,
            )

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

        # 论文推荐超参数（支持通过 tester_params 覆盖）
        eta = float(self.tester_params.get("nar_threshold", 0.6))
        n_kmeans = int(self.tester_params.get("n_kmeans", 3))
        
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
            # unstable_candidates 来自 nar_probs：0 表示 depot，1..N 表示客户。
            # node_xy 是纯客户坐标数组，索引范围为 0..N-1，因此需减 1 做映射。
            candidate_customer_idx = unstable_candidates - 1
            candidate_coords = all_coords[candidate_customer_idx.cpu().numpy()]
            
            # 防止候选点数量比预设的 K 还少
            actual_k = min(n_kmeans, len(unstable_candidates))
            kmeans = KMeans(n_clusters=actual_k, random_state=self.seed, n_init='auto')
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
