"""Iterative destroy-repair search for solving VRP instances one at a time."""

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
from .expert_dataset_collector import ExpertDatasetCollector

from .fsta_core import FSTA_Compressor

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
    """Iterative search that solves one VRP instance at a time."""

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

    
        # 专家标签数据集收集器（独立于求解流程）
        self.dataset_collector = ExpertDatasetCollector(
            env_params=self.env_params,
            tester_params=self.tester_params,
            result_folder=self.result_folder,
            logger=self.logger,
        )

        self.test_dataset = None
        self.test_dataset_size = None

        self.lkh_path = self._resolve_lkh_path()
        raw_iteration_log_interval = int(self.tester_params.get("iteration_log_interval", 5))
        if raw_iteration_log_interval < 0:
            self.logger.warning(
                "iteration_log_interval=%s is negative; clamped to 0 (disable per-iteration logs).",
                raw_iteration_log_interval,
            )
        self.iteration_log_interval = max(0, raw_iteration_log_interval)

        
       

        

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
        self.dataset_collector.save()

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

    def _build_cold_start_tours(
        self,
        num_customers: int,
        node_demand: np.ndarray,
        capacity: float,
        rng: random.Random,
    ) -> List[List[int]]:
        customers = list(range(1, num_customers + 1))
        rng.shuffle(customers)

        cap = float(capacity)
        tours: List[List[int]] = []
        current_route: List[int] = []
        current_load = 0.0

        for node in customers:
            demand_idx = node - 1
            demand = float(node_demand[demand_idx]) if 0 <= demand_idx < len(node_demand) else 0.0
            if current_route and (cap > 0) and (current_load + demand > cap + 1e-9):
                tours.append(current_route)
                current_route = []
                current_load = 0.0
            current_route.append(node)
            current_load += demand

        if current_route:
            tours.append(current_route)
        return tours if tours else [customers]

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
        Solve a single VRP instance using iterative destroy-repair with learned destroy operations.

        Returns dictionary with 'cost', 'runtime', 'nb_iterations', 'solution'.
        """
        aug_factor = self.tester_params["aug_factor"]
        max_iterations = self.tester_params["nb_iterations"]
        rollout_size = self.tester_params["rollout_size"]
        max_runtime = float(self.tester_params.get("max_runtime", 0))
        runtime_limited = max_runtime > 0

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
            max_vehicles=self.env_params.get("max_vehicles", 50),
            lkh_trace=bool(self.tester_params.get("lkh_trace", True)),
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

        seed_input = f"{self.seed}:{int(instance_idx)}".encode("utf-8")
        local_seed = int(hashlib.sha256(seed_input).hexdigest()[:16], 16)
        local_rng = random.Random(local_seed)

        # ==========================================
        # [纯 Python 重构] 4. Bootstrapping：可配置冷启动策略
        # ==========================================
        use_lkh_bootstrap = bool(
            self.tester_params.get(
                "bootstrap_with_lkh",
                self.tester_params.get("expert_data_mode", False),
            )
        )
        if use_lkh_bootstrap:
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
        else:
            if instance_idx == 0:
                self.logger.info("bootstrap_with_lkh=False，使用随机容量可行冷启动以便评测模型改进能力。")
            init_tours = self._build_cold_start_tours(
                num_customers=num_customers,
                node_demand=raw_n_dem,
                capacity=float(raw_capacity),
                rng=local_rng,
            )
        
        # 将这个极好的初始解复制 aug_factor 份
        base_solution = PurePythonSolution(init_tours, self.full_node_xy)
        if not np.isfinite(base_solution.totalCosts):
            raise RuntimeError(f"Initial solution for instance {instance_idx} has non-finite cost.")
        if base_solution.totalCosts <= 0:
            raise RuntimeError(f"Initial solution for instance {instance_idx} has non-positive cost.")
        self.my_python_solutions = [copy.deepcopy(base_solution) for _ in range(aug_factor)]
        self.logger.info(f"✅ 初始解生成完毕！初始 Cost: {base_solution.totalCosts:.2f}")

        # Track best solution
        incumbent_cost = base_solution.totalCosts
        incumbent_solution = copy.deepcopy(base_solution)

        # Iterative search loop
        iteration = 0
        while iteration < max_iterations:

            if self.iteration_log_interval > 0 and iteration % self.iteration_log_interval == 0:
                self.logger.info(
                    f" 实例 {instance_idx} | 专家迭代: {iteration}/{max_iterations} | 当前最佳 Cost: {incumbent_cost:.2f}"
                )

            # 论文 Algorithm 2 风格：R <- R+，不使用 SA 接受/降温逻辑
            samples_before = self.dataset_collector.sample_count
            new_solutions = self._perform_iteration(aug_factor, rollout_size, local_rng)
            samples_after = self.dataset_collector.sample_count
            newly_generated = samples_after - samples_before
            self.my_python_solutions = new_solutions

            for sol in new_solutions:
                if sol.totalCosts < incumbent_cost:
                    incumbent_cost = sol.totalCosts
                    incumbent_solution = sol

            iteration += 1
            self.logger.info(
                f" 实例 {instance_idx} | 第 {iteration}/{max_iterations} 轮生成样本: +{newly_generated} | 累计: {samples_after}"
            )

            # Check runtime limit
            if runtime_limited and (time.time() - start_time) > max_runtime:
                break

        runtime = time.time() - start_time

        return {
            "cost": incumbent_cost,
            "runtime": runtime,
            "nb_iterations": iteration,
            "solution": incumbent_solution,
        }

    def _collect_expert_training_labels(
        self,
        tours_before: List[List[int]],
        current_cost: float,
        new_solution: "PurePythonSolution",
        state_snapshot: Dict[str, torch.Tensor],
        rng: random.Random,
    ) -> None:
        self.dataset_collector.collect_from_transition(
            tours_before=tours_before,
            current_cost=current_cost,
            new_solution=new_solution,
            state_snapshot=state_snapshot,
            rng=rng,
        )

    def _perform_iteration(
        self, aug_factor: int, rollout_size: int, rng: random.Random
    ) -> List[Any]:
        use_ai_accelerator = (len(self.destroy_operators) > 0) and (not self.tester_params.get("use_baseline_destroy", True))

        # =======================================================
        # 👑 动态构建纯 Python 状态 (Tour Index & Neighbours)
        # =======================================================
        raw_tours = self.my_python_solutions[0].getTourList()
        num_customers = len(self.full_node_xy) - 1

        # 直接使用二维路线结构，避免拍平导致多车路线被错误合并。
        tours = [
            [
                int(node)
                for node in tour
                if 0 < int(node) <= num_customers
            ]
            for tour in raw_tours
            if tour
        ]

        # 👑 3. 提取特征
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

        
        # 直接使用传入的 reset_state，保护当前迭代现场不被破坏！

        # 论文推荐超参数（支持通过 tester_params 覆盖）
        eta = float(self.tester_params.get("nar_threshold", 0.6))
        n_kmeans = int(self.tester_params.get("n_kmeans", 3))
        operator_budget = operator.get("num_nodes_to_remove")
        if operator_budget is None:
            operator_budget = self.env_params.get("num_nodes_to_remove", 15)
        destroy_budget = max(1, int(operator_budget))
        
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
            
            # 获取每个节点被判定为不稳定的概率，并统一映射到全局客户 ID（1..N）。
            raw_probs = torch.sigmoid(nar_logits).reshape(-1)
            num_customers = int(reset_state.problem_feat.node_xy.size(1))
            if raw_probs.numel() == num_customers + 1:
                # 模型输出含 depot，跳过索引 0。
                customer_probs = raw_probs[1:]
            else:
                # 模型输出仅含客户。
                customer_probs = raw_probs[:num_customers]

            node_ids = torch.arange(
                1, customer_probs.numel() + 1, device=customer_probs.device, dtype=torch.long
            )

            # 找出所有大于等于阈值 η 的候选客户点
            unstable_mask = customer_probs >= eta
            unstable_candidates = node_ids[unstable_mask]

            # 若阈值过严导致无候选，则回退到 NAR Top-K（保证评测仍由模型驱动）。
            if unstable_candidates.numel() == 0:
                top_k = min(destroy_budget, customer_probs.numel())
                if top_k <= 0:
                    return [[] for _ in range(aug_factor)]
                _, top_idx = torch.topk(customer_probs, k=top_k)
                unstable_candidates = node_ids[top_idx]
                 
            # 阶段二：KMeans 聚类与寻找引爆点 (Clustering & Initial Node Identification)
            # 提取候选节点的真实二维坐标用于空间聚类
            all_coords = reset_state.problem_feat.node_xy.squeeze(0).cpu().numpy()
            # unstable_candidates 为全局客户 ID（1..N），node_xy 为客户数组（0..N-1），需减 1 映射。
            candidate_customer_idx = (unstable_candidates - 1).long()
            candidate_coords = all_coords[candidate_customer_idx.cpu().numpy()]
            
            # 防止候选点数量比预设的 K 还少
            actual_k = min(n_kmeans, len(unstable_candidates))
            kmeans = KMeans(n_clusters=actual_k, random_state=self.seed, n_init='auto')
            cluster_labels = kmeans.fit_predict(candidate_coords)
            
            initial_nodes = []
            candidate_probs = customer_probs[candidate_customer_idx].cpu().numpy()
            
            # 遍历每个簇，寻找 NAR 概率最高的那个节点作为 AR 的起点
            for k in range(actual_k):
                cluster_mask = (cluster_labels == k)
                best_local_idx = np.argmax(candidate_probs[cluster_mask])
                global_idx = unstable_candidates[cluster_mask][best_local_idx].item()
                initial_nodes.append(global_idx)
                
            # 阶段三：AR 局部精准爆破 (Local Unstable Edge Detection via AR)
            final_destroyed_nodes: List[int] = []
            seen_nodes = set()
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
                    node_int = int(node)
                    if 0 < node_int <= num_customers and node_int not in seen_nodes:
                        seen_nodes.add(node_int)
                        final_destroyed_nodes.append(node_int)
                        if len(final_destroyed_nodes) >= destroy_budget:
                            break
                if len(final_destroyed_nodes) >= destroy_budget:
                    break

            # AR 为空或不足预算时，使用 NAR 排序补齐预算（保持模型驱动）。
            if len(final_destroyed_nodes) < destroy_budget:
                ranked = torch.argsort(customer_probs, descending=True)
                for idx in ranked.tolist():
                    node_id = idx + 1
                    if node_id in seen_nodes:
                        continue
                    seen_nodes.add(node_id)
                    final_destroyed_nodes.append(node_id)
                    if len(final_destroyed_nodes) >= destroy_budget:
                        break

        # 把最终确定的不稳定节点列表，复制 aug_factor 份返回
        # 这样能无缝对齐你 _perform_sa_iteration 中 selected_nodes = all_selected_nodes[aug_idx] 的格式
        clean_selected_nodes = [list(final_destroyed_nodes) for _ in range(aug_factor)]

        return clean_selected_nodes
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
