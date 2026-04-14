"""Environment for VRP problems."""

from dataclasses import dataclass
from typing import Tuple, Optional, Any

import numpy as np
import torch

from .problem_cvrp import ProblemCVRP
from .problem_vrptw import ProblemVRPTW
from .problem_pcvrp import ProblemPCVRP
from .instance_set import InstanceSet


@dataclass
class ResetState:
    """State returned when environment is reset."""

    problem_feat: Any = None
    tour_index: torch.Tensor = None
    neighbours: torch.Tensor = None
    # --- 新增 L2Seg 增强特征 ---
    l2seg_static_feats: torch.Tensor = None  # 角度、KNN距离等静态特征
    l2seg_dynamic_feats: torch.Tensor = None # K近邻在同路径的比例

@dataclass
class StepState:
    """State maintained during environment stepping."""

    BATCH_IDX: torch.Tensor = None
    ROLLOUT_IDX: torch.Tensor = None
    selected_count: int = None
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    # --- 新增 L2Seg 边特征 ---
    l2seg_edge_feats: torch.Tensor = None      # 边特征张量
    l2seg_edge_indices: torch.Tensor = None    # 边对应的终点索引 (因为只取了 KNN)


class Env:
    """
    Environment for VRP problems with learned destroy operations.

    Manages problem instances, solutions, and the selection of nodes to remove
    from current solutions via neural network policy.
    """

    def __init__(self, num_processes: int, **env_params):
        """
        Initialize VRP environment.

        Args:
            use_multiprocessing: Whether to use parallel processing for instances
            **env_params: Environment configuration parameters
        """
        self.env_params = env_params
        self.problem_size = env_params["problem_size"]
        self.num_nodes_to_remove = env_params["num_nodes_to_remove"]

        # Device and sizing (set during init_instances)
        self.device = None
        self.batch_size = None
        self.rollout_size = None

        # Batch/rollout indexing tensors
        self.BATCH_IDX = None
        self.ROLLOUT_IDX = None

        # Initialize problem generator
        self.problem = self._create_problem(
            env_params["problem"],
            self.problem_size,
            env_params.get("generator_params", None),
        )

        # Initialize instance set manager
        starting_solution_params = env_params.get("starting_solution_params", {})
        self.instanceSet = InstanceSet(
            env_params["problem"],
            num_processes,
            starting_solution_params=starting_solution_params,
        )

        # Problem data and features
        self.problem_data = None
        self.problem_feat = None

        # Dynamic state during episode
        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.ninf_mask = None
        self.step_state = StepState()

    def _create_problem(
        self, problem_type: str, problem_size: int, generator_params: Optional[dict]
    ):
        """Create problem instance generator based on problem type."""
        if problem_type == "cvrp":
            return ProblemCVRP(problem_size, generator_params)
        elif problem_type == "vrptw":
            return ProblemVRPTW(problem_size, generator_params)
        elif problem_type == "pcvrp":
            return ProblemPCVRP(problem_size, generator_params)
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

    def load_problem_dataset_pkl(
        self, filename: str, num_problems: int, index_begin: int = 0
    ) -> None:
        """Load problem dataset from pickle file."""
        self.problem.load_problem_dataset_pkl(filename, num_problems, index_begin)

    def load_problem_dataset_pt(self, filename: str, device: torch.device) -> None:
        """Load problem dataset from PyTorch file."""
        self.problem.load_problem_dataset_pt(filename, device)

    def init_instances(
        self,
        nb_instances: int,
        rollout_size: int,
        device: torch.device,
        aug_factor: int = 1,
    ) -> None:
        """
        Initialize problem instances and create starting solutions.

        Args:
            nb_instances: Number of problem instances to create
            rollout_size: Number of parallel rollouts per instance
            device: Device to place tensors on
            aug_factor: Data augmentation factor
        """
        self.rollout_size = rollout_size
        self.device = device

        # Generate problem instances
        self.batch_size, self.problem_data, self.problem_feat = (
            self.problem.init_problems(nb_instances, aug_factor)
        )

        # Create batch and rollout index tensors
        self._init_index_tensors()

        # Create initial solutions
        self.instanceSet.init_instances(self.problem_data)

        # --- 新增：计算 L2Seg 静态特征 ---
        self._compute_l2seg_static_features()

        self.get_model_input(device)

    def _init_index_tensors(self) -> None:
        """Initialize batch and rollout indexing tensors."""
        self.BATCH_IDX = torch.arange(self.batch_size, device=self.device)[
            :, None
        ].expand(self.batch_size, self.rollout_size)
        self.ROLLOUT_IDX = torch.arange(self.rollout_size, device=self.device)[
            None, :
        ].expand(self.batch_size, self.rollout_size)
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.ROLLOUT_IDX = self.ROLLOUT_IDX

    def reset(self) -> StepState:
        """
        Reset environment for new episode.

        Returns:
            Initial step state
        """
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros(
            (self.batch_size, self.rollout_size, 0),
            dtype=torch.long,
            device=self.device,
        )

        # Initialize mask (depot cannot be selected)
        self.ninf_mask = torch.zeros(
            size=(self.batch_size, self.rollout_size, self.problem_size + 1),
            device=self.device,
        )
        self.ninf_mask[:, :, 0] = float("-inf")

        # Update step state
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask

        return self.step_state

    def get_model_input(self, device: torch.device) -> ResetState:
        """
        Extract model input features from current solutions, including L2Seg dynamic features.
        """
        neighbours = np.zeros((self.batch_size, self.problem_size, 2), dtype=np.int_)
        tour_index = np.zeros((self.batch_size, self.problem_size), dtype=np.int_) - 1

        tours = self.instanceSet.getTours()

        for b_idx in range(self.batch_size):
            tour = tours[b_idx]
            tour_with_depot = [[0, *route, 0] for route in tour]

            for tour_idx, route in enumerate(tour_with_depot):
                for pos in range(1, len(route) - 1):
                    customer = route[pos]
                    tour_index[b_idx, customer - 1] = tour_idx
                    neighbours[b_idx, customer - 1] = [route[pos - 1], route[pos + 1]]

        tour_index_tensor = torch.tensor(tour_index, dtype=torch.long, device=device)
        neighbours_tensor = torch.tensor(neighbours, dtype=torch.long, device=device)
        
        # ==========================================
        # 补全 Node 特征: 动态内部性, 路径重心, 相连节点坐标
        # ==========================================
        batch_indices = torch.arange(self.batch_size, device=device).view(-1, 1, 1)
        knn_idx_device = self.knn_indices.to(device)
        
        # [特征 A: 内部性 Internality (6维)]
        neighbor_tours = tour_index_tensor[batch_indices, knn_idx_device]
        self_tours = tour_index_tensor.unsqueeze(-1)
        same_tour_mask = (neighbor_tours == self_tours).float()
        
        ratio_k5 = same_tour_mask[:, :, :5].mean(dim=-1, keepdim=True)
        ratio_k15 = same_tour_mask[:, :, :15].mean(dim=-1, keepdim=True)
        ratio_k40 = same_tour_mask[:, :, :40].mean(dim=-1, keepdim=True)
        
        # 计算相对百分比 K% = 5%, 15%, 40%
        k_pct_5 = max(1, int(self.problem_size * 0.05))
        k_pct_15 = max(1, int(self.problem_size * 0.15))
        k_pct_40 = max(1, int(self.problem_size * 0.40))
        ratio_pct_5 = same_tour_mask[:, :, :k_pct_5].mean(dim=-1, keepdim=True)
        ratio_pct_15 = same_tour_mask[:, :, :k_pct_15].mean(dim=-1, keepdim=True)
        ratio_pct_40 = same_tour_mask[:, :, :k_pct_40].mean(dim=-1, keepdim=True)
        
        internality_feats = torch.cat([ratio_k5, ratio_k15, ratio_k40, ratio_pct_5, ratio_pct_15, ratio_pct_40], dim=-1)

        # [特征 B: 路径重心 Centroid (2维)] 高效并行计算，避免 for 循环
        node_xy = self.problem_feat.node_xy.to(device)
        t_idx = tour_index_tensor.clone()
        t_idx[t_idx == -1] = self.problem_size # 将未访问点指向虚拟索引
        t_idx_expanded = t_idx.unsqueeze(-1).expand(-1, -1, 2)
        
        sums = torch.zeros(self.batch_size, self.problem_size + 1, 2, device=device)
        sums.scatter_add_(1, t_idx_expanded, node_xy)
        counts = torch.zeros(self.batch_size, self.problem_size + 1, 1, device=device)
        counts.scatter_add_(1, t_idx.unsqueeze(-1), torch.ones_like(t_idx.unsqueeze(-1), dtype=torch.float))
        
        centroids = sums / (counts + 1e-9)
        centroid_xy = torch.gather(centroids, 1, t_idx_expanded)

        # [特征 C: 左右邻居坐标及连接成本 (6维)]
        depot_xy = self.problem_feat.depot_xy.to(device)
        full_xy = torch.cat([depot_xy, node_xy], dim=1) # (batch, problem_size+1, 2)
        
        left_idx = neighbours_tensor[:, :, 0]
        right_idx = neighbours_tensor[:, :, 1]
        
        left_xy = torch.gather(full_xy, 1, left_idx.unsqueeze(-1).expand(-1, -1, 2))
        right_xy = torch.gather(full_xy, 1, right_idx.unsqueeze(-1).expand(-1, -1, 2))
        
        left_cost = torch.norm(node_xy - left_xy, dim=-1, keepdim=True)
        right_cost = torch.norm(node_xy - right_xy, dim=-1, keepdim=True)
        
        neighbor_geom_feats = torch.cat([left_xy, right_xy, left_cost, right_cost], dim=-1)

        # 组合动态特征：2 (重心) + 6 (相连邻居) + 6 (内部性) = 14 维
        l2seg_dynamic_feats = torch.cat([centroid_xy, neighbor_geom_feats, internality_feats], dim=-1)

        # ==========================================
        # 补全 Edge 特征 (Cost, In_solution, Rank)
        # ==========================================
        edge_cost = self.knn_distances.to(device)
        
        # 距离排名就是索引的排序位置 (0 到 39)
        edge_rank = torch.arange(40, device=device, dtype=torch.float32)
        edge_rank = edge_rank.view(1, 1, -1).expand_as(edge_cost)
        
        left_neighbor = neighbours_tensor[:, :, 0].unsqueeze(-1)
        right_neighbor = neighbours_tensor[:, :, 1].unsqueeze(-1)
        
        in_sol_left = (knn_idx_device == left_neighbor).float()
        in_sol_right = (knn_idx_device == right_neighbor).float()
        edge_in_solution = in_sol_left + in_sol_right 
        
        # 最终边特征: [距离成本, 是否在解中, 成本排名] -> 3 维
        l2seg_edge_feats = torch.stack([edge_cost, edge_in_solution, edge_rank], dim=-1)

        # ==========================================
        # 组装返回状态
        # ==========================================
        reset_state = ResetState()
        reset_state.problem_feat = self.problem_feat
        reset_state.tour_index = tour_index_tensor
        reset_state.neighbours = neighbours_tensor
        
        reset_state.l2seg_static_feats = self.l2seg_static_feats.to(device)
        reset_state.l2seg_dynamic_feats = l2seg_dynamic_feats
        
        reset_state.l2seg_edge_feats = l2seg_edge_feats
        reset_state.l2seg_edge_indices = knn_idx_device

        return reset_state

    def step(self, selected: torch.Tensor) -> Tuple[StepState, bool]:
        """
        Execute one step of node selection.

        Args:
            selected: Selected nodes (batch, rollout)

        Returns:
            Tuple of (updated step state, done flag)
        """
        # Update selection state
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, self.current_node[:, :, None]), dim=2
        )

        # Mask selected nodes
        self.ninf_mask[self.BATCH_IDX, self.ROLLOUT_IDX, selected] = float("-inf")

        # Update step state
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask

        # Check if done
        done = self.selected_count == self.num_nodes_to_remove

        return self.step_state, done

    def _compute_l2seg_static_features(self):
        """Compute static features for L2Seg: angles, distance to depot, and KNN distances."""
        depot_coords = self.problem_feat.depot_xy  # (batch, 1, 2)
        node_coords = self.problem_feat.node_xy    # (batch, problem_size, 2)

        # 1. 计算相对坐标和角度
        rel_coords = node_coords - depot_coords
        angles = torch.atan2(rel_coords[:, :, 1], rel_coords[:, :, 0]).unsqueeze(-1) # 扩展为 (batch, N, 1)
        
        # 计算到仓库的距离
        dist_to_depot = torch.norm(rel_coords, dim=-1, keepdim=True) # 扩展为 (batch, N, 1)
        
        # 2. 计算加权角度 
        weighted_angles = angles * dist_to_depot # (batch, N, 1)

        # 3. 计算所有节点之间的距离矩阵以获取 KNN 距离 
        dist_matrix = torch.cdist(node_coords, node_coords) 
        dist_matrix.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
        
        # 获取最近 3 个邻居的距离和 K=40 的索引
        knn_distances, knn_indices = torch.topk(dist_matrix, k=40, dim=-1, largest=False)
        closest_3_dist = knn_distances[:, :, :3] # (batch, N, 3)
        
        self.knn_indices = knn_indices 
        self.knn_distances = knn_distances

        # ==========================================
        # 组装 8 维静态特征
        # ==========================================
        self.l2seg_static_feats = torch.cat([
            rel_coords,         # 2维: 相对 XY 坐标
            dist_to_depot,      # 1维: 距 Depot 距离
            angles,             # 1维: 极坐标角度
            weighted_angles,    # 1维: 距离加权角度
            closest_3_dist      # 3维: 最近 3 邻居的距离
        ], dim=-1)