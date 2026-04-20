import os
import random
from typing import Any, Dict, List

import torch

from .label_generator import L2SegLabelGenerator


class ExpertDatasetCollector:
    """Collect and persist L2Seg expert labels as standalone dataset artifacts."""

    def __init__(
        self,
        env_params: Dict[str, Any],
        tester_params: Dict[str, Any],
        result_folder: str,
        logger,
    ):
        self.env_params = env_params
        self.tester_params = tester_params
        self.result_folder = result_folder
        self.logger = logger
        self.enabled = bool(self.tester_params.get("expert_data_mode", False))
        self.training_data_buffer: List[Dict[str, Any]] = []
        self.label_generator = (
            L2SegLabelGenerator(int(self.env_params["problem_size"]))
            if self.enabled
            else None
        )

    @property
    def sample_count(self) -> int:
        return len(self.training_data_buffer)

    def _should_accept_expert_label(self, improvement: float, rng: random.Random) -> bool:
        """Algorithm 2 gating: improvement threshold η_improv + stochastic acceptance α_AC."""
        min_improvement = float(self.tester_params.get("eta_improv", 0.0))
        if improvement < min_improvement:
            return False

        alpha_ac = float(self.tester_params.get("alpha_ac", 0.0))
        # alpha_ac <= 0 means no extra random downsampling.
        if alpha_ac <= 0.0:
            return True
        alpha_ac = min(alpha_ac, 1.0)
        return rng.random() <= alpha_ac

    def collect_from_transition(
        self,
        tours_before: List[List[int]],
        current_cost: float,
        new_solution,
        state_snapshot: Dict[str, torch.Tensor],
        rng: random.Random,
    ) -> None:
        if not self.enabled:
            return

        improvement = float(current_cost - new_solution.totalCosts)
        if improvement <= 0:
            return

        subproblem_labels = self.label_generator.generate_labels(
            tours_before, new_solution.getTourList()
        )
        for sub_label in subproblem_labels:
            if not self._should_accept_expert_label(improvement, rng):
                continue

            involved_nodes = sub_label["involved_nodes"]
            raw_nar_labels = sub_label.get("nar_labels", [])
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

            # Rebuild NAR labels in local node space: [depot, customer_1..customer_k].
            node_to_nar = {
                int(node_id): int(label)
                for node_id, label in zip(involved_nodes, raw_nar_labels)
            }
            nar_labels_local = [node_to_nar.get(0, 0)] + [
                node_to_nar.get(x, 0) for x in customer_nodes
            ]

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
                    "nar_labels": nar_labels_local,
                    "ar_sequences": sub_label["ar_sequence"],
                    "state_dict": local_state_dict,
                }
            )

    def save(self) -> None:
        """Persist collected labels into a .pt file."""
        if not self.enabled:
            self.logger.info(
                "expert_data_mode=False，跳过训练标签保存（仅执行求解/评测）。"
            )
            return
        if not self.training_data_buffer:
            self.logger.info("未收集到任何有效的标签数据。")
            return

        save_path = self.tester_params.get(
            "l2s_data_save_path",
            os.path.join(self.result_folder, "l2seg_training_data.pt"),
        )
        save_path = os.path.abspath(os.path.expandvars(os.path.expanduser(save_path)))

        test_data_cfg = self.tester_params.get("test_data_load", {})
        test_data_path = (
            test_data_cfg.get("filename") if test_data_cfg.get("enable", False) else None
        )
        if test_data_path:
            test_data_path = os.path.abspath(
                os.path.expandvars(os.path.expanduser(test_data_path))
            )
            same_path = os.path.normcase(os.path.normpath(save_path)) == os.path.normcase(
                os.path.normpath(test_data_path)
            )
            if same_path:
                base, ext = os.path.splitext(save_path)
                ext = ext if ext else ".pt"
                redirected_path = f"{base}.generated{ext}"
                self.logger.warning(
                    "l2s_data_save_path 与 test_data_load.filename 指向同一路径，"
                    "为避免覆盖测试数据，训练标签将保存到: %s",
                    redirected_path,
                )
                save_path = redirected_path

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(self.training_data_buffer, save_path)
        self.logger.info(
            f"成功保存 {len(self.training_data_buffer)} 条训练数据至 {save_path}"
        )
