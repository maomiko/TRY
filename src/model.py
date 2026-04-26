"""Neural network model to learn selecting nodes to remove from VRP solutions."""

from typing import Tuple, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.problem = model_params["problem"]
        # 注意：需要把 problem_size 也传入 model_params 以便确定词表大小
        self.problem_size = model_params.get("problem_size", 100) 
        embedding_dim = model_params["embedding_dim"]

        # [关键] 必须定义这个学习参数，用于解码器的起始状态
        self.start_last_node = nn.Parameter(
            torch.zeros(embedding_dim), requires_grad=True
        )

        # Encoder and decoder
        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)

        self.encoded_nodes = None

        # ==========================================
        # L2Seg 新增: NAR 与 AR 输出头
        # ==========================================
        # NAR头: 对图上的每个节点(包含Depot)进行二分类，判断是否需要破坏
        self.nar_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        
        # 类别总数: Depot(1) + 客户(problem_size) + END_TOKEN(1) = problem_size + 2
        self.vocab_size = self.problem_size + 2
        
        # 定义专属的 PAD_TOKEN，它的索引是 problem_size + 2
        self.PAD_TOKEN = self.problem_size + 2
        
    def _build_ar_embedding_pool(self) -> torch.Tensor:
        """
        Build a fixed AR token embedding pool:
        [Depot(0), Customers(1..problem_size), END(problem_size+1), PAD(problem_size+2)].
        """
        if self.encoded_nodes is None:
            raise RuntimeError(
                "encoded_nodes is None. Ensure Model.pre_forward() is called before AR decoding."
            )

        batch_size = self.encoded_nodes.size(0)
        embedding_dim = self.model_params["embedding_dim"]
        device = self.encoded_nodes.device

        # 固定池大小，保证 token 索引在全流程中稳定
        pool = torch.zeros(
            batch_size, self.PAD_TOKEN + 1, embedding_dim, device=device
        )

        # 复制可用节点特征到 [Depot + Customers] 槽位
        available_node_slots = self.vocab_size - 1  # 0..problem_size
        copy_len = min(self.encoded_nodes.size(1), available_node_slots)
        pool[:, :copy_len, :] = self.encoded_nodes[:, :copy_len, :]

        # 写入 END_TOKEN 的可学习向量表示
        end_token_emb = self.decoder.end_token_key.transpose(1, 2).expand(
            batch_size, 1, -1
        )
        pool[:, self.vocab_size - 1 : self.vocab_size, :] = end_token_emb
        # PAD_TOKEN 保持全零向量（不参与语义）
        return pool

    def pre_forward(self, reset_state, z: torch.Tensor = None) -> None:
        """
        Encode problem instance and solution structure with L2Seg 25-dim Enhanced Features.
        """
        if z is not None:
            device = z.device
        else:
            device = next(self.parameters()).device

        # Extract and move features to device
        depot_xy = reset_state.problem_feat.depot_xy.to(device)
        node_xy = reset_state.problem_feat.node_xy.to(device)
        node_demand = reset_state.problem_feat.node_demand.to(device)

        # 1. 提取基础特征 (x, y, demand) -> 3维
        base_feat = torch.cat((node_xy, node_demand[:, :, None]), dim=2)

        # ==========================================
        # 🌟 无缝拼接 L2Seg 22维增强特征 
        # ==========================================
        static_feats = reset_state.l2seg_static_feats.to(device)     # 8维
        dynamic_feats = reset_state.l2seg_dynamic_feats.to(device)   # 14维
        if static_feats.size(-1) != 8 or dynamic_feats.size(-1) != 14:
            raise ValueError(
                "L2Seg feature dimension mismatch: expected static=8 and dynamic=14, "
                f"got static={static_feats.size(-1)}, dynamic={dynamic_feats.size(-1)}"
            )
        if static_feats.size(1) != node_xy.size(1) or dynamic_feats.size(1) != node_xy.size(1):
            raise ValueError(
                "L2Seg feature node-count mismatch with node_xy: "
                f"node_xy={node_xy.size(1)}, static={static_feats.size(1)}, dynamic={dynamic_feats.size(1)}"
            )
        
        # 3 + 8 + 14 = 25 维！彻底完成 Node Features 升级
        node_feat = torch.cat((base_feat, static_feats, dynamic_feats), dim=2)

        # Add problem-specific features
        if self.problem == "vrptw":
            node_feat = torch.cat(
                (node_feat, reset_state.problem_feat.node_tw.to(device)), dim=2
            )
        elif self.problem == "pcvrp":
            node_prizes = reset_state.problem_feat.node_prizes.to(device)
            node_feat = torch.cat((node_feat, node_prizes[:, :, None]), dim=2)
        expected_feat_dim = {"cvrp": 25, "vrptw": 28, "pcvrp": 26}[self.problem]
        if node_feat.size(-1) != expected_feat_dim:
            raise ValueError(
                f"node_feat dim mismatch for {self.problem}: "
                f"expected {expected_feat_dim}, got {node_feat.size(-1)}"
            )

        # Solution structure
        solution_neighbours = reset_state.neighbours.to(device)
        tour_index = reset_state.tour_index.to(device)

        # 【提取 Padding 掩码】
        pad_mask = getattr(reset_state, 'pad_mask', None)
        if pad_mask is not None:
            pad_mask = pad_mask.to(device)

        # 喂入 Encoder
        self.encoded_nodes = self.encoder(
            depot_xy, node_feat, solution_neighbours, tour_index, pad_mask
        )
        self.decoder.set_kv(self.encoded_nodes, z, pad_mask)

    def forward(
        self, state, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Select next node to remove.

        Args:
            state: Current selection state
            temperature: Softmax temperature

        Returns:
            Tuple of (selected nodes, probabilities, all probabilities)
        """
        batch_size = state.BATCH_IDX.size(0)
        rollout_size = state.BATCH_IDX.size(1)

        # Get encoding of last selected node (or start token)
        if state.current_node is None:
            encoded_last_node = self.start_last_node[None, None].expand(
                batch_size, rollout_size, -1
            )
        else:
            encoded_last_node = _gather_by_index(self.encoded_nodes, state.current_node)

        # Get selection probabilities
        probs = self.decoder(
            encoded_last_node, ninf_mask=state.ninf_mask, temperature=temperature
        )

        # Sample or take argmax
        if self.training or self.model_params["eval_type"] == "softmax":
            selected, prob = self._sample_from_probs(
                probs, state, batch_size, rollout_size
            )
        else:
            selected = probs.argmax(dim=2)
            prob = None

        return selected, prob, probs[:, :, 1:]  # Exclude depot from probs

    
    # ==========================================
    # L2Seg 新增方法
    # ==========================================
    def nar_forward(self) -> torch.Tensor:
        """
        计算 NAR 分支的 Logits (不稳定节点的分类概率)
        Returns:
            shape: (batch_size, problem_size + 1)
        """
        # self.encoded_nodes 形状: (batch_size, problem_size + 1, embedding_dim)
        # 经过 nar_head 后形状变为 (batch_size, problem_size + 1, 1)，利用 squeeze 展平
        nar_logits = self.nar_head(self.encoded_nodes).squeeze(-1)
        return nar_logits

    def ar_forward(self, ar_sequences: torch.Tensor) -> torch.Tensor:
        """
        计算 AR 分支的序列 Logits (Teacher Forcing)
        Args:
            ar_sequences: 专家给出的真实节点序列，shape (batch_size, seq_len)
        Returns:
            shape: (batch_size, seq_len, vocab_size)
        """
        # 1. 直接从 Encoder 输出动态抓取节点特征，避免静态 ID Embedding
        full_embedding_pool = self._build_ar_embedding_pool()
        safe_sequences = ar_sequences.clamp(min=0, max=full_embedding_pool.size(1) - 1)
        gather_idx = safe_sequences.unsqueeze(-1).expand(
            -1, -1, self.model_params["embedding_dim"]
        )
        seq_embeddings = torch.gather(full_embedding_pool, dim=1, index=gather_idx)
        
        # 2. 交给 Decoder 利用上下文和 Encoder 特征处理整个序列
        logits = self.decoder.forward_sequence(seq_embeddings)
        return logits

    @torch.no_grad()
    def generate_sequence_from_node(self, reset_state, starting_node: int, max_steps: int = 50) -> torch.Tensor:
        """
        L2Seg-SYN 的终极推理阶段：从 KMeans 指定的引爆点 (starting_node) 开始，自回归生成破坏序列。
        
        Args:
            reset_state: 当前图的环境状态
            starting_node: 由 KMeans 聚类确定的 AR 解码起始节点 (全图 index)
            max_steps: 最大生成长度，防止死循环
            
        Returns:
            生成的破坏序列，shape: (batch_size, seq_len)
        """
        batch_size = reset_state.problem_feat.node_xy.size(0)
        device = next(self.parameters()).device
        
        # 1. 确保特征已经提取 (如果在外部 search_sa.py 中提取过，这步极快)
        if self.encoded_nodes is None:
            self.pre_forward(reset_state, z=None)
        
        # ==========================================
        # 👑 核心修改：强行空投到“重灾区”起点
        # ==========================================
        # 不再瞎猜起点，而是直接使用传入的 starting_node 构建 batch 的初始输入
        safe_starting_node = self._sanitize_starting_node(starting_node)
        start_nodes = torch.full(
            (batch_size,), safe_starting_node, dtype=torch.long, device=device
        )
        
        generated_sequences = [start_nodes.unsqueeze(1)]
        current_nodes = start_nodes
        
        # 初始化 GRU 隐藏状态
        self.decoder.GRU_hidden = torch.zeros(batch_size, self.model_params["embedding_dim"], device=device)
        # 如果模型有条件生成，初始化 latent vector z
        z_input = torch.zeros(batch_size, 1, getattr(self.decoder, "z_dim", 16), device=device) 
        if self.decoder.z is not None:
            z_input = self.decoder.z if self.decoder.z.dim() == 3 else self.decoder.z.unsqueeze(1)
        
        # 记录每个 batch 是否已经生成了 END_TOKEN
        # END_TOKEN 的索引是 vocab_size - 1
        end_token_idx = self.vocab_size - 1
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # 3. AR 自回归生成循环 (顺藤摸瓜寻找不稳定边)
        # 获取用于删除阶段约束的邻居矩阵 (batch, N, 2)
        # 注意：通常 neighbours 记录的是 N 个客户的邻居，索引为 0 的是车场
        solution_neighbours = reset_state.neighbours.to(device)

        # 3. AR 自回归生成循环：严格交替的 删除(Delete) 与 插入(Insert) 阶段
        full_embedding_pool = self._build_ar_embedding_pool()
        max_token_idx = full_embedding_pool.size(1) - 1

        for step in range(max_steps - 1):
            # 将当前节点从动态特征池中取 Embedding
            current_nodes = current_nodes.clamp(min=0, max=max_token_idx)
            gather_idx = current_nodes.unsqueeze(-1).unsqueeze(-1).expand(
                -1, 1, self.model_params["embedding_dim"]
            )
            step_input = torch.gather(
                full_embedding_pool, dim=1, index=gather_idx
            ).squeeze(1)  # (batch, dim)
            
            # --- 单步 Decoder 前向过程 ---
            self.decoder.GRU_hidden = self.decoder.GRU(step_input, self.decoder.GRU_hidden)
            context = self.decoder.GRU_hidden.unsqueeze(1)
            
            # ==========================================
            # 👑 动态路由：根据阶段选择不同的神经网络权重视野
            # ==========================================
            is_deletion_stage = (step % 2 == 0)
            
            current_context = context
            memory = self.encoded_nodes  # 获取 Encoder 输出的图特征
            
            if is_deletion_stage:
                # 浅层处理 (Deletion, t=2k)：专用于局部视野
                for layer in self.decoder.delete_mha:
                    current_context = layer(current_context, memory, ninf_mask=None)
            else:
                # 深层处理 (Insertion, t=2k+1)：专用于全局视野
                for layer in self.decoder.insert_mha:
                    current_context = layer(current_context, memory, ninf_mask=None)
            
            mh_out = current_context
            
            poly_out = self.decoder.poly_layer_1(torch.cat((mh_out, z_input), dim=2))
            poly_out = F.relu(poly_out)
            poly_out = self.decoder.poly_layer_2(poly_out)
            context_out = mh_out + poly_out
            
            # Pointer Network 原始打分
            scores = torch.matmul(context_out, self.decoder.single_head_key)
            scores = scores / self.decoder.sqrt_embedding_dim
            scores = self.decoder.logit_clipping * torch.tanh(scores)
            
            # ==========================================
            # 👑 核心科研级改造：双阶段交替 Masking 机制
            # ==========================================
            # 基础屏蔽：屏蔽已经生成过的节点，防止模型在两个点之间死循环打转
            for seq in generated_sequences:
                scores.scatter_(2, seq.unsqueeze(1), float('-inf'))

            is_deletion_stage = (step % 2 == 0)

            if is_deletion_stage:
                # ----------------------------------------------------
                # 阶段 A：删除阶段 (Deletion Stage, t=2k)
                # 约束 1：只能选原图中与 current_nodes 物理相连的节点。
                # 约束 2：绝对不允许选 END_TOKEN (边必须成对操作，不能悬空结束)。
                # ----------------------------------------------------
                batch_indices = torch.arange(batch_size, device=device)
                
                # 初始化一个全为 -inf 的动态 Mask
                deletion_mask = torch.full_like(scores, float('-inf'))
                
                for b in range(batch_size):
                    curr_node = current_nodes[b].item()
                    if curr_node == end_token_idx or curr_node == self.PAD_TOKEN:
                        # 虚拟节点没有物理邻居，允许它继续生成结束符占位即可
                        deletion_mask[b, 0, end_token_idx] = 0.0
                    elif curr_node == 0:
                        # 如果当前节点是车场 (Depot)，它连接了多条边，这里放宽限制，允许连接全图未选节点
                        deletion_mask[b, 0, 1:end_token_idx] = 0.0
                    else:
                        # 如果是普通客户节点，严格提取它在原解中的 2 个邻居
                        # 假设 solution_neighbours 维度为 (batch, problem_size, 2)，客户索引需 -1
                        customer_idx = curr_node - 1
                        if 0 <= customer_idx < solution_neighbours.size(1):
                            left_neighbor = int(solution_neighbours[b, customer_idx, 0].item())
                            right_neighbor = int(solution_neighbours[b, customer_idx, 1].item())
                            if 0 <= left_neighbor < scores.size(2):
                                deletion_mask[b, 0, left_neighbor] = 0.0
                            if 0 <= right_neighbor < scores.size(2):
                                deletion_mask[b, 0, right_neighbor] = 0.0
                        else:
                            deletion_mask[b, 0, 1:end_token_idx] = 0.0
                
                # 叠加 Mask，强迫模型只能在合法的边上进行 Destroy 操作
                scores = scores + deletion_mask
                
                # 强制屏蔽结束符
                scores[:, :, end_token_idx] = float('-inf')

            else:
                # ----------------------------------------------------
                # 阶段 B：插入阶段 (Insertion Stage, t=2k+1)
                # 约束 1：这是寻找“桥梁”去往下一个不稳定区域，所以是全局 Attention，开放所有未选节点。
                # 约束 2：【允许】模型抛出 END_TOKEN，认为图已完全修复，结束搜索。
                # ----------------------------------------------------
                # 基础屏蔽（防重复选点）已经生效，END_TOKEN 天然处于开放状态
                pass 
            
            # ==========================================
            
            # 选出得分最高的下一个节点
            next_nodes = scores.argmax(dim=2).squeeze(1) # (batch_size,)
            next_nodes = next_nodes.clamp(min=0, max=self.PAD_TOKEN)
            
            generated_sequences.append(next_nodes.unsqueeze(1))
            current_nodes = next_nodes
            
            # 只有在插入阶段结束后，检查是否生成了 END_TOKEN
            is_finished = is_finished | (next_nodes == end_token_idx)
            
            if is_finished.all():
                break
                
        # 拼接所有生成的节点 (包含了 starting_node 直到 END_TOKEN)
        final_sequences = torch.cat(generated_sequences, dim=1)
        return final_sequences

    def _sanitize_starting_node(self, starting_node: int) -> int:
        """Clamp invalid starting node index with explicit warning."""
        try:
            start_idx = int(starting_node)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"starting_node must be int-convertible, got {type(starting_node)!r}"
            ) from exc

        if start_idx < 0 or start_idx > self.PAD_TOKEN:
            warnings.warn(
                (
                    f"starting_node={start_idx} is out of valid range [0, {self.PAD_TOKEN}]. "
                    "It will be clamped."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        return max(0, min(start_idx, self.PAD_TOKEN))

    def _sample_from_probs(
        self, probs: torch.Tensor, state, batch_size: int, rollout_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from probability distribution (with retry for zero-probability bug).

        Args:
            probs: Selection probabilities
            state: Current state
            batch_size: Batch size
            rollout_size: Rollout size

        Returns:
            Tuple of (selected nodes, their probabilities)
        """
        # Workaround for PyTorch multinomial bug with zero probabilities
        while True:
            with torch.no_grad():
                selected = (
                    probs.reshape(batch_size * rollout_size, -1)
                    .multinomial(1)
                    .squeeze(dim=1)
                    .reshape(batch_size, rollout_size)
                )
            # Gather selected action probabilities with gradient flow
            prob = probs[state.BATCH_IDX, state.ROLLOUT_IDX, selected]
            # Use detached values for the zero-probability check to avoid creating large graphs
            if (prob.detach() != 0).all():
                break

        return selected, prob


def _gather_by_index(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Gather elements from tensor using indices.

    Args:
        tensor: Tensor to gather from, shape (batch, seq_len, dim)
        indices: Indices to gather, shape (batch, rollout)

    Returns:
        Gathered tensor, shape (batch, rollout, dim)
    """
    batch_size = indices.size(0)
    rollout_size = indices.size(1)
    embedding_dim = tensor.size(2)

    gathering_index = indices[:, :, None].expand(
        batch_size, rollout_size, embedding_dim
    )
    return tensor.gather(dim=1, index=gathering_index)


########################################
# ENCODER
########################################


class CVRP_Encoder(nn.Module):
    """
    Encoder for VRP instances and solution structures.

    Architecture:
    1. Initial embedding layers
    2. Self-attention layers
    3. Tour aggregation layer (optional)
    4. Message passing layers
    5. Final self-attention layers
    """

    def __init__(self, **model_params):
        """Initialize encoder."""
        super().__init__()
        self.model_params = model_params
        self.problem = model_params["problem"]
        self.embedding_dim = model_params["embedding_dim"]

        # Input embedding layers
        self.embedding_depot = nn.Linear(2, self.embedding_dim)
        self.embedding_node = self._create_node_embedding()

        # Encoder layers (self-attention + feedforward)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(**model_params)
                for _ in range(model_params["encoder_layer_num"])
            ]
        )

        # Tour aggregation layer
        self.tour_layer = (
            TourLayer(**model_params) if model_params["tour_layer"] else None
        )

        # Message passing layers (leverage solution structure)
        self.mp_layers = nn.ModuleList(
            [
                MessagePassingLayer(**model_params)
                for _ in range(model_params["message_passing_layer_num"])
            ]
        )

        # Final encoder layers
        self.layers_2 = nn.ModuleList(
            [
                EncoderLayer(**model_params)
                for _ in range(model_params["encoder_layer_num_2"])
            ]
        )

    def _create_node_embedding(self) -> nn.Linear:
        """Create node embedding layer based on problem type."""
        problem_to_features = {
            # ==========================================
            # 【L2Seg 特征升级】将原生 3 维升级为论文标准的 25 维
            # ==========================================
            "cvrp": 25,  # [x, y, demand] + 22 维空间/连通性增强特征
            "vrptw": 28, # vrptw 在 cvrp 基础上多 3 个时间特征
            "pcvrp": 26, 
        }

        num_features = problem_to_features.get(self.problem)
        if num_features is None:
            raise ValueError(f"Unsupported problem type: {self.problem}")

        return nn.Linear(num_features, self.embedding_dim)

    def forward(
        self,
        depot_xy, 
        node_feat, 
        solution_neighbours, 
        tour_index, 
        pad_mask=None
    ) -> torch.Tensor:
        
        batch_size = node_feat.shape[0]
        num_customers = node_feat.shape[1]

        embedded_depot = self.embedding_depot(depot_xy)
        embedded_node = self.embedding_node(node_feat)
        out = torch.cat((embedded_depot, embedded_node), dim=1)

        # 【核心修复】：pad_mask 已经包含了 Depot 和客户！
        # 不要再拼接 depot_mask，直接取反即可！
        attn_mask = None
        if pad_mask is not None:
            # PyTorch 中 True 表示假节点 (Padding)
            # scaled_dot_product_attention 接受 boolean mask 时，True 表示允许参与注意力
            valid_mask = ~pad_mask 
            attn_mask = valid_mask.unsqueeze(1).unsqueeze(2) 

        for layer in self.layers:
            out = layer(out, attn_mask=attn_mask)

        if self.tour_layer is not None:
            out = self.tour_layer(batch_size, num_customers, tour_index, out)

        for layer in self.mp_layers:
            out = layer(batch_size, num_customers, solution_neighbours, out)

        for layer in self.layers_2:
            out = layer(out, attn_mask=attn_mask)

        return out


class TourLayer(nn.Module):
    """
    Aggregates customer embeddings by tour to capture tour-level information.

    For each customer, computes a tour embedding by summing embeddings of all
    customers in the same tour, then combines with customer embedding.
    """

    def __init__(self, **model_params):
        """Initialize tour layer."""
        super().__init__()
        embedding_dim = model_params["embedding_dim"]

        self.embedding_dim = embedding_dim
        self.tour_combiner = nn.Linear(embedding_dim * 2, embedding_dim)
        self.feedforward_layer = nn.Linear(embedding_dim, embedding_dim)
        self.add_and_normalize = AddAndInstanceNormalization(**model_params)

    def forward(
        self,
        batch_size: int,
        num_customers: int,
        tour_index: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply tour aggregation.

        Args:
            batch_size: Batch size
            num_customers: Number of customers
            tour_index: Tour assignments, shape (batch, problem)
            embeddings: Node embeddings, shape (batch, problem+1, embedding)

        Returns:
            Updated embeddings, shape (batch, problem+1, embedding)
        """
        # Extract customer embeddings (exclude depot)
        customer_embeddings = embeddings[:, 1:]

        # Handle unvisited customers (tour_index == -1, e.g., in PCVRP)
        max_tour_idx = tour_index.max()
        has_unvisited = tour_index.min() == -1

        if has_unvisited:
            tour_index = tour_index.clone()
            tour_index[tour_index == -1] = max_tour_idx + 1
            max_tour_idx += 1

        # Initialize tour embeddings
        tour_embeddings = torch.zeros(
            batch_size,
            max_tour_idx + 1,
            self.embedding_dim,
            dtype=customer_embeddings.dtype,
            device=customer_embeddings.device,
        )

        # Aggregate customer embeddings by tour
        expand_dim = customer_embeddings.shape[2]
        tour_embeddings.scatter_add_(
            1, tour_index[:, :, None].expand(-1, -1, expand_dim), customer_embeddings
        )

        # Zero out dummy tour for unvisited customers
        if has_unvisited:
            tour_embeddings[:, -1] = 0

        # Gather tour embedding for each customer
        customer_tour_embeddings = torch.gather(
            tour_embeddings, 1, tour_index[:, :, None].expand(-1, -1, expand_dim)
        )

        # Combine customer and tour embeddings
        combined = torch.cat((customer_embeddings, customer_tour_embeddings), dim=2)
        combined = F.relu(self.tour_combiner(combined))
        combined = self.feedforward_layer(combined)

        # Residual connection with normalization
        updated_customers = self.add_and_normalize(customer_embeddings, combined)

        # Re-attach depot
        return torch.cat((embeddings[:, [0]], updated_customers), dim=1)


class MessagePassingLayer(nn.Module):
    """
    Message passing layer that leverages solution structure (neighbour relationships).

    For each customer, aggregates information from its left and right neighbours
    in the current solution.
    """

    def __init__(self, **model_params):
        """Initialize message passing layer."""
        super().__init__()
        embedding_dim = model_params["embedding_dim"]

        self.embedding_dim = embedding_dim
        self.directed_graph = model_params["problem"] == "vrptw"

        # Neighbour projection layers
        if self.directed_graph:
            self.left_neighbour_projector = nn.Linear(
                embedding_dim, embedding_dim, bias=False
            )
            self.right_neighbour_projector = nn.Linear(
                embedding_dim, embedding_dim, bias=False
            )
        else:
            self.neighbour_projector = nn.Linear(
                embedding_dim, embedding_dim, bias=False
            )

        # Combination layers
        self.neighbour_combiner = nn.Linear(embedding_dim * 2, embedding_dim)
        self.feedforward_layer = nn.Linear(embedding_dim, embedding_dim)
        self.add_and_normalize = AddAndInstanceNormalization(**model_params)

    def forward(
        self,
        batch_size: int,
        num_customers: int,
        solution_neighbours: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply message passing.

        Args:
            batch_size: Batch size
            num_customers: Number of customers
            solution_neighbours: Neighbour indices, shape (batch, problem, 2)
            embeddings: Node embeddings, shape (batch, problem+1, embedding)

        Returns:
            Updated embeddings, shape (batch, problem+1, embedding)
        """
        # Gather left and right neighbour embeddings
        left_neighbours = self._gather_neighbours(
            embeddings, solution_neighbours[:, :, 0], batch_size, num_customers
        )
        right_neighbours = self._gather_neighbours(
            embeddings, solution_neighbours[:, :, 1], batch_size, num_customers
        )

        # Project neighbour embeddings
        if self.directed_graph:
            left_neighbours = self.left_neighbour_projector(left_neighbours)
            right_neighbours = self.right_neighbour_projector(right_neighbours)
        else:
            left_neighbours = self.neighbour_projector(left_neighbours)
            right_neighbours = self.neighbour_projector(right_neighbours)

        # Aggregate neighbour information
        neighbour_info = left_neighbours + right_neighbours

        # Combine with customer embeddings
        customer_embeddings = embeddings[:, 1:]
        combined = torch.cat((customer_embeddings, neighbour_info), dim=2)
        combined = F.relu(self.neighbour_combiner(combined))
        combined = self.feedforward_layer(combined)

        # Residual connection with normalization
        updated_customers = self.add_and_normalize(customer_embeddings, combined)

        # Re-attach depot
        return torch.cat((embeddings[:, [0]], updated_customers), dim=1)

    def _gather_neighbours(
        self,
        embeddings: torch.Tensor,
        indices: torch.Tensor,
        batch_size: int,
        num_customers: int,
    ) -> torch.Tensor:
        """Gather neighbour embeddings by indices."""
        return torch.gather(
            embeddings,
            1,
            indices[:, :, None].expand(batch_size, num_customers, self.embedding_dim),
        )


class EncoderLayer(nn.Module):
    """Standard Transformer encoder layer with multi-head attention and feedforward."""

    def __init__(self, **model_params):
        """Initialize encoder layer."""
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        head_num = model_params["head_num"]
        qkv_dim = model_params["qkv_dim"]

        # Multi-head attention
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # Normalization and feedforward
        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

        self.head_num = head_num

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply encoder layer.

        Args:
            x: Input embeddings, shape (batch, seq_len, embedding)

        Returns:
            Output embeddings, shape (batch, seq_len, embedding)
        """
        # Multi-head self-attention
        q = reshape_by_heads(self.Wq(x), self.head_num)
        k = reshape_by_heads(self.Wk(x), self.head_num)
        v = reshape_by_heads(self.Wv(x), self.head_num)

        attn_out = fast_multi_head_attention(q, k, v, custom_mask=attn_mask)
        attn_out = self.multi_head_combine(attn_out)

        # First residual connection
        x = self.add_n_normalization_1(x, attn_out)

        # Feedforward
        ff_out = self.feed_forward(x)

        # Second residual connection
        x = self.add_n_normalization_2(x, ff_out)

        return x


########################################
# DECODER
########################################


class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        poly_embedding_dim = model_params["poly_embedding_dim"]
        head_num = model_params["head_num"]
        qkv_dim = model_params["qkv_dim"]
        z_dim = model_params["z_dim"]

        # 浅层网络：1 层 Cross-Attention，专用于局部视野的 Deletion
        self.delete_mha = nn.ModuleList([
            DecoderCrossAttentionLayer(**model_params) for _ in range(1)
        ])
        
        # 深层网络：4 层 Cross-Attention，专用于全局大海捞针的 Insertion
        self.insert_mha = nn.ModuleList([
            DecoderCrossAttentionLayer(**model_params) for _ in range(4)
        ])

        # GRU for maintaining decoding state
        self.GRU = nn.GRUCell(embedding_dim, embedding_dim)

        # Polynomial network for conditioning on latent vectors
        self.poly_layer_1 = nn.Linear(embedding_dim + z_dim, poly_embedding_dim)
        self.poly_layer_2 = nn.Linear(poly_embedding_dim, embedding_dim)

        # Model parameters
        self.head_num = head_num
        self.sqrt_embedding_dim = model_params["sqrt_embedding_dim"]
        self.logit_clipping = model_params["logit_clipping"]
        self.z_dim = model_params["z_dim"] # 将 z_dim 保存为类属性
        self.problem_size = model_params.get("problem_size", 100)

        # 兼容 forward / forward_sequence 的共享注意力投影
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        
        # L2Seg 新增: 一个可学习的向量，代表 END_TOKEN 在 Pointer Network 中的被指向 Key
        self.end_token_key = nn.Parameter(torch.randn(1, embedding_dim, 1))
        
        # Cached values
        self.k = None
        self.v = None
        self.single_head_key = None
        self.z = None
        self.GRU_hidden = None

    def set_kv(self, encoded_nodes: torch.Tensor, z: Optional[torch.Tensor], pad_mask: Optional[torch.Tensor] = None) -> None:
        """设置注意力键值，并拼接 END_TOKEN"""

        self.z = z
        self.GRU_hidden = None
        self.pad_mask = pad_mask # 【新增】保存掩码供打分时使用
        
        # 原始节点 Key（后续会对齐到固定全局槽位）: (batch, dim, local_nodes_with_depot)
        base_key = encoded_nodes.transpose(1, 2)
        batch_size = encoded_nodes.size(0)

        # 供注意力层复用的 KV 缓存（不含 END_TOKEN）
        self.k = reshape_by_heads(self.Wk(encoded_nodes), self.head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), self.head_num)
        
        # 将节点 Key 对齐到全局固定槽位 [Depot + problem_size customers]
        fixed_node_slots = self.problem_size + 1
        if base_key.size(2) < fixed_node_slots:
            pad_len = fixed_node_slots - base_key.size(2)
            key_pad = torch.zeros(
                batch_size, base_key.size(1), pad_len, device=base_key.device
            )
            base_key = torch.cat([base_key, key_pad], dim=2)
        elif base_key.size(2) > fixed_node_slots:
            raise ValueError(
                f"encoded node slots ({base_key.size(2)}) exceed configured problem size "
                f"({fixed_node_slots - 1})."
            )

        # 将 END_TOKEN 扩展到当前 batch size，并拼接到节点后面
        # 拼接后形状: (batch, dim, problem_size + 2)
        end_key_expanded = self.end_token_key.expand(batch_size, -1, -1)
        self.single_head_key = torch.cat([base_key, end_key_expanded], dim=2)

    def forward(
        self,
        encoded_last_node: torch.Tensor,
        ninf_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute selection probabilities.

        Args:
            encoded_last_node: Encoding of last selected node
            ninf_mask: Mask for invalid selections
            temperature: Softmax temperature

        Returns:
            Selection probabilities, shape (batch, rollout, problem+1)
        """
        batch_size = encoded_last_node.shape[0]
        rollout_size = encoded_last_node.shape[1]
        embedding_dim = encoded_last_node.shape[2]

        # Update GRU hidden state
        self.GRU_hidden = self.GRU(
            encoded_last_node.reshape(batch_size * rollout_size, embedding_dim),
            self.GRU_hidden,
        )
        context = self.GRU_hidden.reshape(batch_size, rollout_size, embedding_dim)

        # Multi-head attention
        q = reshape_by_heads(self.Wq_last(context), self.head_num)
        attn_out = fast_multi_head_attention(
            q, self.k, self.v, rank3_ninf_mask=ninf_mask
        )
        mh_out = self.multi_head_combine(attn_out)

        # Polynomial network (condition on latent vectors)
        poly_out = self.poly_layer_1(torch.cat((mh_out, self.z), dim=2))
        poly_out = F.relu(poly_out)
        poly_out = self.poly_layer_2(poly_out)

        # Add polynomial output
        context_out = mh_out + poly_out

        # Single-head attention for scoring
        scores = torch.matmul(context_out, self.single_head_key)
        scores = scores / self.sqrt_embedding_dim
        scores = self.logit_clipping * torch.tanh(scores)
        
        # 修复: 切片去掉最后一个 END_TOKEN 分数，以匹配 ninf_mask 的维度 (P + 1)
        scores = scores[:, :, :-1] 
        
        scores = scores + ninf_mask

        # Convert to probabilities
        probs = F.softmax(scores / temperature, dim=2)

        return probs

    # ==========================================
    # L2Seg 新增序列处理方法
    # ==========================================
    def forward_sequence(self, seq_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embedding_dim = seq_embeddings.shape
        self.GRU_hidden = torch.zeros(batch_size, embedding_dim, device=seq_embeddings.device)
        all_logits = []
        
        z_input = self.z if self.z is not None else torch.zeros(batch_size, 1, self.z_dim, device=seq_embeddings.device)
        if z_input.dim() == 2:
            z_input = z_input.unsqueeze(1)
            
        for t in range(seq_len):
            step_input = seq_embeddings[:, t, :]
            self.GRU_hidden = self.GRU(step_input, self.GRU_hidden)
            context = self.GRU_hidden.unsqueeze(1) 
            
            q = reshape_by_heads(self.Wq_last(context), self.head_num)
            attn_out = fast_multi_head_attention(q, self.k, self.v, rank3_ninf_mask=None)
            mh_out = self.multi_head_combine(attn_out)
            
            poly_out = self.poly_layer_1(torch.cat((mh_out, z_input), dim=2))
            poly_out = F.relu(poly_out)
            poly_out = self.poly_layer_2(poly_out)
            context_out = mh_out + poly_out
            
            scores = torch.matmul(context_out, self.single_head_key) 
            scores = scores / self.sqrt_embedding_dim
            scores = self.logit_clipping * torch.tanh(scores)
            
            if getattr(self, 'pad_mask', None) is not None:
                batch_size = scores.size(0)
                node_slots = scores.size(2) - 1  # 最后一个是 END_TOKEN
                node_mask = torch.ones(
                    batch_size, node_slots, dtype=torch.bool, device=scores.device
                )
                node_mask[:, 0] = False  # Depot 永远可用

                if self.pad_mask.size(1) != node_slots:
                    raise ValueError(
                        f"Decoder pad_mask length mismatch: got {self.pad_mask.size(1)}, "
                        f"expected {node_slots} ([depot + customers])."
                    )

                pad_mask_customers = self.pad_mask[:, 1:]
                customer_mask_len = max(0, min(pad_mask_customers.size(1), node_slots - 1))
                if customer_mask_len > 0:
                    node_mask[:, 1 : 1 + customer_mask_len] = pad_mask_customers[
                        :, :customer_mask_len
                    ]

                # END_TOKEN 允许作为合法结束
                end_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=scores.device)
                pointer_mask = torch.cat([node_mask, end_mask], dim=1)
                scores.masked_fill_(pointer_mask.unsqueeze(1), float('-inf'))
                
            all_logits.append(scores)
            
        return torch.cat(all_logits, dim=1)

########################################
# UTILITY LAYERS AND FUNCTIONS
########################################


class AddAndInstanceNormalization(nn.Module):
    """Residual connection with instance normalization."""

    def __init__(self, **model_params):
        """Initialize normalization layer."""
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.norm = nn.InstanceNorm1d(
            embedding_dim, affine=True, track_running_stats=False
        )

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        Apply residual connection with normalization.

        Args:
            input1: Original input
            input2: Transformed input

        Returns:
            Normalized sum
        """
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        return normalized.transpose(1, 2)


class FeedForward(nn.Module):
    """Two-layer feedforward network with ReLU activation."""

    def __init__(self, **model_params):
        """Initialize feedforward network."""
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        ff_hidden_dim = model_params["ff_hidden_dim"]

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feedforward network."""
        return self.W2(F.relu(self.W1(x)))


def reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    """
    Reshape tensor for multi-head attention.

    Args:
        qkv: Input tensor, shape (batch, seq_len, head_num * qkv_dim)
        head_num: Number of attention heads

    Returns:
        Reshaped tensor, shape (batch, head_num, seq_len, qkv_dim)
    """
    batch_size = qkv.size(0)
    seq_len = qkv.size(1)

    qkv = qkv.reshape(batch_size, seq_len, head_num, -1)
    return qkv.transpose(1, 2)


def fast_multi_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rank3_ninf_mask: Optional[torch.Tensor] = None,
    custom_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Efficient multi-head attention using PyTorch's scaled_dot_product_attention.

    Args:
        q: Queries, shape (batch, head_num, seq_len, qkv_dim)
        k: Keys, shape (batch, head_num, seq_len, qkv_dim)
        v: Values, shape (batch, head_num, seq_len, qkv_dim)
        rank3_ninf_mask: Optional attention mask

    Returns:
        Attention output, shape (batch, seq_len, head_num * qkv_dim)
    """
    batch_size = q.size(0)
    head_num = q.size(1)
    seq_len = q.size(2)
    qkv_dim = q.size(3)

    # Prepare mask if provided
    mask = None
    if rank3_ninf_mask is not None:
        mask = rank3_ninf_mask[:, None, :, :].expand(batch_size, head_num, seq_len, -1)
    elif custom_mask is not None:
        mask = custom_mask # 【新增】：直接使用布尔掩码

    # Efficient attention using PyTorch's fused kernel
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    # Reshape output
    out = out.transpose(1, 2)
    out = out.reshape(batch_size, seq_len, head_num * qkv_dim)

    return out

class DecoderCrossAttentionLayer(nn.Module):
    """
    专用于 AR 解码器的交叉注意力层 (Cross-Attention).
    Query 来自 GRU context, Key/Value 来自 Encoder 输出的图特征。
    """
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        head_num = model_params["head_num"]
        qkv_dim = model_params["qkv_dim"]

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # 注意：这里的 Wk 和 Wv 也可以独立，但通常在解码阶段为了节省显存会复用
        # 为了极致隔离，我们为每层赋予独立的投影能力
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.head_num = head_num

    def forward(self, context: torch.Tensor, memory: torch.Tensor, ninf_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        context: GRU 隐藏状态, 形状 (batch, 1, embedding_dim)
        memory: Encoder 输出的节点特征, 形状 (batch, seq_len, embedding_dim)
        """
        q = reshape_by_heads(self.Wq(context), self.head_num)
        k = reshape_by_heads(self.Wk(memory), self.head_num)
        v = reshape_by_heads(self.Wv(memory), self.head_num)

        # 调用你写好的快速注意力计算函数
        attn_out = fast_multi_head_attention(q, k, v, rank3_ninf_mask=ninf_mask)
        out = self.multi_head_combine(attn_out)
        
        # 残差连接 (Residual Connection)
        return context + out
    
L2SegModel = Model
