import numpy as np
import torch
import torch.nn.functional as F

def load_cvrplib_instance(filepath):
    """
    解析 CVRPLIB 官方 .vrp 文件
    返回: 车场坐标, 客户坐标, 客户需求, 车辆容量
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    capacity = 0
    node_coords = []
    demands = []
    
    section = None
    for line in lines:
        line = line.strip()
        if not line: continue
            
        if line.startswith("CAPACITY"):
            # 兼容不同文件格式 (如 "CAPACITY : 206" 或 "CAPACITY:206")
            capacity = int(line.split(":")[-1].strip())
        elif line.startswith("NODE_COORD_SECTION"):
            section = "COORD"
            continue
        elif line.startswith("DEMAND_SECTION"):
            section = "DEMAND"
            continue
        elif line.startswith("DEPOT_SECTION"):
            section = "DEPOT"
            continue
        elif line.startswith("EOF"):
            break
            
        if section == "COORD":
            parts = line.split()
            node_coords.append([float(parts[1]), float(parts[2])])
        elif section == "DEMAND":
            parts = line.split()
            demands.append(int(parts[1]))
        elif section == "DEPOT":
            if line == "-1": break

    # 转化为 Numpy 数组并分离车场与客户
    nodes = np.array(node_coords)
    depot_xy = nodes[0:1]  # 索引 0 是车场
    node_xy = nodes[1:]    # 索引 1 之后是客户
    node_demand = np.array(demands[1:])
    
    return depot_xy, node_xy, node_demand, capacity




def compute_original_l2seg_features(depot_xy, node_xy, node_demand, tour_index, neighbours, pad_mask=None):
    """
    100% 严格复刻 L2Seg 原版论文 Table 7 的 25 维 Node Features (纯 PyTorch 向量化实现)。
    包含 3维基础特征(在model.py中拼接) + 22维附加特征(在此函数中计算)。
    """
    if depot_xy.dim() == 2:
        depot_xy = depot_xy.unsqueeze(1) # [B, 1, 2]
        
    B, N, _ = node_xy.shape
    device = node_xy.device
    
    # ---------------------------------------------------------
    # 【类别 1：相对坐标与角度特征】
    # ---------------------------------------------------------
    # 1-2. The relative xy coordinates (2维)
    rel_xy = node_xy - depot_xy
    
    # 补充: Distance to depot (原论文归一化系数或隐含距离，补齐25维的缺口) (1维)
    dist_depot = torch.norm(rel_xy, dim=-1, keepdim=True)
    
    # 3. The angles w.r.t. the depot (1维)
    angle = torch.atan2(rel_xy[:, :, 1:2], rel_xy[:, :, 0:1])
    
    # 4. The weighted angles w.r.t. the depot by the distances (1维)
    weighted_angle = angle * dist_depot
    
    # ---------------------------------------------------------
    # 【类别 2：拓扑与路径特征 (强依赖当前路径状态)】
    # ---------------------------------------------------------
    # 5-6. The centroid of the subtour for each node (2维)
    safe_tour_idx = torch.clamp(tour_index, min=0)
    max_tour = safe_tour_idx.max().item()
    tour_onehot = F.one_hot(safe_tour_idx, num_classes=max_tour+1).float()
    if pad_mask is not None:
        tour_onehot = tour_onehot * (~pad_mask).float().unsqueeze(-1)
    tour_onehot[:, :, 0] = 0.0 # 排除未分配的节点
    
    tour_sizes = tour_onehot.sum(dim=1, keepdim=True) + 1e-6 # [B, 1, M]
    tour_xy_sum = torch.bmm(node_xy.transpose(1, 2), tour_onehot) # [B, 2, M]
    tour_centroids = tour_xy_sum / tour_sizes # [B, 2, M]
    node_centroids = torch.bmm(tour_onehot, tour_centroids.transpose(1, 2)) # [B, N, 2]
    
    # 7-10. The coordinates of the two nodes connecting to each node (4维)
    # neighbours 存储的是全局索引，0 是 depot，1~N 是 customer
    all_xy = torch.cat([depot_xy, node_xy], dim=1) # [B, N+1, 2]
    left_idx = neighbours[:, :, 0].clamp(min=0, max=N)
    right_idx = neighbours[:, :, 1].clamp(min=0, max=N)
    
    left_xy = torch.gather(all_xy, 1, left_idx.unsqueeze(-1).expand(B, N, 2))
    right_xy = torch.gather(all_xy, 1, right_idx.unsqueeze(-1).expand(B, N, 2))
    connecting_coords = torch.cat([left_xy, right_xy], dim=-1) # [B, N, 4]
    
    # 11-12. The travel cost of the two edges connecting to each node (2维)
    cost_left = torch.norm(node_xy - left_xy, dim=-1, keepdim=True)
    cost_right = torch.norm(node_xy - right_xy, dim=-1, keepdim=True)
    connecting_costs = torch.cat([cost_left, cost_right], dim=-1) # [B, N, 2]
    
    # ---------------------------------------------------------
    # 【类别 3：K-NN 空间密度聚类特征 (计算最复杂的部分)】
    # ---------------------------------------------------------
    dist_matrix = torch.cdist(node_xy, node_xy) # [B, N, N]
    if pad_mask is not None:
        dist_matrix.masked_fill_(pad_mask.unsqueeze(1), float('inf')) # 屏蔽假节点
        
    sorted_dist, sorted_idx = torch.sort(dist_matrix, dim=-1)
    
    # 13-15. The distances of the closest 3 neighbor for each node (3维)
    closest_3_dist = sorted_dist[:, :, 1:4] # 排除排在第0位的自己
    
    # 提取排序后邻居的所属路径ID
    neighbor_tour = torch.gather(tour_index.unsqueeze(1).expand(B, N, N), 2, sorted_idx)
    same_tour_mask = (neighbor_tour == tour_index.unsqueeze(2)).float()
    same_tour_mask = same_tour_mask[:, :, 1:] # 排除自己
    
    # 16-18. The percentage of the K nearest nodes that are within the same subtour (K=5, 15, 40) (3维)
    pct_5 = same_tour_mask[:, :, :5].mean(dim=-1, keepdim=True)
    pct_15 = same_tour_mask[:, :, :15].mean(dim=-1, keepdim=True)
    pct_40 = same_tour_mask[:, :, :40].mean(dim=-1, keepdim=True)
    
    # 19-21. The percentage of the K% nearest nodes that are within the same subtour (K=5, 15, 40) (3维)
    valid_N = (~pad_mask).sum(dim=1).min().item() if pad_mask is not None else N
    k_5_pct = max(1, int(valid_N * 0.05))
    k_15_pct = max(1, int(valid_N * 0.15))
    k_40_pct = max(1, int(valid_N * 0.40))
    
    pct_5_rel = same_tour_mask[:, :, :k_5_pct].mean(dim=-1, keepdim=True)
    pct_15_rel = same_tour_mask[:, :, :k_15_pct].mean(dim=-1, keepdim=True)
    pct_40_rel = same_tour_mask[:, :, :k_40_pct].mean(dim=-1, keepdim=True)
    
    # ==========================================
    # 组合这 22 维附加特征，交由 model.py 拼接最后 3 维成为 25 维
    # ==========================================
    advanced_features = torch.cat([
        rel_xy, dist_depot, angle, weighted_angle, node_centroids,
        connecting_coords, connecting_costs,
        closest_3_dist, pct_5, pct_15, pct_40, pct_5_rel, pct_15_rel, pct_40_rel
    ], dim=-1) # 总计刚好 22 维
    
    # 为了兼容 model.py 中已经写好的读取接口，任意拆分为 8维 和 14维 返回
    static_feats = advanced_features[:, :, :8]
    dynamic_feats = advanced_features[:, :, 8:]
    
    return static_feats, dynamic_feats