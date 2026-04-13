import networkx as nx

class L2SegLabelGenerator:
    """
    生成 L2Seg 模型所需的 NAR (非自回归) 和 AR (自回归) 训练标签。
    """
    def __init__(self, problem_size):
        self.problem_size = problem_size
        self.END_TOKEN = problem_size + 1  # 定义一个特殊的 end token (对应 x_end) [cite: 257]

    def _extract_edges(self, tours):
        """将路径列表转换为无向边的集合。"""
        edges = set()
        for route in tours:
            # 加上首尾的 depot (0)
            full_route = [0] + route + [0]
            for i in range(len(full_route) - 1):
                u, v = full_route[i], full_route[i+1]
                # 统一为无向边 (小索引在前)，避免方向导致的不匹配
                edges.add((min(u, v), max(u, v)))
        return edges

    def generate_labels(self, tours_before, tours_after):
        """
        核心函数：输入优化前后的路径，按局部连通块切分，输出多个独立的微型子问题标签。
        """
        edges_before = self._extract_edges(tours_before)
        edges_after = self._extract_edges(tours_after)

        # 1. 计算 Ediff: 找出被删除的边和新插入的边 
        edges_deleted = edges_before - edges_after
        edges_inserted = edges_after - edges_before
        e_diff = edges_deleted.union(edges_inserted)

        subproblem_labels = []
        
        if not e_diff:
            return subproblem_labels

        # 构建连通块图
        G = nx.MultiGraph()
        for u, v in edges_deleted:
            G.add_edge(u, v, action='delete')
        for u, v in edges_inserted:
            G.add_edge(u, v, action='insert')

        # 提取连通块 (Connected Components) 
        components = list(nx.connected_components(G))

        for comp in components:
            # 找出这个连通块牵扯到了哪几条旧路线
            involved_route_indices = set()
            for idx, route in enumerate(tours_before):
                route_set = set(route)
                # 如果这条路线上的客户点在连通块里，说明这条路线被“动刀”了
                if route_set.intersection(comp):
                    involved_route_indices.add(idx)
            
            # 过滤：只保留牵扯到 1~2 条路线的连通块 
            if len(involved_route_indices) == 0 or len(involved_route_indices) > 2:
                continue
                
            # 收集这 1~2 条路线上的所有节点 (构建 P_K)
            involved_nodes = set()
            involved_nodes.add(0)  # 必须包含车场 (Depot)
            for idx in involved_route_indices:
                involved_nodes.update(tours_before[idx])
                
            # 排序成 list，确保后续特征切片时顺序固定
            involved_nodes = sorted(list(involved_nodes))
            
            # 生成局部的 NAR 标签 (只有这 1~2 条路线上的点才参与打标)
            nar_labels = [1 if node in comp else 0 for node in involved_nodes]

            # 生成局部的 AR 序列
            sub_g = G.subgraph(comp)
            start_node = list(comp)[0]
            dfs_nodes = list(nx.dfs_preorder_nodes(sub_g, source=start_node))
            
            # 加上结束符 (END_TOKEN)
            ar_sequence = dfs_nodes + [self.END_TOKEN]
            
            # 打包成字典，作为一个独立的样本存入
            subproblem_labels.append({
                "involved_nodes": involved_nodes, 
                "nar_labels": nar_labels,         
                "ar_sequence": ar_sequence        
            })

        return subproblem_labels