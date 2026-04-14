import numpy as np
import os
import tempfile
import subprocess
import time
import uuid   
import tarfile
from pathlib import Path
from typing import List, Set, Tuple, Dict

class FSTA_Compressor:
    """
    严谨对齐 L2Seg 论文的 First-Segment-Then-Aggregate 算法
    """
    def __init__(
        self,
        node_xy: np.ndarray,
        node_demand: np.ndarray,
        capacity: int,
        lkh_path: str = "./LKH-3.exe",
        max_vehicles: int = 50,
        timeout_sec: int = 15,
    ):
        self.node_xy = node_xy
        self.node_demand = node_demand
        self.capacity = capacity
        self.lkh_path = self._resolve_lkh_path(lkh_path)
        self.max_vehicles = max_vehicles
        self.timeout_sec = max(1, int(timeout_sec))

    def _resolve_lkh_path(self, lkh_path: str) -> str:
        """Resolve LKH executable path; auto-bootstrap from .tgz source archive on Linux."""
        repo_root = Path(__file__).resolve().parent.parent

        def _as_abs(path_str: str) -> Path:
            p = Path(path_str)
            if not p.is_absolute():
                p = repo_root / p
            return p

        def _make_executable(path: Path) -> bool:
            try:
                if path.exists() and path.is_file():
                    path.chmod(path.stat().st_mode | 0o111)
                return path.exists() and path.is_file() and os.access(path, os.X_OK)
            except OSError:
                return False

        def _probe_candidate(path: Path):
            if path.is_file():
                if path.suffix.lower() in {".tgz", ".gz"}:
                    return None
                if _make_executable(path):
                    return path
                return None
            if path.is_dir():
                for name in ("LKH", "LKH-3", "LKH.exe", "LKH-3.exe"):
                    candidate = path / name
                    if _make_executable(candidate):
                        return candidate
            return None

        configured = _as_abs(lkh_path) if lkh_path else None
        candidates = []
        if configured is not None:
            candidates.append(configured)

        candidates.extend(
            [
                repo_root / "LKH",
                repo_root / "LKH-3",
                repo_root / "LKH-3.exe",
                repo_root / "LKH-3.0.14" / "LKH",
                repo_root / "LKH-3.0.14",
            ]
        )

        for candidate in candidates:
            hit = _probe_candidate(candidate)
            if hit is not None:
                return str(hit)

        archive_candidates = []
        if configured is not None and configured.is_file():
            if configured.suffix.lower() == ".tgz" or configured.name.endswith(".tar.gz"):
                archive_candidates.append(configured)
        archive_candidates.append(repo_root / "LKH-3.0.14.tgz")

        for archive_path in archive_candidates:
            built = self._build_lkh_from_archive(archive_path, _probe_candidate)
            if built is not None:
                return str(built)

        return lkh_path

    def _build_lkh_from_archive(self, archive_path: Path, probe_candidate):
        if not archive_path.exists() or not archive_path.is_file():
            return None

        target_root = archive_path.parent
        extracted_dir = target_root / archive_path.name.replace(".tar.gz", "").replace(".tgz", "")

        try:
            if not extracted_dir.exists():
                with tarfile.open(archive_path, "r:gz") as tar:
                    target_root_real = target_root.resolve()
                    for member in tar.getmembers():
                        member_target = (target_root / member.name).resolve()
                        if not str(member_target).startswith(str(target_root_real) + os.sep):
                            raise RuntimeError(f"Unsafe tar member path: {member.name}")
                    tar.extractall(path=target_root)

            built = probe_candidate(extracted_dir)
            if built is not None:
                return built

            subprocess.run(
                ["make"],
                cwd=str(extracted_dir),
                check=True,
                capture_output=True,
                text=True,
            )

            return probe_candidate(extracted_dir)
        except (tarfile.TarError, OSError, subprocess.SubprocessError, RuntimeError):
            return None

    def _extract_segments(self, tours: List[List[int]], destroyed_nodes: Set[int]) -> List[List[int]]:
        """
        步骤 1：切割稳定段
        如果一条路径上连续几个点没有被破坏，它们就构成一个 Segment。
        """
        segments = []
        for tour in tours:
            current_segment = []
            for node in tour:
                if node in destroyed_nodes:
                    if len(current_segment) > 0:
                        segments.append(current_segment)
                        current_segment = []
                else:
                    current_segment.append(node)
            if len(current_segment) > 0:
                segments.append(current_segment)
        return segments

    def run_fsta_reoptimization(self, tours: List[List[int]], destroyed_nodes: Set[int]):
        """
        步骤 2 & 3：图压缩与 LKH 求解 (严密对齐 Appendix B.1.5)
        """
        def _fallback_tour(tours: List[List[int]]) -> List[int]:
            flat = []
            for route in tours:
                if not route:
                    continue
                flat.extend(route)
                flat.append(0)
            if flat:
                flat.pop()
            return flat

        # 1. 提取所有 Segment
        segments = self._extract_segments(tours, destroyed_nodes)
        
        # 2. 构建新图节点映射
        # 新图包含: Depot(0) + 所有被破坏的点 + 所有 Segment 的首尾节点
        new_nodes = [0] 
        new_nodes.extend(list(destroyed_nodes))
        
        segment_endpoints = [] # 记录 (start, end)
        for seg in segments:
            if len(seg) == 1:
                new_nodes.append(seg[0]) # 孤立的稳定点
            else:
                new_nodes.append(seg[0])   # 提取首节点
                new_nodes.append(seg[-1])  # 提取尾节点
                segment_endpoints.append((seg[0], seg[-1]))
                
        # 建立 全局ID <-> 新图ID 的双向映射
        # LKH 的索引要求从 1 开始
        global_to_new = {g_id: n_id + 1 for n_id, g_id in enumerate(new_nodes)}
        new_to_global = {n_id + 1: g_id for n_id, g_id in enumerate(new_nodes)}
        num_new_nodes = len(new_nodes)
        if num_new_nodes <= 1:
            return _fallback_tour(tours)

        # 3. 构建 LKH 需要的显式距离矩阵和需求表 (核心魔法)
        distances = np.zeros((num_new_nodes, num_new_nodes), dtype=int)
        demands = np.zeros(num_new_nodes, dtype=int)
        
        # 计算新需求 (Demand Aggregation)
        for i, g_id in enumerate(new_nodes):
            n_id = i + 1
            # 找到这个点属于哪个 segment
            belong_seg = next((s for s in segments if g_id in [s[0], s[-1]] and len(s) > 1), None)
            if belong_seg:
                # 【论文对齐】：需求平分 (d_start = d_end = sum / 2)
                total_seg_demand = sum(self.node_demand[n] for n in belong_seg)
                demands[i] = total_seg_demand // 2
            else:
                demands[i] = self.node_demand[g_id]

        # 计算新距离矩阵 (Distance Aggregation)
        # 先全部算真实的欧式距离 (乘以10000转整数以适应LKH)
        for i in range(num_new_nodes):
            for j in range(num_new_nodes):
                g_i, g_j = new_nodes[i], new_nodes[j]
                dist = np.linalg.norm(self.node_xy[g_i] - self.node_xy[g_j])
                distances[i, j] = int(dist * 10000)

        # 【论文对齐】：将双超节点内部的距离强行置为 0
        fixed_edges_lkh = []
        for start_g, end_g in segment_endpoints:
            s_n, e_n = global_to_new[start_g], global_to_new[end_g]
            distances[s_n - 1, e_n - 1] = 0
            distances[e_n - 1, s_n - 1] = 0
            fixed_edges_lkh.append((s_n, e_n)) # 强制锁死


        # ==========================================
        # 👑 新增：防止过度压缩导致的“载货量悖论”
        # ==========================================
        total_demand = sum(demands)
        if self.capacity <= 0:
            return _fallback_tour(tours)
        min_required_vehicles = max(1, int(np.ceil(total_demand / self.capacity)))
        
        if num_new_nodes - 1 < min_required_vehicles:
            # 如果节点被压缩得比必须派出的车辆数还少，这是物理无解的。直接放弃本次退火！
            return _fallback_tour(tours)

        # 4. 写入临时目录并调用 LKH-3 (多进程安全版)
        temp_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
        
        # 👑 核心改造 1：为这一次执行生成一个绝不重复的 32 位随机 ID
        run_id = uuid.uuid4().hex
        
        vrp_path = os.path.join(temp_dir, f"fsta_problem_{run_id}.vrp")
        par_path = os.path.join(temp_dir, f"fsta_params_{run_id}.par")
        out_path = os.path.join(temp_dir, f"fsta_output_{run_id}.tour")

        try:
            self._write_explicit_vrp(vrp_path, num_new_nodes, distances, demands, fixed_edges_lkh)
            self._write_par(par_path, vrp_path, out_path)
            
            # 👑 核心修复 1：动态分配车辆！最多 50 辆，但绝不能超过当前图的客户总数！
            safe_vehicles = min(self.max_vehicles, num_new_nodes - 1)
            self._write_par(par_path, vrp_path, out_path, vehicles=safe_vehicles)


            try:
                self._write_explicit_vrp(vrp_path, num_new_nodes, distances, demands, fixed_edges_lkh)
            
                # 👑 完美动态车辆分配：既满足最低运力，又不超过节点数和最大限制
                max_feasible_vehicles = max(1, num_new_nodes - 1)
                safe_vehicles = max(min_required_vehicles, min(self.max_vehicles, max_feasible_vehicles))
                assert safe_vehicles <= max_feasible_vehicles, (
                    f"safe_vehicles ({safe_vehicles}) exceeded "
                    f"max_feasible_vehicles ({max_feasible_vehicles})"
                )
                self._write_par(par_path, vrp_path, out_path, vehicles=safe_vehicles)
            
                process_result = subprocess.run(
                    [self.lkh_path, par_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec
                )
            
                # 👑 核心防御：如果 LKH 底层崩溃，绝对不能交白卷！必须触发平滑回退
                if process_result.returncode != 0:
                    print("\n" + "="*60)
                    print(f"💀 LKH-3 引擎发生底层崩溃！已成功拦截，正在平滑回退。")
                    print(f"👉 错误原因 (STDERR): {process_result.stderr}")
                    print("="*60 + "\n")
                
                    return _fallback_tour(tours)
                
                # 5. 读取与无损解码
                lkh_tour_new = self._parse_tour(out_path)
                return self._recover_solution(lkh_tour_new, new_to_global, segments)
        
            except subprocess.TimeoutExpired as e:
                print("\n" + "!"*60)
                print(f"LKH-3 陷入死循环，已被 Python 强制狙击！")
                print(f"让我们看看它死前到底卡在哪一步了：")
                # e.stdout 里面保存了 LKH-3 运行到一半被杀时的所有控制台输出！
                print(f"{e.stdout}")
                print(f"顺便检查一下容量参数对不对：Capacity = {self.capacity}")
                print("!"*60 + "\n")
                

                # 👑 核心修复 2：回退时，必须把二维的 tours 拍平成一维（用 0 隔开），无缝喂给外面的代码！
                return _fallback_tour(tours)
            except (FileNotFoundError, PermissionError, OSError) as e:
                print("\n" + "="*60)
                print("💀 LKH-3 不可执行或缺失，已成功拦截，正在平滑回退。")
                print(f"👉 错误原因: {e}")
                print(f"👉 当前 LKH 路径: {self.lkh_path}")
                print("="*60 + "\n")
                return _fallback_tour(tours)


                
            lkh_tour_new = self._parse_tour(out_path)
            return self._recover_solution(lkh_tour_new, new_to_global, segments)
            
        finally:
            # 👑 核心改造 2：无论成功还是报错，必须自动毁尸灭迹，防止硬盘撑爆！
            for f_path in [vrp_path, par_path, out_path]:
                if os.path.exists(f_path):
                    try:
                        os.remove(f_path)
                    except OSError:
                        pass

    def _recover_solution(self, lkh_tour_new: List[int], new_to_global: Dict[int, int], segments: List[List[int]]) -> List[int]:
        """
        步骤 4：解映射 (Solution Recovery)
        将超节点重新展开为完整的物理节点段，并彻底清洗 LKH 产生的假车场。
        """
        # 转回全局 ID
        global_tour = []
        for n in lkh_tour_new:
            if n == 1: # 忽略 LKH 原生的真车场
                continue
            
            # 检查映射表
            if n in new_to_global:
                val = new_to_global[n]
            elif (n - 1) in new_to_global:
                val = new_to_global[n - 1]
            else:
                val = n
            
            # 如果映射出来的是一个段（list），则平铺展开；如果是单点，直接加入
            if isinstance(val, list):
                global_tour.extend(val)
            else:
                global_tour.append(val)
        
        # 建立 Segment 的首尾快速查找字典
        seg_dict = {}
        for seg in segments:
            if len(seg) > 1:
                seg_dict[(seg[0], seg[-1])] = seg
                seg_dict[(seg[-1], seg[0])] = list(reversed(seg))

        current_tour = []
        i = 0
        while i < len(global_tour):
            curr_node = global_tour[i]
            # 检查是否遇到了双超节点
            if i < len(global_tour) - 1:
                next_node = global_tour[i+1]
                pair = (curr_node, next_node)
                if pair in seg_dict:
                    # 展开折叠的超节点！
                    current_tour.extend(seg_dict[pair])
                    i += 2
                    continue
            
            current_tour.append(curr_node)
            i += 1
            
        # =======================================================
        # 👑 终极清洗机：将 LKH 产生的假车场统一转换为标准分隔符 0
        # =======================================================
        num_customers = len(self.node_xy) - 1
        clean_1d_tour = []
        for node in current_tour:
            if node > num_customers or node == 0:
                # 遇到真假车场（例如 132），统一视为路线分割，插入一个 0
                # 必须保证不连续插入 0，防止空路线
                if len(clean_1d_tour) > 0 and clean_1d_tour[-1] != 0:
                    clean_1d_tour.append(0)
            else:
                clean_1d_tour.append(node)
                
        # 掐头去尾，去掉首尾可能多余的 0
        while len(clean_1d_tour) > 0 and clean_1d_tour[0] == 0:
            clean_1d_tour.pop(0)
        while len(clean_1d_tour) > 0 and clean_1d_tour[-1] == 0:
            clean_1d_tour.pop()
            
        return clean_1d_tour
    
    def _write_explicit_vrp(self, path, n, distances, demands, fixed_edges):
        """写入显式全连接矩阵 TSPLIB"""
        with open(path, 'w') as f:
            f.write("NAME : FSTA_Reduced\nTYPE : CVRP\n")
            f.write(f"DIMENSION : {n}\n")
            f.write("EDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
            f.write(f"CAPACITY : {self.capacity}\n")
            
            f.write("EDGE_WEIGHT_SECTION\n")
            for row in distances:
                f.write(" ".join(map(str, row)) + "\n")
                
            f.write("DEMAND_SECTION\n")
            for i in range(n):
                f.write(f"{i+1} {demands[i]}\n")
                
            f.write("DEPOT_SECTION\n1\n-1\n")
            
            if fixed_edges:
                f.write("FIXED_EDGES_SECTION\n")
                for u, v in fixed_edges:
                    f.write(f"{u} {v}\n")
                f.write("-1\n")
            f.write("EOF\n")
            

            
    def _parse_tour(self, path):
        tour = []
        if os.path.exists(path):
            with open(path, 'r') as f:
                lines = f.readlines()
                in_tour = False
                for line in lines:
                    if "TOUR_SECTION" in line: in_tour = True; continue
                    if "-1" in line or "EOF" in line: in_tour = False; continue
                    if in_tour: tour.append(int(line.strip()))
        return tour
    
    # 👑 接收我们传进来的 vehicles
    def _write_par(self, par_path, vrp_path, out_path, vehicles=50):
        with open(par_path, 'w') as f:
            f.write(f"PROBLEM_FILE = {vrp_path}\nTOUR_FILE = {out_path}\n")
            f.write("RUNS = 1\nTIME_LIMIT = 10\nTRACE_LEVEL = 0\n")
            
            # 👑 突破“假车场黑洞”：视距必须穿透所有的假车场，再额外看到 5 个真实客户！
            cands = max(20, vehicles + 5)
            f.write(f"MAX_CANDIDATES = {cands}\n") 
            
            f.write(f"VEHICLES = {max(1, int(vehicles))}\n")
