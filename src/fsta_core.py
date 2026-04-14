import numpy as np
import os
import tempfile
import subprocess
import uuid
import shutil
import stat
from typing import List, Set, Tuple, Dict, Optional

UINT32_MODULUS = 1 << 32
WINDOWS_ACCESS_VIOLATION = -1073741819  # 0xC0000005
WINDOWS_STACK_BUFFER_OVERRUN = -1073740791  # 0xC0000409

class FSTA_Compressor:
    """
    严谨对齐 L2Seg 论文的 First-Segment-Then-Aggregate 算法
    """
    def __init__(
        self,
        node_xy: np.ndarray,
        node_demand: np.ndarray,
        capacity: int,
        lkh_path: str = "./LKH-3",
        max_vehicles: int = 50,
        timeout_sec: int = 15,
    ):
        self.node_xy = node_xy
        self.node_demand = node_demand
        self.capacity = capacity
        self.lkh_path = self._resolve_lkh_path(lkh_path)
        self.max_vehicles = max_vehicles
        self.timeout_sec = max(1, int(timeout_sec))

    def _is_windows_exe(self, path: str) -> bool:
        return path.lower().endswith(".exe")

    def _can_execute(self, path: str) -> bool:
        if not os.path.isfile(path):
            return False
        if os.name == "nt":
            return True
        return os.access(path, os.X_OK)

    def _try_mark_executable(self, path: str) -> None:
        if os.name == "nt" or not os.path.exists(path):
            return
        current_mode = os.stat(path).st_mode
        os.chmod(path, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def _try_build_lkh_from_archive(self, repo_root: str) -> str:
        archive_path = os.path.join(repo_root, "LKH-3.0.14.tgz")
        target_binary = os.path.join(repo_root, "LKH-3")
        if self._can_execute(target_binary):
            return target_binary
        if not os.path.exists(archive_path):
            return ""

        build_dir = tempfile.mkdtemp(prefix="lkh_build_")
        try:
            subprocess.run(
                ["tar", "-xzf", archive_path, "-C", build_dir],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            src_root = os.path.join(build_dir, "LKH-3.0.14")
            env_jobs = os.getenv("LKH_BUILD_JOBS")
            cpu_limit = min(os.cpu_count() or 1, 8)
            try:
                jobs = max(1, int(env_jobs)) if env_jobs is not None else max(1, cpu_limit)
            except (TypeError, ValueError):
                jobs = max(1, cpu_limit)
            subprocess.run(
                ["make", "-C", src_root, f"-j{jobs}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            built_binary = os.path.join(src_root, "LKH")
            if not os.path.exists(built_binary):
                return ""
            shutil.copy2(built_binary, target_binary)
            self._try_mark_executable(target_binary)
            return target_binary if self._can_execute(target_binary) else ""
        except Exception as e:
            print(
                "[LKH auto-build] failed: "
                f"{e}. Ensure tar/make are installed, or manually build LKH-3 from LKH-3.0.14.tgz."
            )
            return ""
        finally:
            shutil.rmtree(build_dir, ignore_errors=True)

    def _resolve_lkh_path(self, configured_path: str) -> str:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        raw_candidates = []
        if configured_path:
            raw_candidates.append(configured_path)
        raw_candidates.extend(["./LKH-3", "./LKH-3.exe", "./LKH"])

        which_lkh = shutil.which("LKH")
        which_lkh3 = shutil.which("LKH-3")
        if which_lkh:
            raw_candidates.append(which_lkh)
        if which_lkh3:
            raw_candidates.append(which_lkh3)

        candidates = []
        seen = set()
        for c in raw_candidates:
            if not c:
                continue
            candidate = c
            if not os.path.isabs(candidate):
                candidate = os.path.abspath(os.path.join(repo_root, candidate))
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)

        checked_candidates = []
        for candidate in candidates:
            checked_candidates.append(candidate)
            if os.name != "nt" and self._is_windows_exe(candidate):
                continue
            if os.path.exists(candidate):
                self._try_mark_executable(candidate)
            if self._can_execute(candidate):
                return candidate

        if os.name != "nt":
            built_path = self._try_build_lkh_from_archive(repo_root)
            if built_path:
                return built_path

        if os.name != "nt":
            raise RuntimeError(
                "No usable LKH binary found. On Linux/macOS, provide tester_params.lkh_path "
                "pointing to a native executable (e.g. ./LKH-3), or keep LKH-3.0.14.tgz in "
                "repository root for auto-build. Checked paths: "
                + ", ".join(checked_candidates)
            )

        raise RuntimeError(
            "No usable LKH binary found. On Windows, set tester_params.lkh_path to LKH-3.exe. "
            "Checked paths: " + ", ".join(checked_candidates)
        )

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

    @staticmethod
    def _format_returncode_as_hex(returncode: int) -> str:
        """Format return code as hex; negative values are mapped to unsigned 32-bit form (common on Windows)."""
        if returncode < 0:
            return hex(UINT32_MODULUS + returncode)
        return hex(returncode)

    def _classify_failure(self, returncode: int, stdout: str, stderr: str) -> str:
        if returncode in (WINDOWS_ACCESS_VIOLATION, WINDOWS_STACK_BUFFER_OVERRUN):
            return "lkh_process_crash"

        msg = f"{stdout}\n{stderr}".lower()
        data_keywords = (
            "dimension",
            "demand",
            "capacity",
            "vehicles",
            "invalid",
            "infeasible",
            "not feasible",
            "parameter",
            "no candidates",
        )
        if any(k in msg for k in data_keywords):
            return "data_or_parameter_error"
        return "unknown_nonzero_exit"

    @staticmethod
    def _extract_head_tail_lines(text: str, line_count: int = 50) -> Tuple[List[str], List[str]]:
        lines = text.splitlines()
        head = lines[:line_count]
        tail = lines[-line_count:] if len(lines) > line_count else lines
        return head, tail

    def _log_lkh_context(
        self,
        run_id: str,
        command: List[str],
        par_path: str,
        vrp_path: str,
        out_path: str,
        safe_vehicles: int,
        min_required_vehicles: int,
        max_feasible_vehicles: int,
    ) -> None:
        print("\n" + "=" * 80)
        print("[LKH trace] minimal reproducible context")
        print(f"run_id: {run_id}")
        print(f"command: {' '.join(command)}")
        print(f"lkh_path: {self.lkh_path}")
        print(f"par_path: {par_path}")
        print(f"vrp_path: {vrp_path}")
        print(f"tour_path: {out_path}")
        print(
            "vehicles: "
            f"safe={safe_vehicles}, min_required={min_required_vehicles}, max_feasible={max_feasible_vehicles}"
        )
        print("=" * 80)

    def _log_process_result(self, process_result: subprocess.CompletedProcess) -> None:
        returncode = process_result.returncode
        stdout = process_result.stdout or ""
        stderr = process_result.stderr or ""
        classification = self._classify_failure(returncode, stdout, stderr)

        print("\n" + "=" * 80)
        print("[LKH trace] process result")
        print(f"returncode: {returncode} ({self._format_returncode_as_hex(returncode)})")
        print(f"classification: {classification}")

        stdout_head, stdout_tail = self._extract_head_tail_lines(stdout, line_count=50)
        stderr_head, stderr_tail = self._extract_head_tail_lines(stderr, line_count=50)
        print(f"stdout_head_50: {stdout_head}")
        print(f"stdout_tail_50: {stdout_tail}")
        print(f"stderr_head_50: {stderr_head}")
        print(f"stderr_tail_50: {stderr_tail}")

        if classification == "lkh_process_crash":
            print("判定: LKH 本体/运行环境崩溃（Windows 异常退出码）")
        elif classification == "data_or_parameter_error":
            print("判定: 更可能是数据/参数问题（建议检查 DIMENSION/DEMAND/CAPACITY/VEHICLES）")
        else:
            print("判定: 非零退出但原因不明确，建议结合手工运行与多输入对比排查")
        print("=" * 80 + "\n")

    def _run_lkh_once(self, par_path: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.lkh_path, par_path],
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
        )

    @staticmethod
    def _calculate_retry_vehicles(
        safe_vehicles: int, min_required_vehicles: int, max_feasible_vehicles: int
    ) -> int:
        return min(max_feasible_vehicles, max(safe_vehicles + 1, min_required_vehicles))

    @staticmethod
    def _deduplicate_preserve_order(values: List[int]) -> List[int]:
        seen = set()
        deduped = []
        for v in values:
            iv = int(v)
            if iv in seen:
                continue
            seen.add(iv)
            deduped.append(iv)
        return deduped

    @staticmethod
    def _sanitize_reduced_instance(
        distances: np.ndarray,
        demands: np.ndarray,
        fixed_edges_lkh: List[Tuple[int, int]],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            return None
        n = distances.shape[0]
        if demands.ndim != 1 or demands.shape[0] != n:
            return None
        if n <= 1:
            return None
        if not np.isfinite(distances).all() or not np.isfinite(demands).all():
            return None

        repaired_dist = np.asarray(np.round(distances), dtype=np.int64)
        repaired_demands = np.asarray(np.round(demands), dtype=np.int64)

        # CVRP depot demand must be 0; customer demand must be non-negative.
        repaired_demands = np.maximum(repaired_demands, 0)
        repaired_demands[0] = 0

        repaired_dist = np.maximum(repaired_dist, 0)
        np.fill_diagonal(repaired_dist, 0)

        fixed_zero_mask = np.zeros((n, n), dtype=bool)
        for u, v in fixed_edges_lkh:
            ui, vi = int(u) - 1, int(v) - 1
            if 0 <= ui < n and 0 <= vi < n and ui != vi:
                fixed_zero_mask[ui, vi] = True
                fixed_zero_mask[vi, ui] = True

        # Keep only fixed edges as zero-cost off-diagonal entries.
        offdiag = ~np.eye(n, dtype=bool)
        bad_zero = offdiag & (repaired_dist == 0) & (~fixed_zero_mask)
        repaired_dist[bad_zero] = 1

        # Ensure every customer node has at least one positive-cost candidate edge.
        for i in range(1, n):
            row_has_candidate = np.any(repaired_dist[i, np.arange(n) != i] > 0)
            if not row_has_candidate:
                return None

        return repaired_dist, repaired_demands

    def _extract_expected_customers_from_tours(self, tours: List[List[int]]) -> List[int]:
        max_customer_id = len(self.node_xy) - 1
        ordered = []
        seen = set()
        for route in tours:
            for node in route:
                nid = int(node)
                if nid <= 0 or nid > max_customer_id:
                    continue
                if nid in seen:
                    continue
                seen.add(nid)
                ordered.append(nid)
        return ordered

    @staticmethod
    def _is_valid_flat_tour(flat_tour: List[int], expected_customers: List[int]) -> bool:
        if not expected_customers:
            return False
        expected = set(int(x) for x in expected_customers)
        seen = set()
        for node in flat_tour:
            nid = int(node)
            if nid == 0:
                continue
            if nid not in expected:
                return False
            if nid in seen:
                return False
            seen.add(nid)
        return seen == expected

    @staticmethod
    def _validate_lkh_inputs(
        n: int,
        distances: np.ndarray,
        demands: np.ndarray,
        fixed_edges: List[Tuple[int, int]],
        vehicles: int,
        capacity: int,
    ) -> bool:
        if n <= 1:
            return False
        if capacity <= 0:
            return False
        if distances.shape != (n, n):
            return False
        if demands.shape != (n,):
            return False
        if not np.isfinite(distances).all() or not np.isfinite(demands).all():
            return False
        if (distances < 0).any() or (demands < 0).any():
            return False
        if not np.all(np.diag(distances) == 0):
            return False
        if int(demands[0]) != 0:
            return False
        if vehicles < 1 or vehicles > (n - 1):
            return False

        for u, v in fixed_edges:
            ui, vi = int(u), int(v)
            if ui < 1 or ui >= (n + 1) or vi < 1 or vi >= (n + 1) or ui == vi:
                return False
        return True

    def run_fsta_reoptimization(self, tours: List[List[int]], destroyed_nodes: Set[int]):
        """
        步骤 2 & 3：图压缩与 LKH 求解 (严密对齐 Appendix B.1.5)
        """
        expected_customers = self._extract_expected_customers_from_tours(tours)

        def _fallback_tour(tours: List[List[int]]) -> List[int]:
            flat = []
            max_customer_id = len(self.node_xy) - 1
            for route in tours:
                if not route:
                    continue
                flat.extend(
                    int(n)
                    for n in route
                    if 0 < int(n) <= max_customer_id
                )
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

        # 可能同时出现在 destroyed_nodes 和 segment endpoint，需去重避免图结构异常。
        new_nodes = self._deduplicate_preserve_order(new_nodes)
                
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

        repaired = self._sanitize_reduced_instance(distances, demands, fixed_edges_lkh)
        if repaired is None:
            return _fallback_tour(tours)
        distances, demands = repaired


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
            # 👑 完美动态车辆分配：既满足最低运力，又不超过节点数和最大限制
            max_feasible_vehicles = max(1, num_new_nodes - 1)
            safe_vehicles = max(min_required_vehicles, min(self.max_vehicles, max_feasible_vehicles))
            assert safe_vehicles <= max_feasible_vehicles, (
                f"safe_vehicles ({safe_vehicles}) exceeded "
                f"max_feasible_vehicles ({max_feasible_vehicles})"
            )

            if not self._validate_lkh_inputs(
                n=num_new_nodes,
                distances=distances,
                demands=demands,
                fixed_edges=fixed_edges_lkh,
                vehicles=safe_vehicles,
                capacity=self.capacity,
            ):
                print("[LKH trace] invalid reduced instance detected before LKH call; fallback applied.")
                return _fallback_tour(tours)

            self._write_explicit_vrp(vrp_path, num_new_nodes, distances, demands, fixed_edges_lkh)
            self._write_par(par_path, vrp_path, out_path, vehicles=safe_vehicles)
            run_command = [self.lkh_path, par_path]
            self._log_lkh_context(
                run_id=run_id,
                command=run_command,
                par_path=par_path,
                vrp_path=vrp_path,
                out_path=out_path,
                safe_vehicles=safe_vehicles,
                min_required_vehicles=min_required_vehicles,
                max_feasible_vehicles=max_feasible_vehicles,
            )

            process_result = self._run_lkh_once(par_path)
            if process_result.returncode != 0:
                self._log_process_result(process_result)
                classification = self._classify_failure(
                    process_result.returncode,
                    process_result.stdout or "",
                    process_result.stderr or "",
                )
                if classification == "data_or_parameter_error":
                    retry_vehicles = self._calculate_retry_vehicles(
                        safe_vehicles=safe_vehicles,
                        min_required_vehicles=min_required_vehicles,
                        max_feasible_vehicles=max_feasible_vehicles,
                    )
                    if retry_vehicles > safe_vehicles:
                        print(
                            f"[LKH trace] data/parameter-like failure detected, retry once with "
                            f"relaxed VEHICLES={retry_vehicles}"
                        )
                        self._write_par(par_path, vrp_path, out_path, vehicles=retry_vehicles)
                        process_result = self._run_lkh_once(par_path)
                        if process_result.returncode == 0:
                            print("[LKH trace] retry succeeded with relaxed VEHICLES.")
                        else:
                            self._log_process_result(process_result)

            # 👑 核心防御：如果 LKH 底层崩溃，绝对不能交白卷！必须触发平滑回退
            if process_result.returncode != 0:
                return _fallback_tour(tours)

            # 5. 读取与无损解码
            lkh_tour_new = self._parse_tour(out_path)
            if not lkh_tour_new:
                print("[LKH trace] empty/invalid TOUR_FILE content; fallback applied.")
                return _fallback_tour(tours)

            recovered = self._recover_solution(lkh_tour_new, new_to_global, segments)
            if not self._is_valid_flat_tour(recovered, expected_customers):
                print("[LKH trace] recovered tour failed validation; fallback applied.")
                return _fallback_tour(tours)

            return recovered

        except subprocess.TimeoutExpired as e:
            print("\n" + "!"*60)
            print("LKH-3 陷入死循环，已被 Python 强制狙击！")
            print("让我们看看它死前到底卡在哪一步了：")
            print(f"{e.stdout}")
            print(f"顺便检查一下容量参数对不对：Capacity = {self.capacity}")
            print("!"*60 + "\n")
            return _fallback_tour(tours)
        except (FileNotFoundError, PermissionError, OSError) as e:
            print("\n" + "="*60)
            print("💀 LKH-3 不可执行或缺失，已成功拦截，正在平滑回退。")
            print(f"👉 错误原因: {e}")
            print(f"👉 当前 LKH 路径: {self.lkh_path}")
            print("="*60 + "\n")
            return _fallback_tour(tours)
            
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
                    if in_tour:
                        for token in line.split():
                            try:
                                tour.append(int(token))
                            except ValueError:
                                continue
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
