import numpy as np
import os
import tempfile
import subprocess
import uuid
import shutil
import stat
from typing import List, Set, Tuple, Dict, Optional

UINT32_MODULUS = 1 << 32
UINT32_MASK = 0xFFFFFFFF
WINDOWS_ACCESS_VIOLATION = -1073741819  # 0xC0000005
WINDOWS_STACK_BUFFER_OVERRUN = -1073740791  # 0xC0000409
WINDOWS_ACCESS_VIOLATION_U32 = 0xC0000005
WINDOWS_STACK_BUFFER_OVERRUN_U32 = 0xC0000409
MIN_MAX_CANDIDATES = 20
MAX_CANDIDATES_BUFFER = 5

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
        lkh_trace: bool = False,
    ):
        self.node_xy = node_xy
        self.node_demand = node_demand
        self.capacity = capacity
        self.lkh_path = self._resolve_lkh_path(lkh_path)
        self.max_vehicles = max_vehicles
        self.timeout_sec = max(1, int(timeout_sec))
        self._disable_lkh = False
        self.lkh_trace = lkh_trace

    def _trace_print(self, message: str) -> None:
        """Conditionally emit LKH debugging traces when lkh_trace is enabled."""
        if self.lkh_trace:
            print(message)

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

    def _normalize_tours(self, tours: List[List[int]]) -> List[List[int]]:
        max_customer_id = len(self.node_xy) - 1
        normalized = []
        for route in tours:
            clean_route = []
            for node in route:
                nid = int(node)
                if 0 < nid <= max_customer_id:
                    clean_route.append(nid)
            if clean_route:
                normalized.append(clean_route)
        return normalized

    def _extract_segments(self, tours: List[List[int]], destroyed_nodes: Set[int]) -> List[List[int]]:
        segments = []
        for tour in tours:
            current_segment = []
            for node in tour:
                nid = int(node)
                if nid in destroyed_nodes:
                    if current_segment:
                        segments.append(current_segment)
                        current_segment = []
                    continue
                current_segment.append(nid)
            if current_segment:
                segments.append(current_segment)
        return segments

    @staticmethod
    def _split_segment_demand(total_demand: int) -> Tuple[int, int]:
        left = int(total_demand) // 2
        right = int(total_demand) - left
        return left, right

    @staticmethod
    def _format_returncode_as_hex(returncode: int) -> str:
        """Format return code as hex; negative values are mapped to unsigned 32-bit form (common on Windows)."""
        if returncode < 0:
            return hex(UINT32_MODULUS + returncode)
        return hex(returncode)

    def _classify_failure(self, returncode: int, stdout: str, stderr: str) -> str:
        returncode_unsigned = int(returncode) & UINT32_MASK
        if (
            returncode in (WINDOWS_ACCESS_VIOLATION, WINDOWS_STACK_BUFFER_OVERRUN)
            or returncode_unsigned in (WINDOWS_ACCESS_VIOLATION_U32, WINDOWS_STACK_BUFFER_OVERRUN_U32)
        ):
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
            "no candidates",
        )
        if any(k in msg for k in data_keywords):
            return "data_or_parameter_error"
        return "unknown_nonzero_exit"

    @staticmethod
    def _is_no_candidates_failure(stdout: str, stderr: str) -> bool:
        msg = f"{stdout}\n{stderr}".lower()
        return "no candidates" in msg

    @staticmethod
    def _recommended_max_candidates(vehicles: int) -> int:
        return max(MIN_MAX_CANDIDATES, int(vehicles) + MAX_CANDIDATES_BUFFER)

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
        self._trace_print("\n" + "=" * 80)
        self._trace_print("[LKH trace] minimal reproducible context")
        self._trace_print(f"run_id: {run_id}")
        self._trace_print(f"command: {' '.join(command)}")
        self._trace_print(f"lkh_path: {self.lkh_path}")
        self._trace_print(f"par_path: {par_path}")
        self._trace_print(f"vrp_path: {vrp_path}")
        self._trace_print(f"tour_path: {out_path}")
        self._trace_print(
            "vehicles: "
            f"safe={safe_vehicles}, min_required={min_required_vehicles}, max_feasible={max_feasible_vehicles}"
        )
        self._trace_print("=" * 80)

    def _log_process_result(self, process_result: subprocess.CompletedProcess) -> None:
        returncode = process_result.returncode
        stdout = process_result.stdout or ""
        stderr = process_result.stderr or ""
        classification = self._classify_failure(returncode, stdout, stderr)

        self._trace_print("\n" + "=" * 80)
        self._trace_print("[LKH trace] process result")
        self._trace_print(f"returncode: {returncode} ({self._format_returncode_as_hex(returncode)})")
        self._trace_print(f"classification: {classification}")

        stdout_head, stdout_tail = self._extract_head_tail_lines(stdout, line_count=50)
        stderr_head, stderr_tail = self._extract_head_tail_lines(stderr, line_count=50)
        self._trace_print(f"stdout_head_50: {stdout_head}")
        self._trace_print(f"stdout_tail_50: {stdout_tail}")
        self._trace_print(f"stderr_head_50: {stderr_head}")
        self._trace_print(f"stderr_tail_50: {stderr_tail}")

        if classification == "lkh_process_crash":
            self._trace_print("判定: LKH 本体/运行环境崩溃（Windows 异常退出码）")
        elif classification == "data_or_parameter_error":
            self._trace_print("判定: 更可能是数据/参数问题（建议检查 DIMENSION/DEMAND/CAPACITY/VEHICLES）")
        else:
            self._trace_print("判定: 非零退出但原因不明确，建议结合手工运行与多输入对比排查")
        self._trace_print("=" * 80 + "\n")

    def _run_lkh_once(self, par_path: str, timeout_sec: Optional[int] = None) -> subprocess.CompletedProcess:
        effective_timeout = self.timeout_sec if timeout_sec is None else max(1, int(timeout_sec))
        return subprocess.run(
            [self.lkh_path, par_path],
            capture_output=True,
            text=True,
            timeout=effective_timeout,
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
        expected_customers = self._extract_expected_customers_from_tours(tours)
        normalized_tours = self._normalize_tours(tours)
        valid_customer_set = set(expected_customers)

        def _fallback_tour(input_tours: List[List[int]]) -> List[int]:
            flat = []
            seen = set()
            for route in self._normalize_tours(input_tours):
                for node in route:
                    if node in seen:
                        continue
                    seen.add(node)
                    flat.append(node)
            if not flat:
                flat = [int(n) for n in expected_customers if int(n) > 0]
            return flat

        if self._disable_lkh:
            return _fallback_tour(normalized_tours)
        if not normalized_tours or not expected_customers:
            return _fallback_tour(normalized_tours)

        destroyed = {int(n) for n in destroyed_nodes if int(n) in valid_customer_set}
        segments = self._extract_segments(normalized_tours, destroyed)

        new_nodes = [0]
        segment_endpoints = []
        endpoint_demand_override: Dict[int, int] = {}

        for node in expected_customers:
            if node in destroyed:
                new_nodes.append(node)

        for seg in segments:
            if len(seg) == 1:
                new_nodes.append(seg[0])
                continue
            start, end = int(seg[0]), int(seg[-1])
            new_nodes.append(start)
            new_nodes.append(end)
            segment_endpoints.append((start, end))

            total_seg_demand = int(np.sum(self.node_demand[np.asarray(seg, dtype=np.int64)]))
            start_demand, end_demand = self._split_segment_demand(total_seg_demand)
            endpoint_demand_override[start] = start_demand
            endpoint_demand_override[end] = end_demand

        new_nodes = self._deduplicate_preserve_order(new_nodes)
        num_new_nodes = len(new_nodes)
        if num_new_nodes <= 1:
            return _fallback_tour(normalized_tours)

        global_to_new = {g_id: n_id + 1 for n_id, g_id in enumerate(new_nodes)}
        new_to_global = {n_id + 1: g_id for n_id, g_id in enumerate(new_nodes)}

        node_coords = self.node_xy[np.asarray(new_nodes, dtype=np.int64)]
        distances = np.linalg.norm(node_coords[:, None, :] - node_coords[None, :, :], axis=2)
        distances = np.asarray(np.rint(distances * 10000), dtype=np.int64)
        np.fill_diagonal(distances, 0)

        demands = np.zeros(num_new_nodes, dtype=np.int64)
        for i, g_id in enumerate(new_nodes):
            if g_id == 0:
                demands[i] = 0
                continue
            if g_id in endpoint_demand_override:
                demands[i] = int(endpoint_demand_override[g_id])
            else:
                demands[i] = int(self.node_demand[g_id])

        fixed_edges_lkh = []
        for start_g, end_g in segment_endpoints:
            if start_g not in global_to_new or end_g not in global_to_new:
                continue
            s_n, e_n = global_to_new[start_g], global_to_new[end_g]
            distances[s_n - 1, e_n - 1] = 0
            distances[e_n - 1, s_n - 1] = 0
            fixed_edges_lkh.append((s_n, e_n))

        repaired = self._sanitize_reduced_instance(distances, demands, fixed_edges_lkh)
        if repaired is None:
            return _fallback_tour(normalized_tours)
        distances, demands = repaired

        total_demand = int(np.sum(demands))
        if self.capacity <= 0:
            return _fallback_tour(normalized_tours)
        min_required_vehicles = max(1, int(np.ceil(total_demand / self.capacity)))
        if num_new_nodes - 1 < min_required_vehicles:
            return _fallback_tour(normalized_tours)

        temp_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
        run_id = uuid.uuid4().hex
        vrp_path = os.path.join(temp_dir, f"fsta_problem_{run_id}.vrp")
        par_path = os.path.join(temp_dir, f"fsta_params_{run_id}.par")
        out_path = os.path.join(temp_dir, f"fsta_output_{run_id}.tour")

        max_feasible_vehicles = max(1, num_new_nodes - 1)
        safe_vehicles = max(min_required_vehicles, min(self.max_vehicles, max_feasible_vehicles))

        try:
            if not self._validate_lkh_inputs(
                n=num_new_nodes,
                distances=distances,
                demands=demands,
                fixed_edges=fixed_edges_lkh,
                vehicles=safe_vehicles,
                capacity=self.capacity,
            ):
                self._trace_print("[LKH trace] invalid reduced instance detected before LKH call; fallback applied.")
                return _fallback_tour(normalized_tours)

            self._write_explicit_vrp(vrp_path, num_new_nodes, distances, demands, fixed_edges_lkh)
            base_max_candidates = self._recommended_max_candidates(safe_vehicles)
            self._write_par(
                par_path,
                vrp_path,
                out_path,
                vehicles=safe_vehicles,
                max_candidates=base_max_candidates,
            )
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
                    if self._is_no_candidates_failure(
                        process_result.stdout or "",
                        process_result.stderr or "",
                    ):
                        retry_max_candidates = max_feasible_vehicles
                        if retry_max_candidates > base_max_candidates:
                            self._write_par(
                                par_path,
                                vrp_path,
                                out_path,
                                vehicles=safe_vehicles,
                                max_candidates=retry_max_candidates,
                            )
                            process_result = self._run_lkh_once(par_path)
                            if process_result.returncode != 0:
                                self._log_process_result(process_result)

                    if process_result.returncode != 0:
                        retry_vehicles = self._calculate_retry_vehicles(
                            safe_vehicles=safe_vehicles,
                            min_required_vehicles=min_required_vehicles,
                            max_feasible_vehicles=max_feasible_vehicles,
                        )
                        if retry_vehicles > safe_vehicles:
                            self._write_par(
                                par_path,
                                vrp_path,
                                out_path,
                                vehicles=retry_vehicles,
                                max_candidates=self._recommended_max_candidates(retry_vehicles),
                            )
                            process_result = self._run_lkh_once(par_path)
                            if process_result.returncode != 0:
                                self._log_process_result(process_result)

            if process_result.returncode != 0:
                final_classification = self._classify_failure(
                    process_result.returncode,
                    process_result.stdout or "",
                    process_result.stderr or "",
                )
                if final_classification == "lkh_process_crash":
                    self._disable_lkh = True
                return _fallback_tour(normalized_tours)

            lkh_tour_new = self._parse_tour(out_path)
            if not lkh_tour_new:
                self._disable_lkh = True
                return _fallback_tour(normalized_tours)

            recovered = self._recover_solution(lkh_tour_new, new_to_global, segments)
            if not self._is_valid_flat_tour(recovered, expected_customers):
                return _fallback_tour(normalized_tours)

            return recovered
        except subprocess.TimeoutExpired:
            return _fallback_tour(normalized_tours)
        except (FileNotFoundError, PermissionError, OSError):
            return _fallback_tour(normalized_tours)
        finally:
            for f_path in [vrp_path, par_path, out_path]:
                if os.path.exists(f_path):
                    try:
                        os.remove(f_path)
                    except OSError:
                        pass

    def _recover_solution(self, lkh_tour_new: List[int], new_to_global: Dict[int, int], segments: List[List[int]]) -> List[int]:
        raw_global = []
        for reduced_id in lkh_tour_new:
            rid = int(reduced_id)
            if rid == 1:
                raw_global.append(0)
                continue
            if rid in new_to_global:
                raw_global.append(int(new_to_global[rid]))
            else:
                raw_global.append(0)

        seg_lookup: Dict[Tuple[int, int], List[int]] = {}
        for seg in segments:
            if len(seg) > 1:
                seg_lookup[(int(seg[0]), int(seg[-1]))] = [int(x) for x in seg]
                seg_lookup[(int(seg[-1]), int(seg[0]))] = [int(x) for x in reversed(seg)]

        expanded = []
        i = 0
        while i < len(raw_global):
            curr = int(raw_global[i])
            if curr == 0:
                if expanded and expanded[-1] != 0:
                    expanded.append(0)
                i += 1
                continue

            if i + 1 < len(raw_global):
                nxt = int(raw_global[i + 1])
                pair = (curr, nxt)
                if pair in seg_lookup:
                    expanded.extend(seg_lookup[pair])
                    i += 2
                    continue

            expanded.append(curr)
            i += 1

        num_customers = len(self.node_xy) - 1
        cleaned = []
        for node in expanded:
            nid = int(node)
            if 0 < nid <= num_customers:
                cleaned.append(nid)
            elif cleaned and cleaned[-1] != 0:
                cleaned.append(0)

        while cleaned and cleaned[0] == 0:
            cleaned.pop(0)
        while cleaned and cleaned[-1] == 0:
            cleaned.pop()
        return cleaned
    
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
    def _write_par(self, par_path, vrp_path, out_path, vehicles=50, max_candidates=None):
        with open(par_path, 'w') as f:
            f.write(f"PROBLEM_FILE = {vrp_path}\nTOUR_FILE = {out_path}\n")
            f.write(f"RUNS = 1\nTIME_LIMIT = {max(1, int(self.timeout_sec))}\nTRACE_LEVEL = 0\n")
            
            # 👑 突破“假车场黑洞”：视距必须穿透所有的假车场，再额外看到 5 个真实客户！
            cands = self._recommended_max_candidates(vehicles)
            if max_candidates is not None:
                cands = max(cands, int(max_candidates))
            f.write(f"MAX_CANDIDATES = {cands}\n") 
            
            f.write(f"VEHICLES = {max(1, int(vehicles))}\n")
