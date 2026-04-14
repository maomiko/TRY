import argparse
import copy
import os
import random
import sys
import traceback

import numpy as np
import torch
import yaml

from src.search_sa import Search


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_minimal_params(config: dict, seed: int, nb_iterations: int):
    env_params = copy.deepcopy(config.get("env_params", {}))
    tester_params = copy.deepcopy(config.get("tester_params", {}))

    env_params["num_processes"] = 1
    env_params["model_params"] = copy.deepcopy(config.get("model_params", {}))

    tester_params["seed"] = int(seed)
    tester_params["deterministic"] = True
    tester_params["use_cuda"] = False
    tester_params["use_baseline_destroy"] = True
    tester_params["model_load"] = []
    tester_params["nb_instances"] = 1
    tester_params["aug_factor"] = 1
    tester_params["rollout_size"] = 1
    tester_params["nb_iterations"] = int(nb_iterations)
    tester_params["max_runtime"] = 0

    return env_params, tester_params


def validate_solution(result: dict, expected_customer_count: int):
    if result.get("solution") is None:
        return False, "solution is None"

    cost = float(result.get("cost", float("inf")))
    if (not np.isfinite(cost)) or cost <= 1e-12:
        return False, f"invalid cost: {cost}"

    tours = result["solution"].getTourList()
    if not tours:
        return False, "empty tours"

    visited = []
    for route in tours:
        for node in route:
            nid = int(node)
            if nid <= 0:
                return False, f"invalid node id in solution: {nid}"
            visited.append(nid)

    if len(visited) != expected_customer_count:
        return (
            False,
            f"customer count mismatch: visited={len(visited)} expected={expected_customer_count}",
        )

    if len(set(visited)) != expected_customer_count:
        return False, "duplicate or missing customers detected"

    expected = set(range(1, expected_customer_count + 1))
    if set(visited) != expected:
        return False, "visited set does not match expected customer set"

    solution_cost = float(result["solution"].totalCosts)
    if abs(solution_cost - cost) > 1e-6:
        return False, f"cost mismatch: result={cost} solution={solution_cost}"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Minimal deterministic single-instance repro for annealing crash/ghost-solution guard validation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/reproduce/label_gen_cvrp100.yaml",
        help="Base yaml config path",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Fixed random seed")
    parser.add_argument(
        "--instance-idx", type=int, default=0, help="Index of the single instance in dataset"
    )
    parser.add_argument(
        "--nb-iterations",
        type=int,
        default=5,
        help="SA iterations for minimal repro",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"❌ config not found: {args.config}")
        return 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        config = load_config(args.config)
        env_params, tester_params = build_minimal_params(
            config=config, seed=args.seed, nb_iterations=args.nb_iterations
        )

        search = Search(env_params, tester_params)
        search._load_test_dataset()

        _, node_xy, _, _ = search._get_instance_raw_data(args.instance_idx)
        expected_customer_count = int(len(node_xy))

        result = search._solve_one_instance(args.instance_idx)
        ok, message = validate_solution(result, expected_customer_count)

        print("=" * 80)
        print("Minimal Repro Result")
        print(f"seed={args.seed}, instance_idx={args.instance_idx}, nb_iterations={args.nb_iterations}")
        print(f"cost={float(result['cost']):.6f}, iterations={int(result['nb_iterations'])}")
        print(f"validation={ok}, message={message}")
        print("=" * 80)

        if not ok:
            return 1

        print("✅ PASS: no crash surfaced to caller and no 0.00/invalid ghost solution detected.")
        return 0
    except Exception as e:
        print(f"❌ repro script crashed: {e}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
