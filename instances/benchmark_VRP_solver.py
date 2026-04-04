"""
benchmark_VRP_solver.py
========================
Multi-method CVRP benchmark harness.

Methods benchmarked
-------------------
1. clarke_wright       -- original CW baseline (fast, no 2-opt)
2. classical_best      -- seed portfolio + 2-opt + repair (no quantum)
3. hybrid_qaoa         -- seed portfolio + 2-opt + repair + local QAOA

Output
------
benchmark_output/benchmark_results.json   -- rich JSON per instance
benchmark_output/benchmark_comparison.json -- summary comparison table
Individual route plots per instance.

Usage
-----
    cd instances/
    python benchmark_VRP_solver.py [--json instances.json] [--no-quantum]
"""

import sys
import json
import time
import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to allow absolute imports of shared modules
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    parse_instance_dict,
    build_distance_matrix,
    route_distance,
    route_load,
    total_solution_distance,
    validate_solution,
)
from decomposition import (
    clarke_wright as cw_decomp,
    generate_seed_portfolio,
    best_valid_candidate,
)
from local_search import cleanup_solution
from repair import repair_routes
from hybrid_solver import HybridSolver


# ---------------------------------------------------------------------------
# Method 1: Clarke-Wright baseline (legacy, no 2-opt)
# ---------------------------------------------------------------------------

def run_clarke_wright_baseline(inst):
    """
    Pure Clarke-Wright with no post-processing.
    Reproduces the original benchmark_VRP_solver.py behaviour.
    """
    nodes = inst["nodes"]
    demands = inst["demands"]
    capacity = inst["capacity"]
    num_vehicles = inst["num_vehicles"]
    num_customers = inst["num_customers"]
    D = build_distance_matrix(nodes)

    t0 = time.perf_counter()
    result = cw_decomp(nodes, demands, capacity, num_vehicles)
    runtime = time.perf_counter() - t0

    routes = result["routes"]
    dist = total_solution_distance(routes, D)
    valid = validate_solution(routes, demands, capacity, num_customers, num_vehicles)

    return {
        "method": "clarke_wright",
        "instance_id": inst["instance_id"],
        "total_distance": round(dist, 6),
        "runtime_s": round(runtime, 6),
        "valid": valid,
        "vehicles_used": len(routes),
        "vehicles_allowed": num_vehicles,
        "cleanup_gain": 0.0,
        "repair_gain": 0.0,
        "quantum_ran": False,
        "qaoa_gain": 0.0,
        "route_details": [
            {
                "vehicle": i + 1,
                "route": [0] + r + [0],
                "load": route_load(r, demands),
                "distance": round(route_distance(r, D), 6),
            }
            for i, r in enumerate(routes)
        ],
    }


# ---------------------------------------------------------------------------
# Method 2: Classical best (seed portfolio + 2-opt + repair)
# ---------------------------------------------------------------------------

def run_classical_best(inst, verbose=False):
    """Seed portfolio + 2-opt + inter-route repair. No quantum."""
    nodes = inst["nodes"]
    demands = inst["demands"]
    capacity = inst["capacity"]
    num_vehicles = inst["num_vehicles"]
    num_customers = inst["num_customers"]
    D = build_distance_matrix(nodes)

    t0 = time.perf_counter()

    # Stage A: seeds
    candidates = generate_seed_portfolio(nodes, demands, capacity, num_vehicles)

    # Stage B: 2-opt on all
    cleaned = []
    for c in candidates:
        routes = cleanup_solution(c["routes"], D)
        d = total_solution_distance(routes, D)
        valid = validate_solution(routes, demands, capacity, num_customers, num_vehicles)
        cleaned.append({**c, "routes": routes, "total_dist": round(d, 6), "valid": valid})

    best_after_cleanup = best_valid_candidate(cleaned)
    dist_after_cleanup = best_after_cleanup["total_dist"]
    dist_after_seed = best_valid_candidate(candidates)["total_dist"]
    cleanup_gain = round(dist_after_seed - dist_after_cleanup, 6)

    # Stage C: repair
    routes = copy.deepcopy(best_after_cleanup["routes"])
    routes, repair_gain = repair_routes(routes, D, demands, capacity, nodes)
    routes = cleanup_solution(routes, D)

    runtime = time.perf_counter() - t0
    dist = total_solution_distance(routes, D)
    valid = validate_solution(routes, demands, capacity, num_customers, num_vehicles)

    return {
        "method": "classical_best",
        "instance_id": inst["instance_id"],
        "total_distance": round(dist, 6),
        "runtime_s": round(runtime, 6),
        "valid": valid,
        "vehicles_used": len(routes),
        "vehicles_allowed": num_vehicles,
        "seed_best_method": best_after_cleanup["method"],
        "cleanup_gain": cleanup_gain,
        "repair_gain": repair_gain,
        "quantum_ran": False,
        "qaoa_gain": 0.0,
        "route_details": [
            {
                "vehicle": i + 1,
                "route": [0] + r + [0],
                "load": route_load(r, demands),
                "distance": round(route_distance(r, D), 6),
            }
            for i, r in enumerate(routes)
        ],
    }


# ---------------------------------------------------------------------------
# Method 3: Hybrid QAOA
# ---------------------------------------------------------------------------

def run_hybrid_qaoa(inst, max_local_qaoa_qubits=25, verbose=False):
    """Full hybrid: seed portfolio + 2-opt + repair + local QAOA."""
    solver = HybridSolver(
        inst,
        max_local_qaoa_qubits=max_local_qaoa_qubits,
        verbose=verbose,
    )
    r = solver.solve(run_quantum=True)

    routes = r["routes_final"]
    D = build_distance_matrix(inst["nodes"])

    # Build qubit/gate info from QAOA meta
    qaoa_meta = r.get("all_meta_qaoa", [])
    qubit_estimates = [m.get("n_qubits", 0) for m in qaoa_meta if isinstance(m, dict)]
    gate_estimates = [m.get("n_gates", 0) for m in qaoa_meta if isinstance(m, dict)]
    neighborhood_sizes = [m.get("neighborhood_size", 0) for m in qaoa_meta if isinstance(m, dict)]
    total_elapsed = sum(m.get("elapsed_s", 0) for m in qaoa_meta if isinstance(m, dict))

    return {
        "method": "hybrid_qaoa",
        "instance_id": inst["instance_id"],
        "total_distance": r["total_dist_final"],
        "runtime_s": r["wall_time_s"],
        "valid": r["valid_final"],
        "vehicles_used": r["vehicles_used"],
        "vehicles_allowed": r["vehicles_allowed"],
        "seed_best_method": r["stages"]["seed_best_method"],
        "cleanup_gain": r["stages"]["cleanup_gain"],
        "repair_gain": r["stages"]["repair_gain"],
        "quantum_ran": r["quantum_ran"],
        "qaoa_gain": r["stages"]["qaoa_gain"],
        "stages": r["stages"],
        "qubit_estimates": qubit_estimates,
        "max_qubits": max(qubit_estimates, default=0),
        "gate_estimates": gate_estimates,
        "neighborhood_sizes": neighborhood_sizes,
        "qaoa_elapsed_s": round(total_elapsed, 3),
        "route_details": [
            {
                "vehicle": i + 1,
                "route": [0] + route + [0],
                "load": route_load(route, inst["demands"]),
                "distance": round(route_distance(route, D), 6),
            }
            for i, route in enumerate(routes)
        ],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_instance_routes(inst, routes, method, output_path):
    """Plot routes for one instance/method to a PNG file."""
    nodes = inst["nodes"]
    depot = nodes[0]
    customers = nodes[1:]
    D = build_distance_matrix(nodes)

    dist = total_solution_distance(routes, D)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(depot[0], depot[1], c="black", marker="X", s=220,
               label="Depot", zorder=5)
    ax.text(depot[0] + 0.4, depot[1] + 0.4, "0", fontsize=11)

    xs = [c[0] for c in customers]
    ys = [c[1] for c in customers]
    ax.scatter(xs, ys, c="steelblue", s=60, zorder=4)
    for idx, (x, y) in enumerate(customers, start=1):
        ax.text(x + 0.4, y + 0.4, str(idx), fontsize=8)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ridx, route in enumerate(routes, start=1):
        pts = [depot] + [customers[c - 1] for c in route] + [depot]
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], linewidth=2.0,
                color=colors[(ridx - 1) % len(colors)],
                label=f"R{ridx}: {route}", zorder=2)

    ax.set_title(f"{inst['instance_id']} — {method}\ndist={dist:.3f}", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_summary(comparison, output_dir):
    """Bar chart comparing methods across instances."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    methods = list({r["method"] for r in comparison})
    methods.sort()

    instance_ids = sorted({r["instance_id"] for r in comparison})
    x = np.arange(len(instance_ids))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(12, len(instance_ids)), 6))
    for m_idx, method in enumerate(methods):
        dists = []
        for iid in instance_ids:
            match = [r for r in comparison if r["instance_id"] == iid and r["method"] == method]
            dists.append(match[0]["total_distance"] if match else 0)
        ax.bar(x + m_idx * width, dists, width, label=method)

    label_step = max(1, len(instance_ids) // 12)
    display = [iid if i % label_step == 0 else "" for i, iid in enumerate(instance_ids)]
    ax.set_xticks(x + width)
    ax.set_xticklabels(display, rotation=45, fontsize=8)
    ax.set_title("Method Comparison: Total Distance")
    ax.set_ylabel("Total Distance")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "method_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def load_instances(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [parse_instance_dict(inst) for inst in raw]


def run_benchmark(json_path="setA_random_instances_grouped.json",
                  output_dir="benchmark_output",
                  run_quantum=False,
                  methods=None,
                  max_local_qaoa_qubits=25,
                  verbose=False):
    """
    Full benchmark run.

    Parameters
    ----------
    json_path             : str
    output_dir            : str
    run_quantum           : bool  -- include hybrid QAOA method
    methods               : list[str] or None  -- subset of methods to run
    max_local_qaoa_qubits : int
    verbose               : bool

    Returns
    -------
    (all_results, comparison)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    instances = load_instances(json_path)
    all_results = []
    comparison = []

    if methods is None:
        methods = ["clarke_wright", "classical_best"]
        if run_quantum:
            methods.append("hybrid_qaoa")

    for inst in instances:
        iid = inst["instance_id"]
        print(f"\n--- {iid} ({inst['num_customers']} customers) ---")
        inst_results = []

        for method in methods:
            print(f"  Running {method}...", end=" ", flush=True)
            t0 = time.perf_counter()

            if method == "clarke_wright":
                result = run_clarke_wright_baseline(inst)
            elif method == "classical_best":
                result = run_classical_best(inst, verbose=verbose)
            elif method == "hybrid_qaoa":
                result = run_hybrid_qaoa(inst, max_local_qaoa_qubits=max_local_qaoa_qubits,
                                         verbose=verbose)
            else:
                print(f"Unknown method {method}, skipping.")
                continue

            elapsed = time.perf_counter() - t0
            print(f"dist={result['total_distance']:.3f} valid={result['valid']} ({elapsed:.2f}s)")

            inst_results.append(result)
            comparison.append({
                "instance_id": iid,
                "method": method,
                "total_distance": result["total_distance"],
                "valid": result["valid"],
                "runtime_s": result.get("runtime_s", 0),
                "cleanup_gain": result.get("cleanup_gain", 0),
                "repair_gain": result.get("repair_gain", 0),
                "qaoa_gain": result.get("qaoa_gain", 0),
                "quantum_ran": result.get("quantum_ran", False),
            })

            # Plot routes
            if method in ("clarke_wright", "classical_best", "hybrid_qaoa"):
                routes = [r["route"][1:-1] for r in result["route_details"]]
                img_path = output_path / f"{iid}_{method}_routes.png"
                plot_instance_routes(inst, routes, method, img_path)

        all_results.extend(inst_results)

    # Write JSON outputs
    results_path = output_path / "benchmark_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results: {results_path}")

    cmp_path = output_path / "benchmark_comparison.json"
    with open(cmp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved comparison: {cmp_path}")

    # Plot summary
    try:
        plot_comparison_summary(comparison, output_dir)
        print(f"Saved comparison chart: {output_path / 'method_comparison.png'}")
    except Exception as e:
        print(f"Warning: could not plot comparison: {e}")

    return all_results, comparison


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVRP Multi-method Benchmark Harness")
    parser.add_argument("--json", default="setA_random_instances_grouped.json",
                        help="Path to instances JSON file")
    parser.add_argument("--output", default="benchmark_output",
                        help="Output directory")
    parser.add_argument("--no-quantum", action="store_true",
                        help="Skip hybrid QAOA method")
    parser.add_argument("--methods", nargs="+",
                        choices=["clarke_wright", "classical_best", "hybrid_qaoa"],
                        default=None,
                        help="Methods to run (default: cw + classical_best, + hybrid if --quantum)")
    parser.add_argument("--qubits", type=int, default=25,
                        help="Max local QAOA qubit budget (default 25 -> k=5)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_quantum = not args.no_quantum
    run_benchmark(
        json_path=args.json,
        output_dir=args.output,
        run_quantum=run_quantum,
        methods=args.methods,
        max_local_qaoa_qubits=args.qubits,
        verbose=args.verbose,
    )
