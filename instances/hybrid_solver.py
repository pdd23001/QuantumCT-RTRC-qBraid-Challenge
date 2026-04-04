"""
hybrid_solver.py
================
Top-level orchestration for the hybrid CVRP solver.

Pipeline stages
---------------
A. Seed generation    -- seed portfolio (sweep, CW, k-means)
B. Classical cleanup  -- 2-opt on every route in every candidate
C. Boundary repair    -- inter-route relocate + swap
D. Quantum local      -- anchored QAOA on bounded neighborhoods
E. Reporting          -- metrics for each stage

Usage
-----
    from hybrid_solver import HybridSolver

    solver = HybridSolver(inst, max_local_qaoa_qubits=25)
    result = solver.solve(run_quantum=True)
    print(result)
"""

import time
import copy

from common import (
    build_distance_matrix,
    total_solution_distance,
    validate_solution,
    solution_summary,
)
from decomposition import generate_seed_portfolio, best_valid_candidate
from local_search import cleanup_solution
from repair import repair_routes
from quantum_improve import improve_all_routes_qaoa


class HybridSolver:
    """
    Orchestrates the full hybrid CVRP pipeline for one instance.

    Parameters
    ----------
    inst : dict
        Canonical instance dict from common.parse_instance_dict().
        Keys: instance_id, num_vehicles, capacity, nodes, demands, num_customers.

    max_local_qaoa_qubits : int
        Qubit budget for QAOA local improvement.  k = floor(sqrt(budget)).

    sweep_offsets : int
        Number of angular offsets for shifted sweep.

    kmeans_seeds : list[int] or None
        Seeds for k-means decomposition.

    seed : int
        Master random seed for reproducibility.

    use_three_opt : bool
        Apply 3-opt (slower) in addition to 2-opt.

    repair_max_iter : int
        Maximum inter-route repair iterations.

    qaoa_reps : int
        QAOA circuit depth (layers).

    qaoa_shots : int
        Measurement shots per QAOA run.

    qaoa_restarts : int
        COBYLA random restarts per neighborhood.

    verbose : bool
        Print progress.
    """

    def __init__(self,
                 inst,
                 max_local_qaoa_qubits=25,
                 sweep_offsets=4,
                 kmeans_seeds=None,
                 seed=42,
                 use_three_opt=False,
                 repair_max_iter=50,
                 qaoa_reps=2,
                 qaoa_shots=2048,
                 qaoa_restarts=2,
                 qaoa_cobyla_maxiter=150,
                 verbose=True):

        self.inst = inst
        self.max_local_qaoa_qubits = max_local_qaoa_qubits
        self.sweep_offsets = sweep_offsets
        self.kmeans_seeds = kmeans_seeds or [42, 7, 123]
        self.seed = seed
        self.use_three_opt = use_three_opt
        self.repair_max_iter = repair_max_iter
        self.qaoa_reps = qaoa_reps
        self.qaoa_shots = qaoa_shots
        self.qaoa_restarts = qaoa_restarts
        self.qaoa_cobyla_maxiter = qaoa_cobyla_maxiter
        self.verbose = verbose

        self.nodes = inst["nodes"]
        self.demands = inst["demands"]
        self.capacity = inst["capacity"]
        self.num_vehicles = inst["num_vehicles"]
        self.num_customers = inst["num_customers"]
        self.instance_id = inst["instance_id"]

        self.D = build_distance_matrix(self.nodes)

    def _log(self, msg):
        if self.verbose:
            print(f"[{self.instance_id}] {msg}")

    # ------------------------------------------------------------------
    # Stage A: seed portfolio
    # ------------------------------------------------------------------

    def stage_a_seeds(self):
        """Generate all decomposition candidates."""
        t0 = time.perf_counter()
        candidates = generate_seed_portfolio(
            self.nodes, self.demands, self.capacity, self.num_vehicles,
            sweep_offsets=self.sweep_offsets,
            kmeans_seeds=self.kmeans_seeds,
        )
        rt = time.perf_counter() - t0
        valid = [c for c in candidates if c["valid"]]
        best = best_valid_candidate(candidates)
        self._log(f"Stage A: {len(candidates)} seeds, {len(valid)} valid, "
                  f"best={best['method']} dist={best['total_dist']:.4f} ({rt:.3f}s)")
        return candidates, best

    # ------------------------------------------------------------------
    # Stage B: classical cleanup (2-opt)
    # ------------------------------------------------------------------

    def stage_b_cleanup(self, candidates):
        """Apply 2-opt to all routes in every candidate."""
        t0 = time.perf_counter()
        cleaned = []
        for cand in candidates:
            routes = cleanup_solution(
                cand["routes"], self.D,
                use_three_opt=self.use_three_opt
            )
            d = total_solution_distance(routes, self.D)
            valid = validate_solution(routes, self.demands, self.capacity,
                                      self.num_customers, self.num_vehicles)
            cleaned.append({**cand, "routes": routes, "total_dist": round(d, 6), "valid": valid})
        best = best_valid_candidate(cleaned)
        rt = time.perf_counter() - t0
        self._log(f"Stage B: best after cleanup={best['method']} dist={best['total_dist']:.4f} ({rt:.3f}s)")
        return cleaned, best

    # ------------------------------------------------------------------
    # Stage C: inter-route repair
    # ------------------------------------------------------------------

    def stage_c_repair(self, best_seed):
        """Apply relocate/swap repair on the best seed candidate."""
        t0 = time.perf_counter()
        routes = copy.deepcopy(best_seed["routes"])
        before = total_solution_distance(routes, self.D)

        routes, gain = repair_routes(
            routes, self.D, self.demands, self.capacity, self.nodes,
            max_iter=self.repair_max_iter,
        )

        # Post-repair 2-opt
        routes = cleanup_solution(routes, self.D, use_three_opt=self.use_three_opt)

        after = total_solution_distance(routes, self.D)
        valid = validate_solution(routes, self.demands, self.capacity,
                                  self.num_customers, self.num_vehicles)
        rt = time.perf_counter() - t0
        self._log(f"Stage C: repair gain={before - after:.4f}, "
                  f"dist={after:.4f}, valid={valid} ({rt:.3f}s)")
        return routes, round(before - after, 6)

    # ------------------------------------------------------------------
    # Stage D: local QAOA improvement
    # ------------------------------------------------------------------

    def stage_d_qaoa(self, routes):
        """Apply bounded local QAOA improvement to all routes."""
        t0 = time.perf_counter()
        before = total_solution_distance(routes, self.D)

        improved, gain, meta = improve_all_routes_qaoa(
            routes, self.D,
            max_local_qaoa_qubits=self.max_local_qaoa_qubits,
            strategy="worst_edges",
            reps=self.qaoa_reps,
            shots=self.qaoa_shots,
            restarts=self.qaoa_restarts,
            cobyla_maxiter=self.qaoa_cobyla_maxiter,
            seed=self.seed,
        )

        # Post-QAOA 2-opt
        improved = cleanup_solution(improved, self.D)

        after = total_solution_distance(improved, self.D)
        valid = validate_solution(improved, self.demands, self.capacity,
                                  self.num_customers, self.num_vehicles)
        rt = time.perf_counter() - t0

        qubit_estimates = [m.get("n_qubits", 0) for m in meta]
        self._log(f"Stage D: QAOA gain={before - after:.4f}, dist={after:.4f}, "
                  f"valid={valid}, max_qubits={max(qubit_estimates, default=0)} ({rt:.3f}s)")
        return improved, round(before - after, 6), meta

    # ------------------------------------------------------------------
    # solve()  -- full pipeline
    # ------------------------------------------------------------------

    def solve(self, run_quantum=True):
        """
        Run the full hybrid pipeline and return a comprehensive result dict.

        Parameters
        ----------
        run_quantum : bool
            If False, skip Stage D (useful for classical-only benchmarking).

        Returns
        -------
        dict with keys:
            instance_id
            stages dict with per-stage metrics
            routes_final
            total_dist_final
            valid_final
            quantum_ran
            all_meta_qaoa
        """
        wall_start = time.perf_counter()

        # Stage A
        candidates, best_seed_raw = self.stage_a_seeds()
        dist_seed_raw = best_seed_raw["total_dist"]
        method_seed = best_seed_raw["method"]

        # Stage B
        candidates_clean, best_seed_clean = self.stage_b_cleanup(candidates)
        dist_after_cleanup = best_seed_clean["total_dist"]
        cleanup_gain = round(dist_seed_raw - dist_after_cleanup, 6)

        # Stage C
        routes_repaired, repair_gain = self.stage_c_repair(best_seed_clean)
        dist_after_repair = total_solution_distance(routes_repaired, self.D)

        # Stage D
        quantum_ran = False
        qaoa_meta = []
        qaoa_gain = 0.0
        routes_final = routes_repaired

        if run_quantum:
            quantum_ran = True
            routes_final, qaoa_gain, qaoa_meta = self.stage_d_qaoa(routes_repaired)

        dist_final = total_solution_distance(routes_final, self.D)
        valid_final = validate_solution(routes_final, self.demands, self.capacity,
                                        self.num_customers, self.num_vehicles)

        wall_time = time.perf_counter() - wall_start
        self._log(f"Done: dist={dist_final:.4f}, valid={valid_final}, "
                  f"total_time={wall_time:.2f}s")

        return {
            "instance_id": self.instance_id,
            "stages": {
                "seed_best_method": method_seed,
                "dist_after_seed": dist_seed_raw,
                "dist_after_cleanup": dist_after_cleanup,
                "cleanup_gain": cleanup_gain,
                "dist_after_repair": round(dist_after_repair, 6),
                "repair_gain": repair_gain,
                "dist_after_qaoa": round(dist_final, 6),
                "qaoa_gain": qaoa_gain,
            },
            "routes_final": routes_final,
            "total_dist_final": round(dist_final, 6),
            "valid_final": valid_final,
            "vehicles_used": len(routes_final),
            "vehicles_allowed": self.num_vehicles,
            "quantum_ran": quantum_ran,
            "all_meta_qaoa": qaoa_meta,
            "wall_time_s": round(wall_time, 3),
        }


# ---------------------------------------------------------------------------
# Convenience: run on a list of instances and collect metrics
# ---------------------------------------------------------------------------

def run_hybrid_batch(instances, run_quantum=True, **solver_kwargs):
    """
    Run HybridSolver on a list of parsed instances.

    Parameters
    ----------
    instances    : list[dict]   -- from common.load_instances_json
    run_quantum  : bool
    **solver_kwargs : passed to HybridSolver constructor

    Returns
    -------
    list[dict]   -- one result dict per instance
    """
    results = []
    for inst in instances:
        solver = HybridSolver(inst, **solver_kwargs)
        r = solver.solve(run_quantum=run_quantum)
        results.append(r)
    return results
