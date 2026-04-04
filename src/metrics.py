from __future__ import annotations

from typing import Sequence

from src.config import SolverConfig
from src.io_utils import to_repo_relative
from src.models import ClusterSolveResult, CVRPInstance, SweepCandidate


def build_metrics(
    instance: CVRPInstance,
    config: SolverConfig,
    sweep_candidate: SweepCandidate,
    cluster_results: Sequence[ClusterSolveResult],
    total_distance: float,
    runtime_seconds: float,
    valid: bool,
    validation_errors: Sequence[str],
    output_paths: dict[str, object],
) -> dict[str, object]:
    cluster_qubits = [result.qubits_used for result in cluster_results]
    cluster_gate_counts = [result.gate_count for result in cluster_results]
    providers = [result.provider_name for result in cluster_results if result.provider_name not in {"none", "trivial"}]
    backends = [result.backend_name for result in cluster_results if result.backend_name not in {"none", "trivial"}]

    metrics = {
        "instance_name": instance.instance_name,
        "mode": config.runtime.mode,
        "provider": providers[0] if providers else "local_or_trivial",
        "backend": backends[0] if backends else config.runtime.mode,
        "vehicles": instance.vehicles,
        "capacity": instance.capacity,
        "num_customers": len(instance.customers),
        "sweep_multi_start": config.sweep.multi_start,
        "sweep_direction": sweep_candidate.direction,
        "sweep_offset": sweep_candidate.offset,
        "sweep_score": sweep_candidate.score,
        "clusters": [list(cluster) for cluster in sweep_candidate.clusters],
        "cluster_sizes": [len(cluster) for cluster in sweep_candidate.clusters],
        "cluster_routes": [list(result.route) for result in cluster_results],
        "cluster_distances": [result.route_distance for result in cluster_results],
        "cluster_sample_counts": [result.sample_count for result in cluster_results],
        "cluster_repaired_flags": [result.repaired for result in cluster_results],
        "cluster_qubits": cluster_qubits,
        "cluster_gate_counts": cluster_gate_counts,
        "qaoa_reps": config.qaoa.reps,
        "shots": config.qaoa.shots,
        "optimizer": config.qaoa.optimizer,
        "maxiter": config.qaoa.maxiter,
        "penalties": {
            "row_exactly_one": config.penalties.row_exactly_one,
            "col_exactly_one": config.penalties.col_exactly_one,
        },
        "two_opt_enabled": config.enable_2opt,
        "total_distance": total_distance,
        "runtime_seconds": runtime_seconds,
        "qubits_used": max(cluster_qubits, default=0),
        "gate_count": sum(cluster_gate_counts),
        "resource_estimation_method": "transpiled_optimal_qaoa_circuit_per_cluster",
        "valid": valid,
        "validation_errors": list(validation_errors),
        "route_files": {
            "internal": to_repo_relative(output_paths["internal_route"]),
            "submission": to_repo_relative(output_paths["submission_route"]),
        },
        "plot_file": to_repo_relative(output_paths["plot"]),
    }
    return metrics
