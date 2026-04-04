from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from scipy.sparse import SparseEfficiencyWarning

from src.config import PenaltyConfig, QAOAConfig, RuntimeConfig, SolverConfig, SweepConfig
from src.geometry import build_distance_matrix, compute_polar_coordinates, route_distance
from src.io_utils import (
    DEFAULT_INSTANCE_DIR,
    DEFAULT_SCHEMA_PATH,
    discover_instance_paths,
    ensure_output_dirs,
    load_json,
    output_paths_for_instance,
    sync_submission_file,
    write_json,
    write_route_file,
)
from src.metrics import build_metrics
from src.models import ClusterSolveResult, CVRPInstance, InstanceSolution
from src.postprocess import two_opt_route
from src.qaoa_solver import solve_cluster
from src.qubo_builder import build_cluster_qubo
from src.runtime import SamplerFactory
from src.sweep import select_sweep_decomposition
from src.validation import InstanceValidationError, parse_instance, validate_routes
from src.visualize import plot_routes

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep plus QAOA solver for hackathon CVRP instances.")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--instance", type=Path, help="Path to a single instance JSON file.")
    target_group.add_argument(
        "--all",
        action="store_true",
        help="Solve every Instance*.json file in data/instances.",
    )

    parser.add_argument(
        "--mode",
        choices=("local_statevector", "local_aer", "qbraid_runtime"),
        default="local_statevector",
        help="Sampler/runtime mode.",
    )
    parser.add_argument("--qaoa-reps", type=int, default=1)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--optimizer", type=str, default="COBYLA")
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-start-sweep", type=int, default=16)
    parser.add_argument("--row-penalty", type=float, default=20.0)
    parser.add_argument("--col-penalty", type=float, default=20.0)
    parser.add_argument("--backend-name", type=str, default=None)
    parser.add_argument("--qbraid-channel", type=str, default=None)
    parser.add_argument("--disable-2opt", action="store_true")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--instance-dir", type=Path, default=DEFAULT_INSTANCE_DIR)
    return parser


def _base_config_from_args(args: argparse.Namespace) -> SolverConfig:
    return SolverConfig(
        sweep=SweepConfig(multi_start=args.multi_start_sweep),
        qaoa=QAOAConfig(
            reps=args.qaoa_reps,
            optimizer=args.optimizer.upper(),
            maxiter=args.maxiter,
            shots=args.shots,
            seed=args.seed,
        ),
        penalties=PenaltyConfig(
            row_exactly_one=args.row_penalty,
            col_exactly_one=args.col_penalty,
        ),
        runtime=RuntimeConfig(
            mode=args.mode,
            backend_name=args.backend_name,
            qbraid_channel=args.qbraid_channel,
        ),
        enable_2opt=not args.disable_2opt,
    )


def _instance_paths(args: argparse.Namespace) -> list[Path]:
    if args.instance is not None:
        return [args.instance]
    return discover_instance_paths(args.instance_dir)


def solve_instance(instance_path: Path, schema_path: Path, base_config: SolverConfig) -> InstanceSolution:
    raw_instance = load_json(instance_path)
    instance = parse_instance(raw_instance, schema_path)
    config = base_config.with_instance_overrides(instance.qaoa_overrides)

    start_time = time.perf_counter()
    ensure_output_dirs()

    distance_matrix = build_distance_matrix(instance)
    node_index = instance.node_index()
    polar = compute_polar_coordinates(instance)
    sweep_candidate = select_sweep_decomposition(
        instance,
        polar,
        distance_matrix,
        node_index,
        config.sweep,
    )

    sampler_handle = None
    cluster_results: list[ClusterSolveResult] = []
    routes: list[tuple[int, ...]] = []

    for cluster_index, cluster_customer_ids in enumerate(sweep_candidate.clusters, start=1):
        cluster_problem = build_cluster_qubo(
            cluster_index=cluster_index,
            customer_ids=cluster_customer_ids,
            distance_matrix=distance_matrix,
            node_index=node_index,
            penalties=config.penalties,
        )

        if len(cluster_customer_ids) > 1 and sampler_handle is None:
            sampler_handle = SamplerFactory.build(
                config.runtime,
                shots=config.qaoa.shots,
                seed=config.qaoa.seed,
            )

        cluster_result = solve_cluster(
            problem=cluster_problem,
            sampler_handle=sampler_handle,
            distance_matrix=distance_matrix,
            node_index=node_index,
            config=config,
        )

        route = cluster_result.route
        if config.enable_2opt and len(route) > 4:
            improved_route = two_opt_route(route, distance_matrix, node_index)
            improved_distance = route_distance(improved_route, distance_matrix, node_index)
            cluster_result = replace(
                cluster_result,
                route=improved_route,
                route_distance=improved_distance,
            )
            route = improved_route

        cluster_results.append(cluster_result)
        routes.append(route)

    valid, validation_errors = validate_routes(instance, routes)
    runtime_seconds = time.perf_counter() - start_time
    total_distance = sum(
        route_distance(route, distance_matrix, node_index)
        for route in routes
    )
    output_paths = output_paths_for_instance(instance.instance_name)

    if valid:
        write_route_file(output_paths["internal_route"], routes)
        sync_submission_file(output_paths["internal_route"], output_paths["submission_route"])
        plot_routes(instance, routes, output_paths["plot"])

    metrics = build_metrics(
        instance=instance,
        config=config,
        sweep_candidate=sweep_candidate,
        cluster_results=cluster_results,
        total_distance=total_distance,
        runtime_seconds=runtime_seconds,
        valid=valid,
        validation_errors=validation_errors,
        output_paths=output_paths,
    )
    write_json(output_paths["metrics"], metrics)

    return InstanceSolution(
        instance=instance,
        routes=tuple(routes),
        total_distance=total_distance,
        valid=valid,
        validation_errors=validation_errors,
        sweep_candidate=sweep_candidate,
        cluster_results=tuple(cluster_results),
        runtime_seconds=runtime_seconds,
        metrics=metrics,
        output_paths={key: str(path) for key, path in output_paths.items()},
    )


def _render_summary(solution: InstanceSolution) -> str:
    status = "VALID" if solution.valid else "INVALID"
    return (
        f"{solution.instance.instance_name}: {status} | "
        f"distance={solution.total_distance:.4f} | "
        f"runtime={solution.runtime_seconds:.2f}s | "
        f"clusters={[list(cluster) for cluster in solution.sweep_candidate.clusters]}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    base_config = _base_config_from_args(args)

    exit_code = 0
    for instance_path in _instance_paths(args):
        try:
            solution = solve_instance(instance_path, args.schema, base_config)
            print(_render_summary(solution))
            if not solution.valid:
                for error in solution.validation_errors:
                    print(f"  - {error}")
                exit_code = 1
        except (InstanceValidationError, RuntimeError, ValueError) as exc:
            print(f"{instance_path.name}: ERROR | {exc}", file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
