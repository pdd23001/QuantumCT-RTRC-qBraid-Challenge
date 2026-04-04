from __future__ import annotations

from collections.abc import Sequence

from qiskit import transpile
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from src.config import SolverConfig
from src.decoder import decode_best_route
from src.geometry import route_distance
from src.models import ClusterProblem, ClusterSolveResult, SamplerHandle


def _build_optimizer(name: str, maxiter: int):
    normalized = name.upper()
    if normalized == "COBYLA":
        return COBYLA(maxiter=maxiter)
    if normalized == "SPSA":
        return SPSA(maxiter=maxiter)
    raise ValueError(f"Unsupported optimizer: {name}")


def _decompose_circuit(circuit):
    current = circuit
    for _ in range(4):
        op_names = set(current.count_ops())
        if "QAOA" not in op_names:
            break
        current = current.decompose()
    return current


def _estimate_resources(raw_result, sampler_handle: SamplerHandle) -> tuple[int, int]:
    circuit = getattr(raw_result, "optimal_circuit", None)
    if circuit is None:
        return 0, 0

    parameters = getattr(raw_result, "optimal_parameters", None) or {}
    if parameters:
        circuit = circuit.assign_parameters(parameters)

    try:
        if sampler_handle.backend is not None:
            compiled = transpile(circuit, backend=sampler_handle.backend, optimization_level=1)
        elif sampler_handle.transpiler is not None:
            compiled = sampler_handle.transpiler.run(circuit)
        else:
            compiled = _decompose_circuit(circuit)
    except Exception:
        compiled = _decompose_circuit(circuit)

    gate_count = int(sum(compiled.count_ops().values()))
    return compiled.num_qubits, gate_count


def solve_cluster(
    problem: ClusterProblem,
    sampler_handle: SamplerHandle | None,
    distance_matrix,
    node_index: dict[int, int],
    config: SolverConfig,
) -> ClusterSolveResult:
    customers = problem.customer_ids

    if not customers:
        route = (0, 0)
        return ClusterSolveResult(
            cluster_index=problem.cluster_index,
            customer_ids=customers,
            route=route,
            route_distance=0.0,
            objective_value=0.0,
            sample_count=0,
            used_qaoa=False,
            repaired=False,
            qubits_used=0,
            gate_count=0,
            provider_name="none",
            backend_name="none",
            status="SKIPPED_EMPTY",
        )

    if len(customers) == 1:
        route = (0, customers[0], 0)
        return ClusterSolveResult(
            cluster_index=problem.cluster_index,
            customer_ids=customers,
            route=route,
            route_distance=route_distance(route, distance_matrix, node_index),
            objective_value=0.0,
            sample_count=1,
            used_qaoa=False,
            repaired=False,
            qubits_used=1,
            gate_count=0,
            provider_name="trivial",
            backend_name="trivial",
            status="TRIVIAL_SINGLE_CUSTOMER",
        )

    if sampler_handle is None:
        raise RuntimeError("A sampler handle is required for non-trivial cluster QAOA solves.")

    optimizer = _build_optimizer(config.qaoa.optimizer, config.qaoa.maxiter)
    qaoa = QAOA(
        sampler=sampler_handle.sampler,
        optimizer=optimizer,
        reps=config.qaoa.reps,
        transpiler=sampler_handle.transpiler,
    )
    minimum_eigen_optimizer = MinimumEigenOptimizer(qaoa)
    result = minimum_eigen_optimizer.solve(problem.quadratic_program)

    decoded_route, repaired = decode_best_route(
        problem,
        result.samples,
        distance_matrix,
        node_index,
    )
    qubits_used, gate_count = _estimate_resources(result.min_eigen_solver_result, sampler_handle)

    return ClusterSolveResult(
        cluster_index=problem.cluster_index,
        customer_ids=customers,
        route=decoded_route,
        route_distance=route_distance(decoded_route, distance_matrix, node_index),
        objective_value=float(result.fval),
        sample_count=len(result.samples),
        used_qaoa=True,
        repaired=repaired,
        qubits_used=qubits_used,
        gate_count=gate_count,
        provider_name=sampler_handle.provider_name,
        backend_name=sampler_handle.backend_name,
        status=str(result.status),
    )
