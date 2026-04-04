from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class Node:
    id: int
    x: float
    y: float


@dataclass(frozen=True)
class CVRPInstance:
    instance_name: str
    depot: Node
    vehicles: int
    capacity: int
    customers: tuple[Node, ...]
    qaoa_overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def customer_ids(self) -> tuple[int, ...]:
        return tuple(customer.id for customer in self.customers)

    @property
    def all_nodes(self) -> tuple[Node, ...]:
        return (self.depot, *self.customers)

    @property
    def node_ids(self) -> tuple[int, ...]:
        return tuple(node.id for node in self.all_nodes)

    def node_index(self) -> dict[int, int]:
        return {node.id: index for index, node in enumerate(self.all_nodes)}

    def coordinates(self) -> dict[int, tuple[float, float]]:
        return {node.id: (node.x, node.y) for node in self.all_nodes}


@dataclass(frozen=True)
class SweepCandidate:
    clusters: tuple[tuple[int, ...], ...]
    score: float
    direction: str
    offset: int


@dataclass(frozen=True)
class ClusterProblem:
    cluster_index: int
    customer_ids: tuple[int, ...]
    variable_names: tuple[str, ...]
    quadratic_program: Any
    num_qubits: int


@dataclass(frozen=True)
class SamplerHandle:
    mode: str
    sampler: Any
    provider_name: str
    backend_name: str
    backend: Any = None
    transpiler: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClusterSolveResult:
    cluster_index: int
    customer_ids: tuple[int, ...]
    route: tuple[int, ...]
    route_distance: float
    objective_value: float
    sample_count: int
    used_qaoa: bool
    repaired: bool
    qubits_used: int
    gate_count: int
    provider_name: str
    backend_name: str
    status: str


@dataclass(frozen=True)
class InstanceSolution:
    instance: CVRPInstance
    routes: tuple[tuple[int, ...], ...]
    total_distance: float
    valid: bool
    validation_errors: tuple[str, ...]
    sweep_candidate: SweepCandidate
    cluster_results: tuple[ClusterSolveResult, ...]
    runtime_seconds: float
    metrics: dict[str, Any]
    output_paths: dict[str, str]
