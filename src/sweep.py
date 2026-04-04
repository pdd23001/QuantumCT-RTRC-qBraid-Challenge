from __future__ import annotations

from typing import Iterable

import numpy as np

from src.config import SweepConfig
from src.geometry import nearest_neighbor_distance
from src.models import CVRPInstance, SweepCandidate


def _ordered_customers(
    polar: dict[int, tuple[float, float]],
    direction: str,
) -> list[int]:
    customers = sorted(
        polar.items(),
        key=lambda item: (item[1][0], item[1][1], item[0]),
    )
    ordered = [customer_id for customer_id, _ in customers]
    if direction == "counterclockwise":
        return list(reversed(ordered))
    if direction != "clockwise":
        raise ValueError(f"Unsupported sweep direction: {direction}")
    return ordered


def _candidate_offsets(count: int, multi_start: int) -> tuple[int, ...]:
    if count == 0:
        return (0,)
    usable = min(count, max(1, multi_start))
    offsets = {int(round(step * count / usable)) % count for step in range(usable)}
    return tuple(sorted(offsets))


def _partition_by_capacity(
    ordered_customers: Iterable[int],
    vehicles: int,
    capacity: int,
) -> tuple[tuple[int, ...], ...]:
    ordered = list(ordered_customers)
    clusters: list[tuple[int, ...]] = []
    cursor = 0
    for _ in range(vehicles):
        if cursor >= len(ordered):
            clusters.append(tuple())
            continue
        chunk = tuple(ordered[cursor : cursor + capacity])
        clusters.append(chunk)
        cursor += len(chunk)
    if cursor != len(ordered):
        raise ValueError("Sweep partition left customers unassigned.")
    return tuple(clusters)


def _score_candidate(
    clusters: tuple[tuple[int, ...], ...],
    distance_matrix: np.ndarray,
    node_index: dict[int, int],
    config: SweepConfig,
) -> float:
    route_estimate = sum(
        nearest_neighbor_distance(cluster, distance_matrix, node_index)
        for cluster in clusters
        if cluster
    )
    loads = [len(cluster) for cluster in clusters]
    mean_load = sum(loads) / len(loads) if loads else 0.0
    imbalance = sum(abs(load - mean_load) for load in loads)
    return float(route_estimate + config.balance_penalty_weight * imbalance)


def select_sweep_decomposition(
    instance: CVRPInstance,
    polar: dict[int, tuple[float, float]],
    distance_matrix: np.ndarray,
    node_index: dict[int, int],
    config: SweepConfig,
) -> SweepCandidate:
    if not instance.customers:
        return SweepCandidate(
            clusters=tuple(tuple() for _ in range(instance.vehicles)),
            score=0.0,
            direction="clockwise",
            offset=0,
        )

    candidates: list[SweepCandidate] = []
    for direction in config.directions:
        ordered = _ordered_customers(polar, direction)
        for offset in _candidate_offsets(len(ordered), config.multi_start):
            rotated = ordered[offset:] + ordered[:offset]
            clusters = _partition_by_capacity(
                rotated,
                vehicles=instance.vehicles,
                capacity=instance.capacity,
            )
            candidates.append(
                SweepCandidate(
                    clusters=clusters,
                    score=_score_candidate(clusters, distance_matrix, node_index, config),
                    direction=direction,
                    offset=offset,
                )
            )

    return min(candidates, key=lambda candidate: (candidate.score, candidate.direction, candidate.offset))
