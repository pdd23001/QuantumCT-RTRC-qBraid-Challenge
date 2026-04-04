from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from src.models import CVRPInstance


def build_distance_matrix(instance: CVRPInstance) -> np.ndarray:
    nodes = instance.all_nodes
    matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
    for left_index, left_node in enumerate(nodes):
        for right_index, right_node in enumerate(nodes):
            matrix[left_index, right_index] = math.hypot(
                left_node.x - right_node.x,
                left_node.y - right_node.y,
            )
    return matrix


def compute_polar_coordinates(
    instance: CVRPInstance,
) -> dict[int, tuple[float, float]]:
    depot_x, depot_y = instance.depot.x, instance.depot.y
    polar: dict[int, tuple[float, float]] = {}
    for customer in instance.customers:
        dx = customer.x - depot_x
        dy = customer.y - depot_y
        polar[customer.id] = (math.atan2(dy, dx), math.hypot(dx, dy))
    return polar


def route_distance(
    route: Sequence[int],
    distance_matrix: np.ndarray,
    node_index: dict[int, int],
) -> float:
    return float(
        sum(
            distance_matrix[node_index[left], node_index[right]]
            for left, right in zip(route, route[1:])
        )
    )


def nearest_neighbor_route(
    customer_ids: Iterable[int],
    distance_matrix: np.ndarray,
    node_index: dict[int, int],
) -> tuple[int, ...]:
    remaining = set(customer_ids)
    if not remaining:
        return (0, 0)

    route = [0]
    current = 0
    while remaining:
        next_customer = min(
            remaining,
            key=lambda customer_id: distance_matrix[node_index[current], node_index[customer_id]],
        )
        route.append(next_customer)
        remaining.remove(next_customer)
        current = next_customer
    route.append(0)
    return tuple(route)


def nearest_neighbor_distance(
    customer_ids: Iterable[int],
    distance_matrix: np.ndarray,
    node_index: dict[int, int],
) -> float:
    return route_distance(
        nearest_neighbor_route(customer_ids, distance_matrix, node_index),
        distance_matrix,
        node_index,
    )
