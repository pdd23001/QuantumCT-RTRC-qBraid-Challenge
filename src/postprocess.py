from __future__ import annotations

from typing import Sequence

import numpy as np

from src.geometry import route_distance


def two_opt_route(
    route: Sequence[int],
    distance_matrix: np.ndarray,
    node_index: dict[int, int],
) -> tuple[int, ...]:
    if len(route) <= 4:
        return tuple(route)

    best_route = tuple(route)
    best_distance = route_distance(best_route, distance_matrix, node_index)
    improved = True

    while improved:
        improved = False
        for start in range(1, len(best_route) - 2):
            for end in range(start + 1, len(best_route) - 1):
                candidate = (
                    best_route[:start]
                    + tuple(reversed(best_route[start : end + 1]))
                    + best_route[end + 1 :]
                )
                candidate_distance = route_distance(candidate, distance_matrix, node_index)
                if candidate_distance + 1e-9 < best_distance:
                    best_route = candidate
                    best_distance = candidate_distance
                    improved = True
                    break
            if improved:
                break

    return best_route
