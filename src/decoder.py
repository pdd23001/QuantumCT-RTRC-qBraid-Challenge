from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.geometry import route_distance
from src.models import ClusterProblem


def _sample_matrix(
    values: Sequence[float],
    problem: ClusterProblem,
) -> np.ndarray:
    customer_to_row = {customer_id: row for row, customer_id in enumerate(problem.customer_ids)}
    matrix = np.zeros((len(problem.customer_ids), len(problem.customer_ids)), dtype=float)
    for value, name in zip(values, problem.variable_names):
        _, customer_text, position_text = name.split("_")
        row = customer_to_row[int(customer_text)]
        column = int(position_text) - 1
        matrix[row, column] = float(value)
    return matrix


def _route_from_valid_matrix(
    matrix: np.ndarray,
    customer_ids: Sequence[int],
) -> tuple[int, ...] | None:
    binary = (matrix >= 0.5).astype(int)
    if not np.all(binary.sum(axis=0) == 1):
        return None
    if not np.all(binary.sum(axis=1) == 1):
        return None

    ordering = [0] * len(customer_ids)
    for row, column in zip(*np.where(binary == 1)):
        ordering[column] = customer_ids[row]
    return (0, *ordering, 0)


def _repair_route(
    matrix: np.ndarray,
    customer_ids: Sequence[int],
) -> tuple[int, ...]:
    tie_break = np.arange(matrix.size, dtype=float).reshape(matrix.shape) * 1e-6
    row_index, column_index = linear_sum_assignment(-matrix + tie_break)
    ordering = [0] * len(customer_ids)
    for row, column in zip(row_index, column_index):
        ordering[column] = customer_ids[row]
    return (0, *ordering, 0)


def decode_best_route(
    problem: ClusterProblem,
    samples: Sequence[object],
    distance_matrix: np.ndarray,
    node_index: dict[int, int],
) -> tuple[tuple[int, ...], bool]:
    if not samples:
        return (0, *problem.customer_ids, 0), True

    best_valid: tuple[float, float, tuple[int, ...]] | None = None
    best_repaired: tuple[float, float, tuple[int, ...]] | None = None

    sorted_samples = sorted(
        samples,
        key=lambda sample: (float(getattr(sample, "fval", 0.0)), -float(getattr(sample, "probability", 0.0))),
    )

    for sample in sorted_samples:
        matrix = _sample_matrix(sample.x, problem)
        valid_route = _route_from_valid_matrix(matrix, problem.customer_ids)
        probability = float(getattr(sample, "probability", 0.0))
        if valid_route is not None:
            distance = route_distance(valid_route, distance_matrix, node_index)
            candidate = (distance, -probability, valid_route)
            if best_valid is None or candidate < best_valid:
                best_valid = candidate
            continue

        repaired_route = _repair_route(matrix, problem.customer_ids)
        distance = route_distance(repaired_route, distance_matrix, node_index)
        candidate = (distance, -probability, repaired_route)
        if best_repaired is None or candidate < best_repaired:
            best_repaired = candidate

    if best_valid is not None:
        return best_valid[2], False
    if best_repaired is not None:
        return best_repaired[2], True
    return (0, *problem.customer_ids, 0), True
