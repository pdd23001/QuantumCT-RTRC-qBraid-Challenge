from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp

from src.config import PenaltyConfig
from src.models import ClusterProblem


def build_cluster_qubo(
    cluster_index: int,
    customer_ids: Sequence[int],
    distance_matrix: np.ndarray,
    node_index: dict[int, int],
    penalties: PenaltyConfig,
) -> ClusterProblem:
    customers = tuple(customer_ids)
    positions = range(len(customers))

    model = Model(name=f"cluster_{cluster_index}_routing")
    variables = {
        (customer_id, position): model.binary_var(name=f"y_{customer_id}_{position + 1}")
        for customer_id in customers
        for position in positions
    }

    depot_index = node_index[0]

    objective = model.sum(
        distance_matrix[depot_index, node_index[customer_id]] * variables[(customer_id, 0)]
        for customer_id in customers
    )

    objective += model.sum(
        distance_matrix[node_index[left_customer], node_index[right_customer]]
        * variables[(left_customer, position)]
        * variables[(right_customer, position + 1)]
        for position in range(len(customers) - 1)
        for left_customer in customers
        for right_customer in customers
    )

    objective += model.sum(
        distance_matrix[node_index[customer_id], depot_index]
        * variables[(customer_id, len(customers) - 1)]
        for customer_id in customers
    )

    row_penalty = penalties.row_exactly_one * model.sum(
        (model.sum(variables[(customer_id, position)] for position in positions) - 1) ** 2
        for customer_id in customers
    )

    column_penalty = penalties.col_exactly_one * model.sum(
        (model.sum(variables[(customer_id, position)] for customer_id in customers) - 1) ** 2
        for position in positions
    )

    model.minimize(objective + row_penalty + column_penalty)
    quadratic_program = from_docplex_mp(model)
    variable_names = tuple(variable.name for variable in quadratic_program.variables)

    return ClusterProblem(
        cluster_index=cluster_index,
        customer_ids=customers,
        variable_names=variable_names,
        quadratic_program=quadratic_program,
        num_qubits=len(variable_names),
    )
