from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

from src.models import CVRPInstance


def plot_routes(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    output_path: Path,
) -> None:
    coordinates = instance.coordinates()
    colors = plt.cm.get_cmap("tab10", max(1, len(routes)))

    figure, axis = plt.subplots(figsize=(7, 7))
    depot_x, depot_y = coordinates[0]
    axis.scatter([depot_x], [depot_y], s=180, marker="*", color="black", label="Depot")
    axis.text(depot_x + 0.1, depot_y + 0.1, "0", fontsize=10, weight="bold")

    customer_x = [customer.x for customer in instance.customers]
    customer_y = [customer.y for customer in instance.customers]
    axis.scatter(customer_x, customer_y, s=90, color="#4c78a8", alpha=0.9, label="Customers")
    for customer in instance.customers:
        axis.text(customer.x + 0.08, customer.y + 0.08, str(customer.id), fontsize=9)

    for index, route in enumerate(routes):
        xs = [coordinates[node_id][0] for node_id in route]
        ys = [coordinates[node_id][1] for node_id in route]
        axis.plot(xs, ys, linewidth=2.0, color=colors(index), label=f"r{index + 1}")

    axis.set_title(f"{instance.instance_name} Routes")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
