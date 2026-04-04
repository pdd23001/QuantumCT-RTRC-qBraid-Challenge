import json
import time
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


class VRPClarkeWright:
    """Clarke-Wright savings heuristic with faster route bookkeeping."""

    def __init__(self, instance_id, num_vehicles, capacity, depot, customers):
        self.instance_id = instance_id
        self.num_vehicles = num_vehicles
        self.capacity = capacity
        self.depot = tuple(depot)
        self.customers = customers  # [(x, y, demand)] indexed from customer_id 1..n in routes
        self.num_customers = len(customers)
        self.routes = []
        self.distance_matrix = self.compute_distances()

    def compute_distances(self):
        points = np.array([self.depot] + [(c[0], c[1]) for c in self.customers], dtype=float)
        diff = points[:, None, :] - points[None, :, :]
        return np.hypot(diff[:, :, 0], diff[:, :, 1])

    def route_demand(self, route):
        return sum(self.customers[i - 1][2] for i in route)

    def route_distance(self, route):
        if not route:
            return 0.0
        full_route = [0] + route + [0]
        return float(sum(self.distance_matrix[full_route[i]][full_route[i + 1]] for i in range(len(full_route) - 1)))

    def route_with_depot(self, route):
        return [0] + route + [0]

    def total_distance(self):
        return float(sum(self.route_distance(r) for r in self.routes))

    def validate(self):
        seen = []
        if len(self.routes) > self.num_vehicles:
            return False
        for r in self.routes:
            if self.route_demand(r) > self.capacity:
                return False
            seen.extend(r)
        return sorted(seen) == list(range(1, self.num_customers + 1))

    def clarke_and_wright(self):
        n = self.num_customers
        d0 = self.distance_matrix[0, 1:]
        dij = self.distance_matrix[1:, 1:]

        savings = []
        for i in range(n):
            for j in range(i + 1, n):
                savings.append((d0[i] + d0[j] - dij[i, j], i + 1, j + 1))
        savings.sort(key=lambda x: x[0], reverse=True)

        routes = {i: [i] for i in range(1, n + 1)}
        route_load = {i: self.customers[i - 1][2] for i in range(1, n + 1)}
        route_of = {i: i for i in range(1, n + 1)}
        next_route_id = n + 1

        for _, i, j in savings:
            ri_id = route_of.get(i)
            rj_id = route_of.get(j)
            if ri_id is None or rj_id is None or ri_id == rj_id:
                continue

            ri = routes[ri_id]
            rj = routes[rj_id]

            if route_load[ri_id] + route_load[rj_id] > self.capacity:
                continue

            if i not in (ri[0], ri[-1]) or j not in (rj[0], rj[-1]):
                continue

            if ri[-1] == i and rj[0] == j:
                merged = ri + rj
            elif ri[0] == i and rj[-1] == j:
                merged = rj + ri
            elif ri[0] == i and rj[0] == j:
                merged = ri[::-1] + rj
            else:  # ri[-1] == i and rj[-1] == j
                merged = ri + rj[::-1]

            new_id = next_route_id
            next_route_id += 1

            routes[new_id] = merged
            route_load[new_id] = route_load[ri_id] + route_load[rj_id]
            for customer in merged:
                route_of[customer] = new_id

            del routes[ri_id], routes[rj_id]
            del route_load[ri_id], route_load[rj_id]

            if len(routes) <= self.num_vehicles:
                # Keep going can still improve within vehicle limit, but this early check lets us skip nothing.
                pass

        self.routes = list(routes.values())

    def export_solution(self):
        route_details = []
        for idx, route in enumerate(self.routes, start=1):
            route_details.append(
                {
                    "vehicle": idx,
                    "route": self.route_with_depot(route),
                    "load": self.route_demand(route),
                    "distance": round(self.route_distance(route), 6),
                }
            )

        return {
            "id": self.instance_id,
            "customers": self.num_customers,
            "vehicles_used": len(self.routes),
            "distance": round(self.total_distance(), 6),
            "routes": route_details,
            "valid": self.validate(),
        }


def load_instances(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    vrps = []
    for inst in data:
        depot = None
        customers = []
        for c in inst["customers"]:
            if c["customer_id"] == 0:
                depot = (c["x"], c["y"])
            else:
                customers.append((c["x"], c["y"], c["demand"]))

        vrps.append(
            VRPClarkeWright(
                inst["instance_id"],
                inst["Nv"],
                inst["C"],
                depot,
                customers,
            )
        )
    return vrps


def run_benchmark(instances):
    results = []
    for vrp in instances:
        t0 = time.perf_counter()
        vrp.clarke_and_wright()
        runtime = time.perf_counter() - t0

        result = vrp.export_solution()
        result["runtime"] = round(runtime, 6)
        results.append(result)
    return results


def save_results(results, output_file="benchmark_results.json"):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def _configure_id_axis(ax, ids):
    count = len(ids)
    positions = np.arange(count)
    ax.set_xticks(positions)

    if count <= 20:
        fontsize = 8
        shown_labels = ids
    else:
        fontsize = 6
        step = max(1, count // 15)
        shown_labels = [label if i % step == 0 else "" for i, label in enumerate(ids)]

    ax.set_xticklabels(shown_labels, rotation=45, ha="right", fontsize=fontsize)
    return positions


def plot_results(results):
    ids = [r["id"] for r in results]
    distances = [r["distance"] for r in results]
    runtimes = [r["runtime"] for r in results]
    customers = [r["customers"] for r in results]
    valid_colors = [1 if r["valid"] else 0 for r in results]

    # Distance plot - cleaner labels and tighter layout
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = _configure_id_axis(ax, ids)
    ax.bar(positions, distances)
    ax.set_title("Solution Distance by Instance")
    ax.set_ylabel("Distance")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig("distance.png", dpi=180)
    plt.close(fig)

    # Runtime plot
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = _configure_id_axis(ax, ids)
    ax.bar(positions, runtimes)
    ax.set_title("Runtime by Instance")
    ax.set_ylabel("Seconds")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig("runtime.png", dpi=180)
    plt.close(fig)

    # Runtime vs size plot
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(customers, runtimes, c=valid_colors)
    ax.set_xlabel("Customers")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime vs Problem Size")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("runtime_vs_size.png", dpi=180)
    plt.close(fig)


def main():
    base = Path(__file__).resolve().parent
    json_path = base / "setA_random_instances_grouped.json"

    instances = load_instances(json_path)
    results = run_benchmark(instances)

    for r in results:
        print(r)

    save_results(results, base / "benchmark_results.json")
    plot_results(results)


if __name__ == "__main__":
    main()
