import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class VRPClarkeWright:
    def __init__(self, instance):
        self.instance_id = instance["instance_id"]
        self.num_vehicles = int(instance["Nv"])
        self.capacity = int(instance["C"])
        self.depot = None
        self.customers = []  # customer ids are 1..n, each entry is (x, y, demand)
        self.routes = []

        for customer in instance["customers"]:
            customer_id = int(customer["customer_id"])
            x = float(customer["x"])
            y = float(customer["y"])
            demand = int(customer["demand"])

            if customer_id == 0:
                self.depot = (x, y)
            else:
                self.customers.append((x, y, demand))

        if self.depot is None:
            raise ValueError(f"Instance {self.instance_id} is missing a depot.")

        self.num_customers = len(self.customers)
        self.distance_matrix = self._compute_distances()

    def _compute_distances(self):
        points = np.array([self.depot] + [(c[0], c[1]) for c in self.customers], dtype=float)
        deltas = points[:, None, :] - points[None, :, :]
        return np.sqrt((deltas ** 2).sum(axis=2))

    def route_demand(self, route):
        return sum(self.customers[i - 1][2] for i in route)

    def route_distance(self, route):
        if not route:
            return 0.0

        total = self.distance_matrix[0, route[0]]
        for i in range(len(route) - 1):
            total += self.distance_matrix[route[i], route[i + 1]]
        total += self.distance_matrix[route[-1], 0]
        return float(total)

    def total_distance(self):
        return sum(self.route_distance(route) for route in self.routes)

    def validate(self):
        seen = []
        for route in self.routes:
            if self.route_demand(route) > self.capacity:
                return False
            seen.extend(route)

        return (
            len(self.routes) <= self.num_vehicles
            and sorted(seen) == list(range(1, self.num_customers + 1))
        )

    def route_details(self):
        details = []
        for idx, route in enumerate(self.routes, start=1):
            details.append(
                {
                    "vehicle": idx,
                    "route": [0] + route + [0],
                    "load": self.route_demand(route),
                    "distance": round(self.route_distance(route), 6),
                }
            )
        return details

    def clarke_and_wright(self):
        savings = []
        for i in range(1, self.num_customers + 1):
            for j in range(i + 1, self.num_customers + 1):
                saving = self.distance_matrix[0, i] + self.distance_matrix[0, j] - self.distance_matrix[i, j]
                savings.append((saving, i, j))

        savings.sort(reverse=True)

        routes = [[i] for i in range(1, self.num_customers + 1)]
        customer_to_route = {i: routes[i - 1] for i in range(1, self.num_customers + 1)}

        def remap(route):
            for customer in route:
                customer_to_route[customer] = route

        for _, i, j in savings:
            route_i = customer_to_route.get(i)
            route_j = customer_to_route.get(j)

            if route_i is None or route_j is None or route_i is route_j:
                continue

            if self.route_demand(route_i) + self.route_demand(route_j) > self.capacity:
                continue

            if i not in (route_i[0], route_i[-1]) or j not in (route_j[0], route_j[-1]):
                continue

            left = route_i[:] if route_i[-1] == i else route_i[::-1]
            right = route_j[:] if route_j[0] == j else route_j[::-1]
            merged = left + right

            routes.remove(route_i)
            routes.remove(route_j)
            routes.append(merged)
            remap(merged)

        self.routes = routes

    def plot_routes(self, output_path):
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.scatter(
            self.depot[0],
            self.depot[1],
            c="black",
            marker="X",
            s=220,
            label="Depot",
            zorder=5,
        )
        ax.text(self.depot[0] + 0.4, self.depot[1] + 0.4, "0", fontsize=11)

        xs = [c[0] for c in self.customers]
        ys = [c[1] for c in self.customers]
        ax.scatter(xs, ys, c="black", s=40, zorder=4)

        for idx, (x, y, _) in enumerate(self.customers, start=1):
            ax.text(x + 0.4, y + 0.4, str(idx), fontsize=8)

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for route_idx, route in enumerate(self.routes, start=1):
            points = [self.depot]
            points.extend((self.customers[c - 1][0], self.customers[c - 1][1]) for c in route)
            points.append(self.depot)
            points = np.array(points)

            ax.plot(
                points[:, 0],
                points[:, 1],
                linewidth=2.2,
                color=colors[(route_idx - 1) % len(colors)],
                label=f"Route {route_idx}: {route}",
                zorder=2,
            )

        ax.set_title(
            f"{self.instance_id} - Clarke & Wright\n"
            f"dist={self.total_distance():.3f}, time={getattr(self, 'last_runtime', 0.0):.6f}s",
            fontsize=16,
        )
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.grid(True, alpha=0.5)
        ax.legend(fontsize=8, loc="best")

        plt.tight_layout()
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def load_instances(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_benchmark(json_path="setA_random_instances_grouped.json", output_dir="benchmark_output"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    instances = load_instances(json_path)
    results = []

    for instance in instances:
        vrp = VRPClarkeWright(instance)

        t0 = time.perf_counter()
        vrp.clarke_and_wright()
        runtime = time.perf_counter() - t0
        vrp.last_runtime = runtime

        result = {
            "id": vrp.instance_id,
            "customers": vrp.num_customers,
            "vehicles_allowed": vrp.num_vehicles,
            "vehicles_used": len(vrp.routes),
            "distance": round(vrp.total_distance(), 6),
            "runtime": runtime,
            "valid": vrp.validate(),
            "routes": vrp.route_details(),
        }
        results.append(result)

        image_path = output_path / f"outputs_png_{vrp.instance_id}.png"
        vrp.plot_routes(image_path)
        print(f"Saved plot: {image_path}")

    json_out = output_path / "benchmark_results.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results JSON: {json_out}")
    return results


def plot_summary(results, output_dir="benchmark_output"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ids = [r["id"] for r in results]
    distances = [r["distance"] for r in results]
    runtimes = [r["runtime"] for r in results]
    customers = [r["customers"] for r in results]

    label_step = max(1, len(ids) // 12)
    display_labels = [instance_id if idx % label_step == 0 else "" for idx, instance_id in enumerate(ids)]

    plt.figure(figsize=(14, 6))
    plt.bar(ids, distances)
    plt.title("Distance (Clarke-Wright)")
    plt.ylabel("Total Distance")
    plt.xticks(range(len(ids)), display_labels, rotation=45, fontsize=8)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    for bar, value in zip(plt.gca().containers[0], distances):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    plt.savefig(output_path / "distance.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.bar(ids, runtimes)
    plt.title("Runtime")
    plt.ylabel("Seconds")
    plt.xticks(range(len(ids)), display_labels, rotation=45, fontsize=8)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "runtime.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(customers, runtimes)
    plt.xlabel("Customers")
    plt.ylabel("Runtime")
    plt.title("Runtime vs Problem Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #Create a linear regression line
    if len(customers) > 1:
        coeffs = np.polyfit(customers, runtimes, deg=1)
        x_fit = np.linspace(min(customers), max(customers), 100)
        y_fit = np.polyval(coeffs, x_fit)
        plt.plot(x_fit, y_fit, color="red", linestyle="--", label=f"Fit: Runtime={coeffs[0]:.4f}customers + {coeffs[1]:.4f}")
    plt.savefig(output_path / "runtime_vs_size.png", dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Saved summary plots in: {output_path}")


if __name__ == "__main__":
    benchmark_results = run_benchmark(
        json_path="c5n25.json",
        output_dir="ClarkeWright/c5n25_output",
    )
    plot_summary(benchmark_results, output_dir="ClarkeWright/c5n25_output")
