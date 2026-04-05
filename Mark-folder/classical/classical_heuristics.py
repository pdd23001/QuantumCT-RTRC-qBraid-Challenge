import csv
import math
import time
import itertools
from functools import lru_cache


import numpy as np
import matplotlib.pyplot as plt


class VRP:
    def __init__(self, instance_id, num_customers, num_vehicles, vehicle_capacity, depot_location):
        self.instance_id = instance_id
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.depot_location = depot_location
        self.customers = []  # list of tuples: (x, y, demand)
        self.distance_matrix = None
        self.routes = []

    def calculate_distance_matrix(self):
        points = [self.depot_location] + [(c[0], c[1]) for c in self.customers]
        n = len(points)
        distance_matrix = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(n):
                distance_matrix[i][j] = math.hypot(
                    points[j][0] - points[i][0],
                    points[j][1] - points[i][1]
                )
        return distance_matrix

    def route_demand(self, route):
        return sum(self.customers[c - 1][2] for c in route)

    def route_distance(self, route):
        if not route:
            return 0.0

        total = self.distance_matrix[0][route[0]]
        for i in range(len(route) - 1):
            total += self.distance_matrix[route[i]][route[i + 1]]
        total += self.distance_matrix[route[-1]][0]
        return total

    def total_distance(self, routes=None):
        routes = self.routes if routes is None else routes
        return sum(self.route_distance(route) for route in routes)

    def validate_routes(self, routes=None):
        routes = self.routes if routes is None else routes

        seen = []
        for route in routes:
            if self.route_demand(route) > self.vehicle_capacity:
                return False, "Capacity violation"
            for c in route:
                seen.append(c)

        expected = list(range(1, self.num_customers + 1))
        if sorted(seen) != expected:
            return False, "Customers are not visited exactly once"

        if len(routes) > self.num_vehicles:
            return False, "Too many vehicles used"

        return True, "Valid"

    def clarke_and_wright(self):
        """
        Parallel Clarke-Wright savings algorithm.
        Uses at most num_vehicles vehicles.
        """
        savings = []
        for i in range(1, self.num_customers + 1):
            for j in range(i + 1, self.num_customers + 1):
                s = self.distance_matrix[0][i] + self.distance_matrix[0][j] - self.distance_matrix[i][j]
                savings.append((s, i, j))

        savings.sort(reverse=True)

        routes = [[i] for i in range(1, self.num_customers + 1)]

        def find_route(customer):
            for route in routes:
                if customer in route:
                    return route
            return None

        for _, i, j in savings:
            route_i = find_route(i)
            route_j = find_route(j)

            if route_i is None or route_j is None or route_i == route_j:
                continue

            if self.route_demand(route_i) + self.route_demand(route_j) > self.vehicle_capacity:
                continue

            # Only merge if i and j are endpoints of their routes
            if i not in (route_i[0], route_i[-1]):
                continue
            if j not in (route_j[0], route_j[-1]):
                continue

            original_i = route_i
            original_j = route_j

            # Orient so that i is at the end of route_i, j is at the start of route_j
            if route_i[0] == i:
                route_i = route_i[::-1]
            if route_j[-1] == j:
                route_j = route_j[::-1]

            merged = route_i + route_j

            routes.remove(original_i)
            routes.remove(original_j)
            routes.append(merged)

        # If we still use too many vehicles, instance is infeasible under this heuristic
        # or needs a stronger repair/merge strategy.
        self.routes = routes

    def plot_routes(self, routes, title, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 7))

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        # depot
        ax.scatter(
            self.depot_location[0],
            self.depot_location[1],
            c='black',
            marker='X',
            s=150,
            label='Depot'
        )
        ax.text(self.depot_location[0] + 0.08, self.depot_location[1] + 0.08, "0", fontsize=10)

        # customers
        for idx, customer in enumerate(self.customers, start=1):
            ax.scatter(customer[0], customer[1], c='black', marker='o', s=45)
            ax.text(customer[0] + 0.08, customer[1] + 0.08, str(idx), fontsize=10)

        # routes
        for r_idx, route in enumerate(routes):
            pts = [self.depot_location] + [(self.customers[c - 1][0], self.customers[c - 1][1]) for c in route] + [self.depot_location]
            pts = np.array(pts)
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=colors[r_idx % len(colors)],
                linewidth=2,
                label=f'Route {r_idx + 1}: {route}'
            )

        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.legend(fontsize=8)


class ExactVRPSolver:
    """
    Exact exhaustive solver for small CVRP instances.
    Enumerates all feasible set partitions (up to num_vehicles routes),
    then finds best and worst route orderings for each route subset.
    """

    def __init__(self, vrp):
        self.vrp = vrp
        self.n = vrp.num_customers
        self.capacity = vrp.vehicle_capacity
        self.max_vehicles = vrp.num_vehicles

        self.best_distance = float("inf")
        self.best_routes = None

        self.worst_distance = -float("inf")
        self.worst_routes = None

        self._subset_customers = {}
        for mask in range(1, 1 << self.n):
            self._subset_customers[mask] = [i + 1 for i in range(self.n) if (mask >> i) & 1]

    def solve(self):
        all_mask = (1 << self.n) - 1
        self._search_partitions(remaining_mask=all_mask, current_partition=[])
        return {
            "best_distance": self.best_distance,
            "best_routes": self.best_routes,
            "worst_distance": self.worst_distance,
            "worst_routes": self.worst_routes,
        }

    def _mask_demand(self, mask):
        demand = 0
        bit = 0
        m = mask
        while m:
            if m & 1:
                demand += self.vrp.customers[bit][2]
            bit += 1
            m >>= 1
        return demand

    @lru_cache(maxsize=None)
    def _route_best_and_worst(self, mask):
        customers = self._subset_customers[mask]

        best_cost = float("inf")
        best_perm = None

        worst_cost = -float("inf")
        worst_perm = None

        for perm in itertools.permutations(customers):
            cost = self.vrp.route_distance(list(perm))
            if cost < best_cost:
                best_cost = cost
                best_perm = list(perm)
            if cost > worst_cost:
                worst_cost = cost
                worst_perm = list(perm)

        return best_cost, best_perm, worst_cost, worst_perm

    def _search_partitions(self, remaining_mask, current_partition):
        if remaining_mask == 0:
            if len(current_partition) <= self.max_vehicles:
                best_total = 0.0
                best_routes = []
                worst_total = 0.0
                worst_routes = []

                for mask in current_partition:
                    b_cost, b_route, w_cost, w_route = self._route_best_and_worst(mask)
                    best_total += b_cost
                    worst_total += w_cost
                    best_routes.append(b_route)
                    worst_routes.append(w_route)

                if best_total < self.best_distance:
                    self.best_distance = best_total
                    self.best_routes = best_routes

                if worst_total > self.worst_distance:
                    self.worst_distance = worst_total
                    self.worst_routes = worst_routes
            return

        used_routes = len(current_partition)
        routes_left = self.max_vehicles - used_routes
        if routes_left <= 0:
            return

        remaining_demand = self._mask_demand(remaining_mask)
        min_routes_needed = math.ceil(remaining_demand / self.capacity)
        if min_routes_needed > routes_left:
            return

        # pick the first remaining customer to avoid duplicate partitions
        first_bit = (remaining_mask & -remaining_mask).bit_length() - 1
        first_customer_mask = 1 << first_bit

        submask = remaining_mask
        while submask:
            if (submask & first_customer_mask) and self._mask_demand(submask) <= self.capacity:
                self._search_partitions(
                    remaining_mask ^ submask,
                    current_partition + [submask]
                )
            submask = (submask - 1) & remaining_mask


def load_instance(filename):
    with open(filename, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    instance_id = (rows[0]["instance_id"])
    num_vehicles = int(rows[0]["Nv"])
    vehicle_capacity = int(rows[0]["C"])

    depot_row = next(row for row in rows if int(row["customer_id"]) == 0)
    depot_location = (float(depot_row["x"]), float(depot_row["y"]))

    customers = []
    for row in rows:
        cid = int(row["customer_id"])
        if cid != 0:
            customers.append((
                float(row["x"]),
                float(row["y"]),
                int(row["demand"])
            ))

    num_customers = len(customers)
    vrp = VRP(instance_id, num_customers, num_vehicles, vehicle_capacity, depot_location)
    vrp.customers = customers
    vrp.distance_matrix = vrp.calculate_distance_matrix()
    return vrp


def normalized_quality(heuristic_distance, best_distance, worst_distance):
    """
    100% = optimal
    0%   = worst feasible solution
    """
    if abs(worst_distance - best_distance) < 1e-12:
        return 100.0
    return 100.0 * (worst_distance - heuristic_distance) / (worst_distance - best_distance)


def optimality_gap_percent(heuristic_distance, best_distance):
    if abs(best_distance) < 1e-12:
        return 0.0
    return 100.0 * (heuristic_distance - best_distance) / best_distance


def compare_instance(vrp, make_plot=True, save_prefix=None):
    # Heuristic
    t0 = time.perf_counter()
    vrp.clarke_and_wright()
    heuristic_time = time.perf_counter() - t0

    heuristic_routes = vrp.routes
    heuristic_distance = vrp.total_distance(heuristic_routes)
    heuristic_valid, heuristic_msg = vrp.validate_routes(heuristic_routes)

    # Exact
    t1 = time.perf_counter()
    exact_solver = ExactVRPSolver(vrp)
    exact_result = exact_solver.solve()
    exact_time = time.perf_counter() - t1

    best_distance = exact_result["best_distance"]
    best_routes = exact_result["best_routes"]
    worst_distance = exact_result["worst_distance"]
    worst_routes = exact_result["worst_routes"]

    gap_pct = optimality_gap_percent(heuristic_distance, best_distance) if heuristic_valid else None
    quality_pct = normalized_quality(heuristic_distance, best_distance, worst_distance) if heuristic_valid else None

    result = {
        "instance_id": vrp.instance_id,
        "num_customers": vrp.num_customers,
        "num_vehicles": vrp.num_vehicles,
        "capacity": vrp.vehicle_capacity,
        "heuristic_routes": heuristic_routes,
        "heuristic_distance": heuristic_distance,
        "heuristic_time_sec": heuristic_time,
        "heuristic_valid": heuristic_valid,
        "heuristic_validation_message": heuristic_msg,
        "optimal_routes": best_routes,
        "optimal_distance": best_distance,
        "worst_routes": worst_routes,
        "worst_distance": worst_distance,
        "exact_time_sec": exact_time,
        "gap_to_optimal_percent": gap_pct,
        "normalized_quality_percent": quality_pct,
    }

    if make_plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        vrp.plot_routes(
            heuristic_routes,
            f"Instance {vrp.instance_id} - Clarke & Wright\n"
            f"dist={heuristic_distance:.3f}, time={heuristic_time:.6f}s",
            ax=axes[0]
        )

        vrp.plot_routes(
            best_routes,
            f"Instance {vrp.instance_id} - Exact Best\n"
            f"dist={best_distance:.3f}, time={exact_time:.6f}s",
            ax=axes[1]
        )

        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"CWvsExact/{save_prefix}_instance_{vrp.instance_id}_routes.png", dpi=200, bbox_inches="tight")
        plt.show()

    return result


def plot_summary(results, save_prefix=None):
    instance_ids = [r["instance_id"] for r in results]
    labels = [f"Inst {i}" for i in instance_ids]

    heuristic_distances = [r["heuristic_distance"] for r in results]
    optimal_distances = [r["optimal_distance"] for r in results]

    heuristic_times = [r["heuristic_time_sec"] for r in results]
    exact_times = [r["exact_time_sec"] for r in results]

    quality_scores = [
        r["normalized_quality_percent"] if r["normalized_quality_percent"] is not None else 0.0
        for r in results
    ]
    gap_scores = [
        r["gap_to_optimal_percent"] if r["gap_to_optimal_percent"] is not None else 0.0
        for r in results
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Distance comparison
    axes[0, 0].bar(x - width/2, heuristic_distances, width, label="Clarke-Wright")
    axes[0, 0].bar(x + width/2, optimal_distances, width, label="Exact Best")
    axes[0, 0].set_title("Distance Comparison")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].set_ylabel("Total Distance")
    axes[0, 0].legend()
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    # Runtime comparison
    axes[0, 1].bar(x - width/2, heuristic_times, width, label="Clarke-Wright")
    axes[0, 1].bar(x + width/2, exact_times, width, label="Exact Search")
    axes[0, 1].set_title("Runtime Comparison")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].set_ylabel("Seconds")
    axes[0, 1].legend()
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    # Optimality gap
    axes[1, 0].bar(x, gap_scores)
    axes[1, 0].set_title("Gap to Optimal (%)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_ylabel("% above optimal")
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    # Normalized quality
    axes[1, 1].bar(x, quality_scores)
    axes[1, 1].set_title("Normalized Quality (%)")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_ylabel("% between worst and best")
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"CWvsExact/{save_prefix}_summary.png", dpi=200, bbox_inches="tight")
    plt.show()


def print_results_to_txt_exact(results):
    #Print the result of instance 1 to 4 in .txt files with the format "instance_{instance_id}.txt":
    #Vehicle 1: 0 -> 3 -> 5 -> 0
    #...
    for r in results:
        with open(f"exactBruteForce/instance_{r['instance_id']}.txt", "w") as f:
            for i, route in enumerate(r['optimal_routes']):
                route_str = " -> ".join(str(c) for c in route)
                f.write(f"Vehicle {i + 1}: 0 -> {route_str} -> 0\n")
def print_results_to_txt_CW(results):
    for r in results:
        with open(f"clarkeWright/instance_{r['instance_id']}.txt", "w") as f:
            for i, route in enumerate(r['heuristic_routes']):
                route_str = " -> ".join(str(c) for c in route)
                f.write(f"Vehicle {i + 1}: 0 -> {route_str} -> 0\n")

def print_results(results):
    print("\n" + "=" * 90)
    print("VRP COMPARISON RESULTS")
    print("=" * 90)

    for r in results:
        print(f"\nInstance {r['instance_id']}")
        print("-" * 90)
        print(f"Customers: {r['num_customers']}, Vehicles: {r['num_vehicles']}, Capacity: {r['capacity']}")
        print(f"Heuristic valid: {r['heuristic_valid']} ({r['heuristic_validation_message']})")
        print(f"Heuristic routes: {r['heuristic_routes']}")
        print(f"Heuristic distance: {r['heuristic_distance']:.6f}")
        print(f"Heuristic runtime: {r['heuristic_time_sec']:.6f} s")

        print(f"Exact best routes: {r['optimal_routes']}")
        print(f"Exact best distance: {r['optimal_distance']:.6f}")
        print(f"Exact worst routes: {r['worst_routes']}")
        print(f"Exact worst distance: {r['worst_distance']:.6f}")
        print(f"Exact runtime: {r['exact_time_sec']:.6f} s")

        if r["heuristic_valid"]:
            print(f"Gap to optimal: {r['gap_to_optimal_percent']:.4f}%")
            print(f"Normalized quality: {r['normalized_quality_percent']:.4f}%")
        else:
            print("Gap to optimal: N/A (heuristic solution invalid)")
            print("Normalized quality: N/A (heuristic solution invalid)")


if __name__ == "__main__":
    instance_files = ["synthetic_data_1.csv"] #Add the instance files here from the list ["instance_1.csv", "instance_2.csv", "instance_3.csv", "instance_4.csv", "synthetic_data_1.csv"]
    results = []
    for filename in instance_files:
        vrp = load_instance(filename)
        result = compare_instance(vrp, make_plot=True, save_prefix="vrp")
        results.append(result)
    print_results_to_txt_exact(results)
    print_results_to_txt_CW(results)
    print_results(results)
    plot_summary(results, save_prefix="vrp")