import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import pygad
except ImportError as exc:
    raise SystemExit(
        "PyGAD is not installed. Run: pip install pygad\n"
        "Then rerun this script."
    ) from exc


# ============================================================
# DISTANCE / ROUTE HELPERS
# ============================================================

def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_distance_matrix(customers: List[Dict[str, Any]]) -> np.ndarray:
    """
    Assumes customer_id values match row/column positions:
    0, 1, 2, ..., n
    which is true for your official_instances.json.
    """
    coords = [(float(c["x"]), float(c["y"])) for c in customers]
    n = len(coords)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dist[i, j] = euclidean_distance(coords[i], coords[j])
    return dist


def route_cost(route: List[int], dist: np.ndarray) -> float:
    if len(route) < 2:
        return 0.0
    return float(sum(dist[route[i], route[i + 1]] for i in range(len(route) - 1)))


def total_cost(routes: List[List[int]], dist: np.ndarray) -> float:
    return float(sum(route_cost(route, dist) for route in routes))


# ============================================================
# DECODER
# ============================================================

def decode_permutation(
    permutation: List[int],
    demands: Dict[int, int],
    capacity: int,
    max_vehicles: int,
) -> Tuple[List[List[int]], bool]:
    """
    Returns routes in full form, e.g.
    [0, 3, 2, 0], [0, 1, 4, 0]
    """
    routes: List[List[int]] = []
    current_route: List[int] = [0]
    current_load = 0

    for customer_id in permutation:
        demand = demands[customer_id]

        if demand > capacity:
            return [], False

        if current_load + demand <= capacity:
            current_route.append(customer_id)
            current_load += demand
        else:
            current_route.append(0)
            routes.append(current_route)
            current_route = [0, customer_id]
            current_load = demand

    current_route.append(0)
    routes.append(current_route)

    feasible = len(routes) <= max_vehicles
    return routes, feasible


# ============================================================
# PERMUTATION OPERATORS
# ============================================================

def ordered_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    size = len(parent1)
    child = np.full(size, -1, dtype=int)

    left, right = sorted(random.sample(range(size), 2))
    child[left:right + 1] = parent1[left:right + 1]

    remaining = [gene for gene in parent2 if gene not in child]
    fill_positions = [i for i in range(size) if child[i] == -1]

    for pos, gene in zip(fill_positions, remaining):
        child[pos] = gene

    return child


def crossover_func(parents: np.ndarray, offspring_size: Tuple[int, int], ga_instance) -> np.ndarray:
    offspring = []
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0]].astype(int)
        p2 = parents[(k + 1) % parents.shape[0]].astype(int)
        offspring.append(ordered_crossover(p1, p2))
    return np.array(offspring, dtype=int)


def mutation_func(offspring: np.ndarray, ga_instance) -> np.ndarray:
    mutated = offspring.copy()
    for i in range(mutated.shape[0]):
        if random.random() < ga_instance.mutation_probability:
            a, b = random.sample(range(mutated.shape[1]), 2)
            mutated[i, a], mutated[i, b] = mutated[i, b], mutated[i, a]
    return mutated


# ============================================================
# SOLVER
# ============================================================

class OfficialPyGADVRP:
    def __init__(
        self,
        instance: Dict[str, Any],
        num_generations: int = 400,
        sol_per_pop: int = 80,
        num_parents_mating: int = 20,
        mutation_probability: float = 0.2,
        seed: int = 42,
    ):
        self.instance_id = instance["instance_id"]
        self.Nv = int(instance["Nv"])
        self.C = int(instance["C"])
        self.raw_customers = instance["customers"]

        self.num_generations = int(num_generations)
        self.sol_per_pop = int(sol_per_pop)
        self.num_parents_mating = int(num_parents_mating)
        self.mutation_probability = float(mutation_probability)
        self.seed = int(seed)

        random.seed(self.seed)
        np.random.seed(self.seed)

        # Keep full customer list including depot for distance matrix
        self.customers = self.raw_customers

        # Extract depot and customer-only structures
        self.depot = None
        self.customer_map: Dict[int, Tuple[float, float, int]] = {}
        self.customer_ids: List[int] = []
        self.demands: Dict[int, int] = {}

        for c in self.customers:
            cid = int(c["customer_id"])
            x = float(c["x"])
            y = float(c["y"])
            demand = int(c["demand"])

            self.demands[cid] = demand

            if cid == 0:
                self.depot = (x, y)
            else:
                self.customer_ids.append(cid)
                self.customer_map[cid] = (x, y, demand)

        if self.depot is None:
            raise ValueError(f"{self.instance_id}: depot (customer_id=0) not found.")

        self.customer_ids.sort()
        self.dist = build_distance_matrix(self.customers)

        self.best_distance = float("inf")
        self.best_routes: List[List[int]] = []
        self.best_valid = False
        self.last_runtime = 0.0

    def initial_population(self) -> np.ndarray:
        base = np.array(self.customer_ids, dtype=int)
        population = []
        for _ in range(self.sol_per_pop):
            candidate = base.copy()
            np.random.shuffle(candidate)
            population.append(candidate)
        return np.array(population, dtype=int)

    def evaluate(self, solution: np.ndarray) -> Tuple[float, float, bool, List[List[int]]]:
        perm = [int(x) for x in solution.tolist()]

        if sorted(perm) != sorted(self.customer_ids):
            penalty = 1e9
            return 1.0 / (1.0 + penalty), penalty, False, []

        routes, feasible = decode_permutation(perm, self.demands, self.C, self.Nv)
        distance = total_cost(routes, self.dist) if routes else 1e9

        if not feasible:
            overflow = max(0, len(routes) - self.Nv)
            penalty_cost = distance + 100000.0 * overflow + 100000.0
            return 1.0 / (1.0 + penalty_cost), penalty_cost, False, routes

        fitness = 1.0 / (1.0 + distance)
        return fitness, distance, True, routes

    def fitness_func(self, ga_instance, solution, solution_idx):
        fitness, distance, valid, routes = self.evaluate(solution)

        if valid and distance < self.best_distance:
            self.best_distance = distance
            self.best_routes = [route[:] for route in routes]
            self.best_valid = True

        return fitness

    def total_distance(self) -> float:
        if not self.best_routes:
            return 0.0
        return total_cost(self.best_routes, self.dist)

    def solve(self) -> Dict[str, Any]:
        start = time.perf_counter()

        ga = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            fitness_func=self.fitness_func,
            sol_per_pop=self.sol_per_pop,
            num_genes=len(self.customer_ids),
            initial_population=self.initial_population(),
            gene_type=int,
            parent_selection_type="sss",
            keep_parents=4,
            crossover_type=crossover_func,
            mutation_type=mutation_func,
            mutation_probability=self.mutation_probability,
            suppress_warnings=True,
            random_seed=self.seed,
        )
        ga.run()

        best_solution, best_fitness, _ = ga.best_solution()
        _, final_distance, final_valid, final_routes = self.evaluate(best_solution)
        runtime = time.perf_counter() - start
        self.last_runtime = runtime

        # Store final solution on object so plotting works
        self.best_routes = [route[:] for route in final_routes]
        self.best_distance = final_distance
        self.best_valid = final_valid

        return {
            "instance_id": self.instance_id,
            "distance": round(float(final_distance), 2),
            "runtime": float(runtime),
            "valid": bool(final_valid),
            "routes": [[int(x) for x in route] for route in final_routes],
            "fitness": float(best_fitness),
            "Nv": self.Nv,
            "C": self.C,
            "customers": len(self.customer_ids),
        }

    def plot_routes(self, output_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(12, 8))

        depot_x, depot_y = self.depot

        # Depot
        ax.scatter(
            depot_x,
            depot_y,
            c="black",
            marker="X",
            s=220,
            label="Depot",
            zorder=5,
        )
        ax.text(depot_x + 0.25, depot_y + 0.25, "0", fontsize=11, weight="bold")

        # Customers
        xs = [self.customer_map[cid][0] for cid in sorted(self.customer_map)]
        ys = [self.customer_map[cid][1] for cid in sorted(self.customer_map)]
        ax.scatter(xs, ys, c="black", s=40, zorder=4)

        for cid in sorted(self.customer_map):
            x, y, _ = self.customer_map[cid]
            ax.text(x + 0.25, y + 0.25, str(cid), fontsize=9)

        # Routes
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for route_idx, route in enumerate(self.best_routes, start=1):
            # route is like [0, 3, 2, 0]
            route_customers = [node for node in route if node != 0]

            points = [(depot_x, depot_y)]
            for customer_id in route_customers:
                x, y, _ = self.customer_map[customer_id]
                points.append((x, y))
            points.append((depot_x, depot_y))
            points = np.array(points, dtype=float)

            ax.plot(
                points[:, 0],
                points[:, 1],
                linewidth=2.2,
                color=colors[(route_idx - 1) % len(colors)],
                marker="o",
                markersize=4,
                label=f"Route {route_idx}: {route}",
                zorder=2,
            )

        ax.set_title(
            f"{self.instance_id} - PyGAD VRP\n"
            f"dist={self.total_distance():.3f}, time={self.last_runtime:.6f}s",
            fontsize=16,
        )
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.grid(True, alpha=0.5)
        ax.legend(fontsize=8, loc="best")
        ax.set_aspect("equal", adjustable="box")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# OUTPUT HELPERS
# ============================================================

def write_instance_txt(result: Dict[str, Any], output_dir: Path) -> None:
    txt_path = output_dir / f"{result['instance_id']}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for idx, route in enumerate(result["routes"], start=1):
            route_str = " -> ".join(str(node) for node in route)
            f.write(f"Vehicle {idx}: {route_str}\n")


def save_plot(results: List[Dict[str, Any]], output_dir: Path) -> None:
    labels = [r["instance_id"].replace("_", " ").title() for r in results]
    values = [r["distance"] for r in results]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(labels, values)
    plt.title("CVRP Objective Values for Official Instances (PyGAD-VRP)")
    plt.xlabel("Instance")
    plt.ylabel("Objective Value")
    plt.grid(axis="y", alpha=0.6)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "all_instance_costs.png", dpi=200)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / ".." / "c5n25.json"
    output_dir = base_dir / "c5n25_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        instances = json.load(f)

    results: List[Dict[str, Any]] = []
    results.clear()
    for instance in instances:
        solver = OfficialPyGADVRP(instance)
        result = solver.solve()
        results.append(result)

        write_instance_txt(result, output_dir)

        print(
            f"[PyGAD-VRP] {result['instance_id']}: "
            f"distance={result['distance']:.2f}, "
            f"runtime={result['runtime']:.4f}s, "
            f"valid={result['valid']}"
            )
        solver.plot_routes(output_dir / "png" / f"{result['instance_id']}_routes.png")

        save_plot(results, output_dir)


    print(f"Saved outputs to: {output_dir}")

if __name__ == "__main__":
    main()
