import json
import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint


# ============================================================
# DATA MODEL
# ============================================================

@dataclass(order=True)
class Label:
    sort_key: float = field(init=False, repr=False)
    node: int
    visited_mask: int
    load: int
    cost: float
    route: tuple

    def __post_init__(self):
        self.sort_key = self.cost


@dataclass(frozen=True)
class RouteColumn:
    customers_mask: int
    ordered_route: tuple
    load: int
    cost: float


class VRPSPPRC:
    def __init__(self, instance):
        self.instance_id = instance["instance_id"]
        self.num_vehicles = int(instance["Nv"])
        self.capacity = int(instance["C"])
        self.depot = None
        self.customers = []  # customer ids 1..n, stored as (x, y, demand)

        for customer in instance["customers"]:
            cid = int(customer["customer_id"])
            x = float(customer["x"])
            y = float(customer["y"])
            d = int(customer["demand"])
            if cid == 0:
                self.depot = (x, y)
            else:
                self.customers.append((x, y, d))

        if self.depot is None:
            raise ValueError(f"Instance {self.instance_id} missing depot")

        self.num_customers = len(self.customers)
        self.distance_matrix = self._compute_distances()
        self.demands = np.array([0] + [c[2] for c in self.customers], dtype=int)

        # Nearest-neighbor lists for stable restricted label-setting
        self.nearest_neighbors = {}
        for i in range(self.num_customers + 1):
            nbrs = [j for j in range(1, self.num_customers + 1) if j != i]
            nbrs.sort(key=lambda j: self.distance_matrix[i, j])
            self.nearest_neighbors[i] = nbrs

    def _compute_distances(self):
        points = np.array([self.depot] + [(c[0], c[1]) for c in self.customers], dtype=float)
        delta = points[:, None, :] - points[None, :, :]
        return np.sqrt((delta ** 2).sum(axis=2))

    def route_demand(self, route):
        return sum(self.customers[c - 1][2] for c in route)

    def route_distance(self, route):
        if not route:
            return 0.0
        total = self.distance_matrix[0, route[0]]
        for i in range(len(route) - 1):
            total += self.distance_matrix[route[i], route[i + 1]]
        total += self.distance_matrix[route[-1], 0]
        return float(total)

    def total_distance(self, routes):
        return float(sum(self.route_distance(route) for route in routes))

    def validate_routes(self, routes):
        seen = []
        for route in routes:
            if self.route_demand(route) > self.capacity:
                return False
            seen.extend(route)

        return (
            len(routes) <= self.num_vehicles
            and sorted(seen) == list(range(1, self.num_customers + 1))
        )

    def route_details(self, routes):
        out = []
        for idx, route in enumerate(routes, start=1):
            out.append(
                {
                    "vehicle": idx,
                    "route": [0] + list(route) + [0],
                    "load": self.route_demand(route),
                    "distance": round(self.route_distance(route), 6),
                }
            )
        return out

    def plot_routes(self, routes, output_path, runtime_sec=0.0):
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
        for route_idx, route in enumerate(routes, start=1):
            points = [self.depot]
            points.extend((self.customers[c - 1][0], self.customers[c - 1][1]) for c in route)
            points.append(self.depot)
            points = np.array(points)

            ax.plot(
                points[:, 0],
                points[:, 1],
                linewidth=2.2,
                color=colors[(route_idx - 1) % len(colors)],
                label=f"Route {route_idx}: {list(route)}",
                zorder=2,
            )

        ax.set_title(
            f"{self.instance_id} - Label-Setting SPPRC\n"
            f"dist={self.total_distance(routes):.3f}, time={runtime_sec:.6f}s",
            fontsize=16,
        )
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.grid(True, alpha=0.5)
        ax.legend(fontsize=8, loc="best")

        plt.tight_layout()
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# JSON
# ============================================================

def load_instances(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# INITIAL ROUTES / COLUMNS
# ============================================================

def nearest_neighbor_seed_routes(vrp: VRPSPPRC):
    remaining = set(range(1, vrp.num_customers + 1))
    routes = []

    for _ in range(vrp.num_vehicles):
        if not remaining:
            break

        route = []
        load = 0
        current = 0

        while remaining:
            feasible = [c for c in remaining if load + vrp.demands[c] <= vrp.capacity]
            if not feasible:
                break
            nxt = min(feasible, key=lambda c: vrp.distance_matrix[current, c])
            route.append(nxt)
            remaining.remove(nxt)
            load += vrp.demands[nxt]
            current = nxt

        if route:
            routes.append(route)

    while remaining:
        inserted = False
        c = next(iter(remaining))
        for route in routes:
            if vrp.route_demand(route) + vrp.demands[c] <= vrp.capacity:
                route.append(c)
                remaining.remove(c)
                inserted = True
                break
        if not inserted:
            routes.append([c])
            remaining.remove(c)

    return routes


def routes_to_columns(vrp: VRPSPPRC, routes):
    cols = []
    seen = set()
    for route in routes:
        mask = 0
        for c in route:
            mask |= 1 << (c - 1)
        if mask not in seen:
            cols.append(
                RouteColumn(
                    customers_mask=mask,
                    ordered_route=tuple(route),
                    load=vrp.route_demand(route),
                    cost=vrp.route_distance(route),
                )
            )
            seen.add(mask)
    return cols


# ============================================================
# LABEL-SETTING SPPRC
# ============================================================

def dominates(a: Label, b: Label):
    if a.node != b.node:
        return False
    if a.load > b.load:
        return False
    if a.cost > b.cost + 1e-12:
        return False
    # visited subset dominance
    if (a.visited_mask | b.visited_mask) != b.visited_mask:
        return False
    return True


def pricing_label_setting(
    vrp: VRPSPPRC,
    max_labels_per_node=60,
    k_nearest=12,
    max_total_expansions=12000,
    max_route_stops=12,
    time_limit_sec=20.0,
    use_reduced_cost=True,
    duals=None,
    verbose=False,
):
    """
    Stable restricted label-setting route generator.
    Intentionally restricted for benchmark-sized instances.
    """
    D = vrp.distance_matrix
    labels_at_node = {i: [] for i in range(vrp.num_customers + 1)}

    start = Label(
        node=0,
        visited_mask=0,
        load=0,
        cost=0.0,
        route=tuple(),
    )

    active = deque([start])
    complete_routes = {}
    iterations = 0
    t_start = time.perf_counter()

    while active and iterations < max_total_expansions:
        if time.perf_counter() - t_start > time_limit_sec:
            if verbose:
                print(f"      pricing stopped by time limit at {iterations} expansions")
            break

        label = active.popleft()
        iterations += 1

        if verbose and iterations % 2000 == 0:
            print(f"      pricing expansions={iterations}, active={len(active)}")

        if len(label.route) >= max_route_stops:
            continue

        candidates = []
        for j in vrp.nearest_neighbors[label.node][:k_nearest]:
            bit = 1 << (j - 1)
            if label.visited_mask & bit:
                continue
            if label.load + vrp.demands[j] > vrp.capacity:
                continue
            candidates.append(j)

        for j in candidates:
            bit = 1 << (j - 1)
            arc_cost = D[label.node, j]
            if use_reduced_cost and duals is not None:
                arc_cost -= duals[j - 1]

            new_label = Label(
                node=j,
                visited_mask=label.visited_mask | bit,
                load=label.load + vrp.demands[j],
                cost=label.cost + arc_cost,
                route=label.route + (j,),
            )

            bucket = labels_at_node[j]
            dominated = False
            survivors = []

            for old in bucket:
                if dominates(old, new_label):
                    dominated = True
                    break
                if not dominates(new_label, old):
                    survivors.append(old)

            if dominated:
                continue

            survivors.append(new_label)
            survivors.sort(key=lambda x: x.cost)
            if len(survivors) > max_labels_per_node:
                survivors = survivors[:max_labels_per_node]
            labels_at_node[j] = survivors

            active.append(new_label)

            # Close route to depot and store
            ordered = new_label.route
            mask = new_label.visited_mask
            load = new_label.load
            real_cost = vrp.route_distance(list(ordered))

            existing = complete_routes.get(mask)
            if existing is None or real_cost < existing.cost - 1e-12:
                complete_routes[mask] = RouteColumn(
                    customers_mask=mask,
                    ordered_route=ordered,
                    load=load,
                    cost=real_cost,
                )

    return list(complete_routes.values()), iterations


# ============================================================
# MASTER PROBLEM
# ============================================================

def solve_set_partitioning(vrp: VRPSPPRC, columns):
    n = vrp.num_customers
    m = len(columns)

    A_cover = np.zeros((n, m), dtype=float)
    A_vehicle = np.zeros((1, m), dtype=float)
    c = np.zeros(m, dtype=float)

    for j, col in enumerate(columns):
        c[j] = col.cost
        A_vehicle[0, j] = 1.0
        mask = col.customers_mask
        for i in range(n):
            if mask & (1 << i):
                A_cover[i, j] = 1.0

    constraints = [
        LinearConstraint(A_cover, np.ones(n), np.ones(n)),
        LinearConstraint(A_vehicle, -np.inf, np.array([vrp.num_vehicles], dtype=float)),
    ]

    integrality = np.ones(m, dtype=int)
    bounds = Bounds(np.zeros(m), np.ones(m))

    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    return res


# ============================================================
# COLUMN POOL CONSTRUCTION
# ============================================================

def build_route_pool(vrp: VRPSPPRC, verbose=False):
    seed_routes = nearest_neighbor_seed_routes(vrp)
    pool = routes_to_columns(vrp, seed_routes)

    seen = {col.customers_mask for col in pool}

    # singletons
    for i in range(1, vrp.num_customers + 1):
        mask = 1 << (i - 1)
        if mask not in seen:
            pool.append(
                RouteColumn(
                    customers_mask=mask,
                    ordered_route=(i,),
                    load=vrp.demands[i],
                    cost=vrp.route_distance([i]),
                )
            )
            seen.add(mask)

    # pass 1: pure distance-guided
    if verbose:
        print("    pass 1: distance-guided label setting")
    cols1, it1 = pricing_label_setting(
        vrp,
        max_labels_per_node=60,
        k_nearest=12,
        max_total_expansions=12000,
        max_route_stops=min(12, vrp.capacity),
        time_limit_sec=20.0,
        use_reduced_cost=False,
        duals=None,
        verbose=verbose,
    )
    for col in cols1:
        if col.customers_mask not in seen:
            pool.append(col)
            seen.add(col.customers_mask)

    # pass 2: pseudo reduced-cost
    if verbose:
        print("    pass 2: pseudo reduced-cost label setting")
    pseudo_duals = np.array([0.20 * vrp.distance_matrix[0, i] for i in range(1, vrp.num_customers + 1)])
    cols2, it2 = pricing_label_setting(
        vrp,
        max_labels_per_node=60,
        k_nearest=12,
        max_total_expansions=12000,
        max_route_stops=min(12, vrp.capacity),
        time_limit_sec=20.0,
        use_reduced_cost=True,
        duals=pseudo_duals,
        verbose=verbose,
    )
    for col in cols2:
        if col.customers_mask not in seen:
            pool.append(col)
            seen.add(col.customers_mask)

    diagnostics = {
        "seed_routes": len(seed_routes),
        "pool_size": len(pool),
        "label_iterations_pass1": it1,
        "label_iterations_pass2": it2,
    }
    return pool, diagnostics


# ============================================================
# SOLVER
# ============================================================

def solve_instance_spprc(instance, verbose=False):
    vrp = VRPSPPRC(instance)

    if verbose:
        print(f"\n=== Solving {vrp.instance_id} ({vrp.num_customers} customers) ===")

    t0 = time.perf_counter()
    pool, diagnostics = build_route_pool(vrp, verbose=verbose)
    t1 = time.perf_counter()

    if verbose:
        print(f"    route pool built: {diagnostics['pool_size']} columns")

    res = solve_set_partitioning(vrp, pool)
    t2 = time.perf_counter()

    if not res.success:
        raise RuntimeError(f"Master MILP failed on {vrp.instance_id}: {res.message}")

    chosen_routes = []
    for j, x in enumerate(res.x):
        if x > 0.5:
            chosen_routes.append(list(pool[j].ordered_route))

    runtime = t2 - t0
    valid = vrp.validate_routes(chosen_routes)

    result = {
        "id": vrp.instance_id,
        "customers": vrp.num_customers,
        "vehicles_allowed": vrp.num_vehicles,
        "vehicles_used": len(chosen_routes),
        "distance": round(vrp.total_distance(chosen_routes), 6),
        "runtime": runtime,
        "valid": valid,
        "routes": vrp.route_details(chosen_routes),
        "diagnostics": diagnostics,
        "pool_build_time": t1 - t0,
        "master_time": t2 - t1,
    }
    return vrp, chosen_routes, result


# ============================================================
# BENCHMARK
# ============================================================

def run_benchmark(json_path="setA_random_instances_grouped.json", output_dir="spprc"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    instances = load_instances(json_path)
    results = []

    for instance in instances:
        vrp, routes, result = solve_instance_spprc(instance, verbose=True)
        results.append(result)

        image_path = out / f"outputs_png/{vrp.instance_id}_routes.png"
        vrp.plot_routes(routes, image_path, runtime_sec=result["runtime"])
        print(f"Saved route plot: {image_path}")
        print(f"  distance={result['distance']:.6f}, vehicles={result['vehicles_used']}, valid={result['valid']}")

    json_out = out / "benchmark_results_spprc.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results JSON: {json_out}")
    return results


# ============================================================
# ANALYSIS PLOTS
# ============================================================

def plot_summary(results, output_dir="benchmark_output_spprc"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ids = [r["id"] for r in results]
    distances = [r["distance"] for r in results]
    runtimes = [r["runtime"] for r in results]
    customers = [r["customers"] for r in results]
    vehicles_used = [r["vehicles_used"] for r in results]
    pool_sizes = [r["diagnostics"]["pool_size"] for r in results]
    pass1_iters = [r["diagnostics"]["label_iterations_pass1"] for r in results]
    pass2_iters = [r["diagnostics"]["label_iterations_pass2"] for r in results]
    avg_route_dist = [r["distance"] / max(r["vehicles_used"], 1) for r in results]

    label_step = max(1, len(ids) // 12)
    display_labels = [ids[i] if i % label_step == 0 else "" for i in range(len(ids))]

    plt.figure(figsize=(14, 6))
    plt.bar(ids, distances)
    plt.title("Distance (Label-Setting SPPRC)")
    plt.ylabel("Total Distance")
    plt.xticks(range(len(ids)), display_labels, rotation=45, fontsize=8)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "distance_spprc.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.bar(ids, runtimes)
    plt.title("Runtime (Label-Setting SPPRC)")
    plt.ylabel("Seconds")
    plt.xticks(range(len(ids)), display_labels, rotation=45, fontsize=8)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "runtime_spprc.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.bar(ids, vehicles_used)
    plt.title("Vehicles Used")
    plt.ylabel("Vehicle Count")
    plt.xticks(range(len(ids)), display_labels, rotation=45, fontsize=8)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "vehicles_used_spprc.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.bar(ids, pool_sizes)
    plt.title("Generated Route Pool Size")
    plt.ylabel("Columns")
    plt.xticks(range(len(ids)), display_labels, rotation=45, fontsize=8)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "columns_generated_spprc.png", dpi=180, bbox_inches="tight")
    plt.close()

    x = np.arange(len(ids))
    w = 0.38
    plt.figure(figsize=(14, 6))
    plt.bar(x - w / 2, pass1_iters, w, label="Pass 1")
    plt.bar(x + w / 2, pass2_iters, w, label="Pass 2")
    plt.title("Label-Setting Iterations by Pass")
    plt.ylabel("Iterations")
    plt.xticks(x, display_labels, rotation=45, fontsize=8)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "label_iterations_spprc.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(customers, runtimes)
    plt.xlabel("Customers")
    plt.ylabel("Runtime")
    plt.title("Runtime vs Problem Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "runtime_vs_size_spprc.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(customers, pool_sizes)
    plt.xlabel("Customers")
    plt.ylabel("Route Pool Size")
    plt.title("Route Pool Size vs Problem Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "poolsize_vs_size_spprc.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.bar(ids, avg_route_dist)
    plt.title("Average Distance per Used Vehicle")
    plt.ylabel("Distance / Vehicle")
    plt.xticks(range(len(ids)), display_labels, rotation=45, fontsize=8)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "avg_route_distance_spprc.png", dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Saved summary plots in: {out}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    benchmark_results = run_benchmark(
        json_path="setA_random_instances_grouped.json",
        output_dir="spprc",
    )
    plot_summary(benchmark_results, output_dir="spprc")
t_