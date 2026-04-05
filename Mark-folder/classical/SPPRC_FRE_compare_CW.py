import csv
import math
import itertools
import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog, milp, Bounds, LinearConstraint


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class RouteColumn:
    customers_set: frozenset
    ordered_route: tuple
    cost: float


class VRP:
    def __init__(self, instance_id, num_customers, num_vehicles, vehicle_capacity, depot_location):
        self.instance_id = instance_id
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.depot_location = depot_location
        self.customers = []   # list of tuples: (x, y, demand)
        self.distance_matrix = None

    def calculate_distance_matrix(self):
        points = [self.depot_location] + [(c[0], c[1]) for c in self.customers]
        n = len(points)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                D[i][j] = math.hypot(points[j][0] - points[i][0], points[j][1] - points[i][1])
        return D

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

    def total_distance(self, routes):
        return sum(self.route_distance(route) for route in routes)

    def validate_routes(self, routes):
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
            return False, f"Expected no more than {self.num_vehicles} vehicles, got {len(routes)}"

        return True, "Valid"

    def plot_routes(self, routes, title, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 7))

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        # Depot
        ax.scatter(
            self.depot_location[0],
            self.depot_location[1],
            c='black',
            marker='X',
            s=180,
            label='Depot'
        )
        ax.text(self.depot_location[0] + 0.08, self.depot_location[1] + 0.08, "0", fontsize=10)

        # Customers
        for idx, customer in enumerate(self.customers, start=1):
            ax.scatter(customer[0], customer[1], c='black', marker='o', s=45)
            ax.text(customer[0] + 0.08, customer[1] + 0.08, str(idx), fontsize=10)

        # Routes
        for r_idx, route in enumerate(routes):
            pts = [self.depot_location] + [(self.customers[c - 1][0], self.customers[c - 1][1]) for c in route] + [self.depot_location]
            pts = np.array(pts)
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=colors[r_idx % len(colors)],
                linewidth=2.5,
                label=f'Route {r_idx + 1}: {list(route)}'
            )

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.5)
        ax.legend(fontsize=9)


# ============================================================
# LOADING
# ============================================================

def load_instance(filename):
    with open(filename, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    instance_id = rows[0]["instance_id"]
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

    vrp = VRP(
        instance_id=instance_id,
        num_customers=len(customers),
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        depot_location=depot_location
    )
    vrp.customers = customers
    vrp.distance_matrix = vrp.calculate_distance_matrix()
    return vrp


# ============================================================
# CLARKE & WRIGHT
# ============================================================

def clarke_and_wright(vrp):
    """
    Parallel Clarke-Wright savings algorithm.
    """
    savings = []
    for i in range(1, vrp.num_customers + 1):
        for j in range(i + 1, vrp.num_customers + 1):
            s = vrp.distance_matrix[0][i] + vrp.distance_matrix[0][j] - vrp.distance_matrix[i][j]
            savings.append((s, i, j))

    savings.sort(reverse=True)

    routes = [[i] for i in range(1, vrp.num_customers + 1)]

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

        if vrp.route_demand(route_i) + vrp.route_demand(route_j) > vrp.vehicle_capacity:
            continue

        # i and j must be endpoints
        if i not in (route_i[0], route_i[-1]):
            continue
        if j not in (route_j[0], route_j[-1]):
            continue

        original_i = route_i
        original_j = route_j

        if route_i[0] == i:
            route_i = route_i[::-1]
        if route_j[-1] == j:
            route_j = route_j[::-1]

        merged = route_i + route_j

        routes.remove(original_i)
        routes.remove(original_j)
        routes.append(merged)

    return routes


# ============================================================
# ROUTE ENUMERATION FOR SMALL-CAPACITY SPPRC PRICING
# ============================================================

def best_order_for_subset(vrp, subset):
    """
    For a subset of customers, return the best ordered route and its cost.
    This is exact for the subset by checking all permutations.
    """
    best_cost = float("inf")
    best_perm = None
    for perm in itertools.permutations(subset):
        cost = vrp.route_distance(list(perm))
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    return tuple(best_perm), best_cost


def generate_all_feasible_columns(vrp):
    """
    Since capacity is small in your instances, we can enumerate all feasible routes
    up to vehicle capacity. This plays the role of the SPPRC pricing search space.
    """
    columns = []
    seen = set()

    for k in range(1, vrp.vehicle_capacity + 1):
        for subset in itertools.combinations(range(1, vrp.num_customers + 1), k):
            demand = sum(vrp.customers[c - 1][2] for c in subset)
            if demand > vrp.vehicle_capacity:
                continue

            ordered_route, cost = best_order_for_subset(vrp, subset)
            cset = frozenset(subset)
            if cset not in seen:
                columns.append(RouteColumn(
                    customers_set=cset,
                    ordered_route=ordered_route,
                    cost=cost
                ))
                seen.add(cset)

    return columns


# ============================================================
# COLUMN GENERATION MASTER
# ============================================================

def build_master_matrices(vrp, columns):
    """
    Equality rows:
      - one row per customer: covered exactly once
      - one row for number of routes: exactly Nv
    """
    n_customers = vrp.num_customers
    n_cols = len(columns)

    A_eq = np.zeros((n_customers + 1, n_cols), dtype=float)
    b_eq = np.ones(n_customers + 1, dtype=float)
    b_eq[-1] = vrp.num_vehicles

    for j, col in enumerate(columns):
        for c in col.customers_set:
            A_eq[c - 1, j] = 1.0
        A_eq[-1, j] = 1.0   # route-count row

    c = np.array([col.cost for col in columns], dtype=float)
    return c, A_eq, b_eq


def solve_rmp_lp(vrp, columns):
    c, A_eq, b_eq = build_master_matrices(vrp, columns)
    res = linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0, None)] * len(columns),
        method="highs"
    )
    if not res.success:
        raise RuntimeError(f"RMP LP failed: {res.message}")
    return res


def solve_master_integer(vrp, columns):
    c, A_eq, b_eq = build_master_matrices(vrp, columns)

    constraints = [LinearConstraint(A_eq, b_eq, b_eq)]
    integrality = np.ones(len(columns), dtype=int)
    bounds = Bounds(np.zeros(len(columns)), np.ones(len(columns)))

    res = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds
    )
    if not res.success:
        raise RuntimeError(f"Integer master failed: {res.message}")
    return res


# ============================================================
# SPPRC-STYLE PRICING
# ============================================================

def pricing_by_reduced_cost(vrp, all_columns, current_columns, dual_customer, dual_vehicle):
    """
    Reduced cost:
        rc(r) = c_r - sum(pi_i for i in route) - sigma

    We search across all feasible routes not yet in the RMP.
    This is a small-instance exact pricing oracle over the SPPRC route space.
    """
    current_sets = {col.customers_set for col in current_columns}

    best_rc = float("inf")
    best_col = None

    for col in all_columns:
        if col.customers_set in current_sets:
            continue
        rc = col.cost - sum(dual_customer[c - 1] for c in col.customers_set) - dual_vehicle
        if rc < best_rc:
            best_rc = rc
            best_col = col

    return best_rc, best_col


def spprc_column_generation(vrp, verbose=True):
    """
    Column generation with:
      - initial feasible basis from Clarke-Wright
      - pricing by reduced-cost search over feasible route space
      - final integer master solve
    """
    # Full feasible route space for pricing
    all_columns = generate_all_feasible_columns(vrp)

    # Seed with CW routes so the initial RMP is feasible
    cw_routes = clarke_and_wright(vrp)
    valid, msg = vrp.validate_routes(cw_routes)
    if not valid:
        raise RuntimeError(f"Initial Clarke-Wright solution is not feasible for CG start: {msg}")

    current_columns = []
    used_sets = set()

    for route in cw_routes:
        cset = frozenset(route)
        if cset not in used_sets:
            current_columns.append(RouteColumn(
                customers_set=cset,
                ordered_route=tuple(route),
                cost=vrp.route_distance(route)
            ))
            used_sets.add(cset)

    # Add all singleton routes too (helps LP flexibility)
    for i in range(1, vrp.num_customers + 1):
        cset = frozenset([i])
        if cset not in used_sets:
            current_columns.append(RouteColumn(
                customers_set=cset,
                ordered_route=(i,),
                cost=vrp.route_distance([i])
            ))
            used_sets.add(cset)

    iteration = 0
    start = time.perf_counter()

    while True:
        iteration += 1
        lp_res = solve_rmp_lp(vrp, current_columns)

        duals = lp_res.eqlin.marginals
        dual_customer = duals[:-1]
        dual_vehicle = duals[-1]

        best_rc, best_col = pricing_by_reduced_cost(
            vrp=vrp,
            all_columns=all_columns,
            current_columns=current_columns,
            dual_customer=dual_customer,
            dual_vehicle=dual_vehicle
        )

        if verbose:
            print(f"[CG][Iter {iteration:02d}] LP obj = {lp_res.fun:.6f}")
            print(f"[CG][Iter {iteration:02d}] best reduced cost = {best_rc:.6f}")
            if best_col is not None:
                print(f"[CG][Iter {iteration:02d}] entering route = {list(best_col.ordered_route)}, cost = {best_col.cost:.6f}")

        if best_col is None or best_rc >= -1e-9:
            break

        current_columns.append(best_col)

    int_res = solve_master_integer(vrp, current_columns)

    chosen_columns = []
    for j, x in enumerate(int_res.x):
        if x > 0.5:
            chosen_columns.append(current_columns[j])

    chosen_routes = [list(col.ordered_route) for col in chosen_columns]
    total_cost = sum(col.cost for col in chosen_columns)
    runtime = time.perf_counter() - start

    valid, msg = vrp.validate_routes(chosen_routes)
    if not valid:
        raise RuntimeError(f"CG final routes invalid: {msg}")

    return {
        "routes": chosen_routes,
        "distance": total_cost,
        "runtime_sec": runtime,
        "iterations": iteration,
        "n_columns_final": len(current_columns),
    }


# ============================================================
# COMPARISON + PLOT
# ============================================================

def compare_instance(vrp, save_prefix=None, show_plot=True):
    # Clarke-Wright
    t0 = time.perf_counter()
    cw_routes = clarke_and_wright(vrp)
    cw_runtime = time.perf_counter() - t0
    cw_distance = vrp.total_distance(cw_routes)
    cw_valid, cw_msg = vrp.validate_routes(cw_routes)

    if not cw_valid:
        raise RuntimeError(f"Clarke-Wright invalid: {cw_msg}")

    # SPPRC-CG
    cg_result = spprc_column_generation(vrp, verbose=True)
    cg_routes = cg_result["routes"]
    cg_distance = cg_result["distance"]
    cg_runtime = cg_result["runtime_sec"]

    improvement_pct = 100.0 * (cw_distance - cg_distance) / cw_distance if cw_distance > 1e-12 else 0.0

    print("\n" + "=" * 80)
    print(f"INSTANCE {vrp.instance_id}")
    print("=" * 80)
    print(f"Clarke-Wright routes: {cw_routes}")
    print(f"Clarke-Wright distance: {cw_distance:.6f}")
    print(f"Clarke-Wright time: {cw_runtime:.6f} s")
    print()
    print(f"SPPRC-CG routes: {cg_routes}")
    print(f"SPPRC-CG distance: {cg_distance:.6f}")
    print(f"SPPRC-CG time: {cg_runtime:.6f} s")
    print(f"SPPRC-CG iterations: {cg_result['iterations']}")
    print(f"Columns in final master: {cg_result['n_columns_final']}")
    print(f"Improvement over Clarke-Wright: {improvement_pct:.4f}%")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    vrp.plot_routes(
        cw_routes,
        f"Instance {vrp.instance_id} - Clarke & Wright\n"
        f"dist={cw_distance:.3f}, time={cw_runtime:.6f}s",
        ax=axes[0]
    )

    vrp.plot_routes(
        cg_routes,
        f"Instance {vrp.instance_id} - SPPRC Column Generation\n"
        f"dist={cg_distance:.3f}, time={cg_runtime:.6f}s",
        ax=axes[1]
    )

    plt.tight_layout()

    if save_prefix:
        out = f"CWvsSPPRC/{save_prefix}_instance_{vrp.instance_id}_cw_vs_spprc.png"
        plt.savefig(out, dpi=220, bbox_inches="tight")
        print(f"\nSaved plot: {out}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "instance_id": vrp.instance_id,
        "cw_routes": cw_routes,
        "cw_distance": cw_distance,
        "cw_time_sec": cw_runtime,
        "spprc_routes": cg_routes,
        "spprc_distance": cg_distance,
        "spprc_time_sec": cg_runtime,
        "improvement_pct": improvement_pct,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Change this list however you want
    instance_files = [
        "instance_1.csv",
        "instance_2.csv",
        "instance_3.csv",
        "instance_4.csv",
        "synthetic_data_1.csv"
    ]

    all_results = []

    for filename in instance_files:
        print("\n" + "#" * 90)
        print(f"Running {filename}")
        print("#" * 90)

        vrp = load_instance(filename)
        result = compare_instance(vrp, save_prefix="vrp", show_plot=True)
        all_results.append(result)

    print("\n" + "=" * 90)
    print("FINAL COMPARISON")
    print("=" * 90)
    for r in all_results:
        print(
            f"Instance {r['instance_id']}: "
            f"CW={r['cw_distance']:.3f}, "
            f"SPPRC-CG={r['spprc_distance']:.3f}, "
            f"Improve={r['improvement_pct']:.3f}%"
        )
        with open(f"SPPRC/Instance_{r['instance_id']}_SPPRC_routes.txt", "w") as f:
            for i,route in enumerate(r["spprc_routes"]):
                route_str = " -> ".join(str(c) for c in route)
                f.write(f"Vehicle {i + 1}: 0 -> {route_str} -> 0\n")