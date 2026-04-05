import math
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def solve_cvrp_gurobi(customers, num_vehicles, capacity, depot=(0, 0), time_limit=None, mip_gap=None, verbose=True):
    """
    Solve a unit-demand CVRP with:
      - depot at given depot coordinates
      - each customer has unit demand
      - each customer visited exactly once
      - each vehicle may be used at most once
      - each vehicle can serve at most `capacity` customers
    """

    n = len(customers)

    if n > num_vehicles * capacity:
        raise ValueError(
            f"Infeasible: {n} customers > {num_vehicles} * {capacity} = {num_vehicles * capacity}."
        )

    coords = {0: depot}
    coords.update(customers)

    V = [0] + sorted(customers.keys())
    CUST = sorted(customers.keys())

    dist = {(i, j): euclidean(coords[i], coords[j]) for i in V for j in V if i != j}

    model = gp.Model("CVRP_BranchAndBound")

    if not verbose:
        model.setParam("OutputFlag", 0)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    if mip_gap is not None:
        model.setParam("MIPGap", mip_gap)

    # x[i,j,k] = 1 if vehicle k travels from i to j
    x = model.addVars(
        [(i, j, k) for i in V for j in V for k in range(num_vehicles) if i != j],
        vtype=GRB.BINARY,
        name="x"
    )

    # y[k] = 1 if vehicle k is used
    y = model.addVars(range(num_vehicles), vtype=GRB.BINARY, name="y")

    # u[i,k] = order of customer i on vehicle k's route (MTZ)
    u = model.addVars(
        [(i, k) for i in CUST for k in range(num_vehicles)],
        lb=1,
        ub=capacity,
        vtype=GRB.CONTINUOUS,
        name="u"
    )

    # Objective
    model.setObjective(
        gp.quicksum(
            dist[i, j] * x[i, j, k]
            for i in V for j in V for k in range(num_vehicles) if i != j
        ),
        GRB.MINIMIZE
    )

    # Each customer visited exactly once
    model.addConstrs(
        (
            gp.quicksum(x[i, j, k] for k in range(num_vehicles) for i in V if i != j) == 1
            for j in CUST
        ),
        name="VisitExactlyOnce"
    )

    # Flow conservation at each customer for each vehicle
    model.addConstrs(
        (
            gp.quicksum(x[i, m, k] for i in V if i != m)
            ==
            gp.quicksum(x[m, j, k] for j in V if j != m)
            for m in CUST for k in range(num_vehicles)
        ),
        name="FlowConservation"
    )

    # Vehicle starts from depot iff used
    model.addConstrs(
        (
            gp.quicksum(x[0, j, k] for j in CUST) == y[k]
            for k in range(num_vehicles)
        ),
        name="DepotDeparture"
    )

    # Vehicle returns to depot iff used
    model.addConstrs(
        (
            gp.quicksum(x[i, 0, k] for i in CUST) == y[k]
            for k in range(num_vehicles)
        ),
        name="DepotReturn"
    )

    # Capacity: each vehicle serves at most `capacity` customers
    model.addConstrs(
        (
            gp.quicksum(x[i, j, k] for i in V for j in CUST if i != j) <= capacity
            for k in range(num_vehicles)
        ),
        name="Capacity"
    )

    # MTZ subtour elimination for each vehicle
    for k in range(num_vehicles):
        for i in CUST:
            for j in CUST:
                if i != j:
                    model.addConstr(
                        u[i, k] - u[j, k] + capacity * x[i, j, k] <= capacity - 1,
                        name=f"MTZ_{i}_{j}_{k}"
                    )

    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"Gurobi ended with status code {model.Status}")

    if model.SolCount == 0:
        raise RuntimeError("No feasible solution found.")

    routes = extract_routes_from_x(x, CUST, num_vehicles)
    return routes, model.ObjVal, model


def extract_routes_from_x(x, customers, num_vehicles):
    routes = []

    for k in range(num_vehicles):
        starts = [j for j in customers if (0, j, k) in x and x[0, j, k].X > 0.5]
        if not starts:
            continue

        route = [0]
        current = 0
        visited_guard = set()

        while True:
            next_nodes = [
                j for j in [0] + customers
                if j != current and (current, j, k) in x and x[current, j, k].X > 0.5
            ]

            if not next_nodes:
                break

            nxt = next_nodes[0]
            route.append(nxt)

            if nxt == 0:
                break

            if nxt in visited_guard:
                break

            visited_guard.add(nxt)
            current = nxt

        routes.append(route)

    return routes


def write_solution_file(filename, routes):
    with open(filename, "w") as f:
        for idx, route in enumerate(routes, start=1):
            f.write(f"Vehicle {idx}: " + " -> ".join(map(str, route)) + "\n")


def save_costs_chart(results, filename="instance_costs.png"):
    """
    Save one chart showing objective values for all solved instances.

    results: list of dicts like
      [{"name": "Instance 1", "cost": 12.34}, ...]
    """
    solved = [r for r in results if r["cost"] is not None]

    if not solved:
        print("No solved instances to plot.")
        return

    names = [r["name"] for r in solved]
    costs = [r["cost"] for r in solved]

    plt.figure(figsize=(9, 6))
    plt.bar(names, costs)
    plt.xlabel("Instance")
    plt.ylabel("Objective Value")
    plt.title("CVRP Objective Values for Hardcoded Instances")
    plt.grid(True, axis="y")

    for i, cost in enumerate(costs):
        plt.text(i, cost, f"{cost:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Cost chart saved to: {filename}")


def solve_instance(name, customers, num_vehicles, capacity, time_limit=60, mip_gap=0.0):
    print(f"\n{'=' * 70}")
    print(f"{name}: v={num_vehicles}, C={capacity}")
    print(f"{'=' * 70}")

    try:
        routes, cost, model = solve_cvrp_gurobi(
            customers=customers,
            num_vehicles=num_vehicles,
            capacity=capacity,
            time_limit=time_limit,
            mip_gap=mip_gap,
            verbose=True
        )

        print(f"Objective value: {cost:.4f}")
        for i, route in enumerate(routes, start=1):
            print(f"Vehicle {i}: {' -> '.join(map(str, route))}")

        out_file = f"{name.replace(' ', '_')}.txt"
        write_solution_file(out_file, routes)
        print(f"Solution written to: {out_file}")

        return {
            "name": name,
            "cost": cost,
            "routes": routes,
            "status": "Solved"
        }

    except Exception as e:
        print(f"{name} failed: {e}")
        return {
            "name": name,
            "cost": None,
            "routes": None,
            "status": f"Failed: {e}"
        }


if __name__ == "__main__":


    instance_n4_customers = {
            1: (-2, 2),
            2: (-5, 8),
            3: (2, 3),
            4: (5, 7),
            5: (2, 4),
            6: (2, -3),
            7: (-4, 1),
            8: (0, 6),
            9: (3, -2),
            10: (-1, 5),
            11: (6, 1),
            12: (-3, 4),
            13: (4, 3),
            14: (-6, 2),
            15: (1, 7),
            16: (5, -1),
            17: (-2, -4),
            18: (3, 6),
            19: (-5, 5),
            20: (0, -2),
        }
    # # Instance 1: v = 2, C = 5
    # instance_1_customers = {
    #     1: (-2, 2),
    #     2: (-5, 8),
    #     3: (2, 3),
    # }

    # # Instance 2: v = 2, C = 2
    # instance_2_customers = {
    #     1: (-2, 2),
    #     2: (-5, 8),
    #     3: (2, 3),
    # }

    # # Instance 3: v = 3, C = 2
    # instance_3_customers = {
    #     1: (-2, 2),
    #     2: (-5, 8),
    #     3: (2, 3),
    #     4: (5, 7),
    #     5: (2, 4),
    #     6: (2, -3),
    # }

    # # Instance 4: v = 4, C = 3
    # instance_4_customers = {
    #     1: (-2, 2),
    #     2: (-5, 8),
    #     3: (6, 3),
    #     4: (4, 4),
    #     5: (3, 2),
    #     6: (0, 2),
    #     7: (-2, 3),
    #     8: (-4, 3),
    #     9: (2, 3),
    #     10: (2, 7),
    #     11: (-2, 5),
    #     12: (-1, 4),
    # }

    results = []

    results.append(
        solve_instance(
            name="instance_1",
            customers=instance_n4_customers,
            num_vehicles=5,
            capacity=4,
            time_limit=60,
            mip_gap=0.0
        )
    )

    # results.append(
    #     solve_instance(
    #         name="Instance 2",
    #         customers=instance_2_customers,
    #         num_vehicles=2,
    #         capacity=2,
    #         time_limit=60,
    #         mip_gap=0.0
    #     )
    # )

    # results.append(
    #     solve_instance(
    #         name="Instance 3",
    #         customers=instance_3_customers,
    #         num_vehicles=3,
    #         capacity=2,
    #         time_limit=60,
    #         mip_gap=0.0
    #     )
    # )

    # results.append(
    #     solve_instance(
    #         name="Instance 4",
    #         customers=instance_4_customers,
    #         num_vehicles=4,
    #         capacity=3,
    #         time_limit=60,
    #         mip_gap=0.0
    #     )
    # )

    print("\nSummary:")
    for r in results:
        print(f"{r['name']}: {r['status']}, cost={r['cost']}")

    save_costs_chart(results, filename="all_instance_costs.png")