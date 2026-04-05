import math
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import networkx as nx


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def solve_cvrp_gurobi(customers, num_vehicles, capacity, depot=(0, 0),
                       time_limit=None, mip_gap=None, verbose=True):
    """
    Solve a unit-demand CVRP using a two-index formulation with lazy
    subtour/capacity elimination via callbacks.

    Two-index formulation (no vehicle index):
      - x[i,j] = 1 if *some* vehicle traverses arc i→j
      - Each customer has in-degree = out-degree = 1
      - Depot has out-degree = in-degree = number of vehicles used
      - Subtour + capacity constraints added lazily via callback

    This gives MUCH tighter LP relaxations than MTZ (typically <10% root gap
    vs 50-70% with MTZ), and the model has O(n²) variables instead of O(n²K).
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

    dist = {(i, j): euclidean(coords[i], coords[j])
            for i in V for j in V if i != j}

    model = gp.Model("CVRP_LazySubtour")

    if not verbose:
        model.setParam("OutputFlag", 0)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    if mip_gap is not None:
        model.setParam("MIPGap", mip_gap)

    # Allow lazy constraints
    model.setParam("LazyConstraints", 1)

    # --- Variables ---
    # x[i,j] = 1 if some vehicle travels arc i→j
    x = model.addVars(
        [(i, j) for i in V for j in V if i != j],
        vtype=GRB.BINARY, name="x"
    )

    # --- Objective ---
    model.setObjective(
        gp.quicksum(dist[i, j] * x[i, j] for i in V for j in V if i != j),
        GRB.MINIMIZE
    )

    # --- Constraints ---

    # Each customer has exactly one incoming arc
    model.addConstrs(
        (gp.quicksum(x[i, j] for i in V if i != j) == 1 for j in CUST),
        name="InDegree"
    )

    # Each customer has exactly one outgoing arc
    model.addConstrs(
        (gp.quicksum(x[j, i] for i in V if i != j) == 1 for j in CUST),
        name="OutDegree"
    )

    # Depot: number of vehicles leaving = number returning
    # Use exactly num_vehicles (can relax to <= if desired)
    min_vehicles = math.ceil(n / capacity)
    model.addConstr(
        gp.quicksum(x[0, j] for j in CUST) >= min_vehicles,
        name="MinVehicles"
    )
    model.addConstr(
        gp.quicksum(x[0, j] for j in CUST) <= num_vehicles,
        name="MaxVehicles"
    )
    # Symmetry: out-degree of depot = in-degree of depot
    model.addConstr(
        gp.quicksum(x[0, j] for j in CUST)
        == gp.quicksum(x[j, 0] for j in CUST),
        name="DepotBalance"
    )

    # --- Lazy callback for subtour + capacity elimination ---
    def subtour_callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return

        xval = model.cbGetSolution(x)

        # Build directed graph from solution
        G = nx.DiGraph()
        for (i, j) in x:
            if xval[i, j] > 0.5:
                G.add_edge(i, j)

        # Find connected components among customers only
        # (remove depot, find components, each must connect back to depot)
        G_cust = G.copy()
        if 0 in G_cust:
            G_cust.remove_node(0)

        components = list(nx.weakly_connected_components(G_cust))

        for comp in components:
            S = list(comp)

            # Check if this component connects to depot in original graph
            connects_to_depot = any(
                G.has_edge(0, j) or G.has_edge(j, 0) for j in S
            )

            if not connects_to_depot:
                # Pure subtour not involving depot — must be eliminated
                # SEC: sum of arcs within S <= |S| - 1
                model.cbLazy(
                    gp.quicksum(x[i, j] for i in S for j in S if i != j)
                    <= len(S) - 1
                )
            else:
                # Component connects to depot — check capacity
                if len(S) > capacity:
                    # Rounded capacity cut:
                    # At least ceil(|S|/C) vehicles must enter S
                    # i.e., sum of arcs from V\S to S >= ceil(|S|/C)
                    outside = [v for v in V if v not in S]
                    rhs = math.ceil(len(S) / capacity)
                    model.cbLazy(
                        gp.quicksum(
                            x[i, j] for i in outside for j in S if i != j
                        ) >= rhs
                    )

    model.optimize(subtour_callback)

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"Gurobi ended with status code {model.Status}")

    if model.SolCount == 0:
        raise RuntimeError("No feasible solution found.")

    routes = extract_routes(x, CUST)
    return routes, model.ObjVal, model


def extract_routes(x, customers):
    """Extract vehicle routes from the two-index solution."""
    # Build successor map
    succ = {}
    for (i, j) in x:
        if x[i, j].X > 0.5:
            succ[i] = j

    routes = []
    # Each route starts with an arc 0→j
    starts = [j for j in customers if (0, j) in x and x[0, j].X > 0.5]

    for start in starts:
        route = [0, start]
        current = start
        guard = 0
        while current != 0 and guard < len(customers) + 2:
            nxt = succ.get(current)
            if nxt is None:
                break
            route.append(nxt)
            current = nxt
            guard += 1
        routes.append(route)

    return routes


def write_solution_file(filename, routes):
    with open(filename, "w") as f:
        for idx, route in enumerate(routes, start=1):
            f.write(f"Vehicle {idx}: " + " -> ".join(map(str, route)) + "\n")


def save_costs_chart(results, filename="instance_costs.png"):
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


def solve_instance(name, customers, num_vehicles, capacity,
                   time_limit=60, mip_gap=0.0):
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

    instance_1_customers = {
        1: (-3, 4),
        2: (-6, 7),
        3: (2, 3),
        4: (5, 8),
        5: (3, 5),
        6: (1, -4),
        7: (-5, 1),
        8: (0, 7),
        9: (4, -2),
        10: (-1, 6),
        11: (6, 2),
        12: (-4, 3),
        13: (7, 5),
        14: (-7, 2),
        15: (1, 9),
        16: (5, -1),
        17: (-2, -5),
        18: (3, 7),
        19: (-6, 5),
        20: (0, -3),
        21: (8, 1),
        22: (-2, 8),
        23: (4, 0),
        24: (-5, -2),
        25: (2, 6),
    }

    results = []

    results.append(
        solve_instance(
            name="instance_1",
            customers=instance_1_customers,
            num_vehicles=5,
            capacity=5,
            time_limit=None,
            mip_gap=0.0
        )
    )

    print("\nSummary:")
    for r in results:
        print(f"{r['name']}: {r['status']}, cost={r['cost']}")

    save_costs_chart(results, filename="all_instance_costs.png")