"""
common.py
=========
Core shared utilities: geometry helpers, distance matrix computation,
route cost / load / feasibility checks, and instance parsing.

All customer IDs are 1-indexed integers.
Depot is always index 0 in the distance matrix (and node list).
Routes are stored as lists of customer IDs (without depot at either end
in the internal representation).
"""

import math
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def euclidean(a, b):
    """Euclidean distance between two (x, y) points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def build_distance_matrix(nodes):
    """
    Build a symmetric distance matrix from a list of (x, y) tuples.

    Parameters
    ----------
    nodes : list of (float, float)
        Index 0 must be the depot; indices 1..n are customers.

    Returns
    -------
    D : list[list[float]]
        D[i][j] == D[j][i] == Euclidean distance between nodes[i] and nodes[j].
    """
    n = len(nodes)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean(nodes[i], nodes[j])
            D[i][j] = d
            D[j][i] = d
    return D


# ---------------------------------------------------------------------------
# Route cost / load / angle helpers
# ---------------------------------------------------------------------------

def route_distance(route, D):
    """
    Total travel distance for a route (depot -> customers -> depot).

    Parameters
    ----------
    route : list[int]
        Customer IDs (1-indexed), no depot at ends.
    D : list[list[float]]
        Full distance matrix (depot = index 0).

    Returns
    -------
    float
    """
    if not route:
        return 0.0
    total = D[0][route[0]]
    for k in range(len(route) - 1):
        total += D[route[k]][route[k + 1]]
    total += D[route[-1]][0]
    return total


def route_load(route, demands):
    """
    Total demand on a route.

    Parameters
    ----------
    route : list[int]
        Customer IDs (1-indexed).
    demands : list[int]
        demands[i] is the demand of customer i (index 0 = depot, demand 0).

    Returns
    -------
    int
    """
    return sum(demands[c] for c in route)


def total_solution_distance(routes, D):
    """Sum of route_distance over all routes."""
    return sum(route_distance(r, D) for r in routes)


def customer_angle(depot, node):
    """
    Polar angle (radians) from depot to a customer node.

    Parameters
    ----------
    depot : (float, float)
    node  : (float, float)

    Returns
    -------
    float in [-pi, pi]
    """
    return math.atan2(node[1] - depot[1], node[0] - depot[0])


def route_centroid(route, nodes):
    """
    Centroid of customer positions in a route.

    Parameters
    ----------
    route : list[int]  -- customer IDs (1-indexed)
    nodes : list[(float, float)]  -- index 0 = depot

    Returns
    -------
    (float, float)
    """
    if not route:
        return nodes[0]
    xs = [nodes[c][0] for c in route]
    ys = [nodes[c][1] for c in route]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


# ---------------------------------------------------------------------------
# Feasibility / validation
# ---------------------------------------------------------------------------

def is_route_feasible(route, demands, capacity):
    """
    Check that a route does not exceed capacity.

    Parameters
    ----------
    route    : list[int]
    demands  : list[int]
    capacity : int

    Returns
    -------
    bool
    """
    return route_load(route, demands) <= capacity


def validate_solution(routes, demands, capacity, num_customers, num_vehicles):
    """
    Full solution validity check.

    Rules
    -----
    - Every customer 1..num_customers appears exactly once across all routes.
    - No route exceeds capacity.
    - Number of routes <= num_vehicles.

    Returns
    -------
    bool
    """
    if len(routes) > num_vehicles:
        return False
    seen = []
    for route in routes:
        if route_load(route, demands) > capacity:
            return False
        seen.extend(route)
    return sorted(seen) == list(range(1, num_customers + 1))


# ---------------------------------------------------------------------------
# Instance parsing
# ---------------------------------------------------------------------------

def parse_instance_dict(instance):
    """
    Parse one instance dict (from the JSON benchmark file) into a canonical form.

    Parameters
    ----------
    instance : dict
        Keys: instance_id, Nv, C, customers (list of dicts with customer_id, x, y, demand).

    Returns
    -------
    dict with keys:
        instance_id : str
        num_vehicles : int
        capacity : int
        nodes : list[(float, float)]   -- index 0 = depot
        demands : list[int]            -- index 0 = 0 (depot), 1..n = customer demands
        num_customers : int
    """
    num_vehicles = int(instance["Nv"])
    capacity = int(instance["C"])
    instance_id = instance["instance_id"]

    depot = None
    raw_customers = []

    for c in instance["customers"]:
        cid = int(c["customer_id"])
        x = float(c["x"])
        y = float(c["y"])
        d = int(c["demand"])
        if cid == 0:
            depot = (x, y)
        else:
            raw_customers.append((cid, x, y, d))

    if depot is None:
        raise ValueError(f"Instance {instance_id} has no depot (customer_id=0).")

    # Sort by customer ID to ensure consistent ordering
    raw_customers.sort(key=lambda t: t[0])

    nodes = [depot] + [(c[1], c[2]) for c in raw_customers]
    demands = [0] + [c[3] for c in raw_customers]
    num_customers = len(raw_customers)

    return {
        "instance_id": instance_id,
        "num_vehicles": num_vehicles,
        "capacity": capacity,
        "nodes": nodes,
        "demands": demands,
        "num_customers": num_customers,
    }


def load_instances_json(json_path):
    """
    Load all instances from a benchmark JSON file.

    Parameters
    ----------
    json_path : str or Path

    Returns
    -------
    list[dict]  -- each dict is the output of parse_instance_dict
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [parse_instance_dict(inst) for inst in raw]


def challenge_instances():
    """
    Return the four small challenge instances in canonical form.
    These are defined inline -- no file dependency.

    Returns
    -------
    list[dict]
    """
    raw = [
        {
            "instance_id": "Instance1",
            "Nv": 2,
            "C": 5,
            "customers": [
                {"customer_id": 0, "x": 0, "y": 0, "demand": 0},
                {"customer_id": 1, "x": -2, "y": 2, "demand": 1},
                {"customer_id": 2, "x": -5, "y": 8, "demand": 1},
                {"customer_id": 3, "x":  2, "y": 3, "demand": 1},
            ],
        },
        {
            "instance_id": "Instance2",
            "Nv": 2,
            "C": 2,
            "customers": [
                {"customer_id": 0, "x": 0, "y": 0, "demand": 0},
                {"customer_id": 1, "x": -2, "y": 2, "demand": 1},
                {"customer_id": 2, "x": -5, "y": 8, "demand": 1},
                {"customer_id": 3, "x":  2, "y": 3, "demand": 1},
            ],
        },
        {
            "instance_id": "Instance3",
            "Nv": 3,
            "C": 2,
            "customers": [
                {"customer_id": 0, "x": 0, "y":  0, "demand": 0},
                {"customer_id": 1, "x": -2, "y":  2, "demand": 1},
                {"customer_id": 2, "x": -5, "y":  8, "demand": 1},
                {"customer_id": 3, "x":  2, "y":  3, "demand": 1},
                {"customer_id": 4, "x":  5, "y":  7, "demand": 1},
                {"customer_id": 5, "x":  2, "y":  4, "demand": 1},
                {"customer_id": 6, "x":  2, "y": -3, "demand": 1},
            ],
        },
        {
            "instance_id": "Instance4",
            "Nv": 4,
            "C": 3,
            "customers": [
                {"customer_id":  0, "x":  0, "y":  0, "demand": 0},
                {"customer_id":  1, "x": -2, "y":  2, "demand": 1},
                {"customer_id":  2, "x": -5, "y":  8, "demand": 1},
                {"customer_id":  3, "x":  6, "y":  3, "demand": 1},
                {"customer_id":  4, "x":  4, "y":  4, "demand": 1},
                {"customer_id":  5, "x":  3, "y":  2, "demand": 1},
                {"customer_id":  6, "x":  0, "y":  2, "demand": 1},
                {"customer_id":  7, "x": -2, "y":  3, "demand": 1},
                {"customer_id":  8, "x": -4, "y":  3, "demand": 1},
                {"customer_id":  9, "x":  2, "y":  3, "demand": 1},
                {"customer_id": 10, "x":  2, "y":  7, "demand": 1},
                {"customer_id": 11, "x": -2, "y":  5, "demand": 1},
                {"customer_id": 12, "x": -1, "y":  4, "demand": 1},
            ],
        },
    ]
    return [parse_instance_dict(inst) for inst in raw]


# ---------------------------------------------------------------------------
# Solution formatting helpers
# ---------------------------------------------------------------------------

def format_routes_text(routes, instance_id=""):
    """
    Format routes in the required hackathon submission format.

    Example output::

        r1: 0, 2, 3, 0
        r2: 0, 1, 0

    Parameters
    ----------
    routes : list[list[int]]  -- customer IDs, no depot
    instance_id : str

    Returns
    -------
    str
    """
    lines = []
    if instance_id:
        lines.append(f"# {instance_id}")
    for i, route in enumerate(routes, start=1):
        seq = [0] + route + [0]
        lines.append("r{}: {}".format(i, ", ".join(map(str, seq))))
    return "\n".join(lines)


def solution_summary(routes, D, demands, capacity, num_customers, num_vehicles, method=""):
    """
    Return a summary dict for a solved instance.

    Returns
    -------
    dict
    """
    dist = total_solution_distance(routes, D)
    valid = validate_solution(routes, demands, capacity, num_customers, num_vehicles)
    return {
        "method": method,
        "total_distance": round(dist, 6),
        "vehicles_used": len(routes),
        "vehicles_allowed": num_vehicles,
        "valid": valid,
        "routes": [
            {
                "vehicle": i + 1,
                "route": [0] + r + [0],
                "load": route_load(r, demands),
                "distance": round(route_distance(r, D), 6),
            }
            for i, r in enumerate(routes)
        ],
    }
