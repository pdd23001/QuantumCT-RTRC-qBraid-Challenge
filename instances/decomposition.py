"""
decomposition.py
================
Seed portfolio: multiple classical decomposition strategies that each produce
a valid (or near-valid) initial route set for a CVRP instance.

Strategies implemented
----------------------
1. sweep               -- angular sweep from depot, standard direction
2. reverse_sweep       -- angular sweep, reverse direction
3. shifted_sweep       -- multiple angular offsets for the standard sweep
4. clarke_wright       -- savings-based heuristic
5. capacitated_kmeans  -- centroid-based clustering (no sklearn required)

Each strategy returns a CandidateSolution named-tuple (or dict).

Output contract
---------------
Every returned solution has these keys:
    routes       : list[list[int]]   customer IDs, no depot at ends
    total_dist   : float
    vehicles     : int               number of non-empty routes
    valid        : bool
    method       : str               short label
    meta         : dict              extra info (offset, direction, etc.)
"""

import math
import random
from common import (
    build_distance_matrix,
    route_distance,
    total_solution_distance,
    validate_solution,
    customer_angle,
    route_load,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_result(routes, D, demands, capacity, num_customers, num_vehicles, method, meta=None):
    """Pack a route list into a standard result dict."""
    routes = [r for r in routes if r]  # drop empty routes
    dist = total_solution_distance(routes, D)
    valid = validate_solution(routes, demands, capacity, num_customers, num_vehicles)
    return {
        "routes": routes,
        "total_dist": round(dist, 6),
        "vehicles": len(routes),
        "valid": valid,
        "method": method,
        "meta": meta or {},
    }


def _split_into_routes(ordered_customers, demands, capacity, max_routes):
    """
    Pack customers (in given order) greedily into routes respecting capacity.
    Splits into at most max_routes routes; overflows go into extra routes.

    Returns list[list[int]]
    """
    routes = []
    current_route = []
    current_load = 0

    for c in ordered_customers:
        d = demands[c]
        if current_load + d > capacity and current_route:
            routes.append(current_route)
            current_route = []
            current_load = 0
        current_route.append(c)
        current_load += d

    if current_route:
        routes.append(current_route)

    return routes


# ---------------------------------------------------------------------------
# 1. Sweep decomposition
# ---------------------------------------------------------------------------

def sweep_decomposition(nodes, demands, capacity, num_vehicles, offset=0.0, reverse=False, method_name=None):
    """
    Angular sweep from depot -- sort customers by polar angle, then pack
    greedily into routes.

    Parameters
    ----------
    nodes       : list[(float, float)]  -- index 0 = depot
    demands     : list[int]
    capacity    : int
    num_vehicles: int
    offset      : float                 -- angle offset in radians
    reverse     : bool
    method_name : str or None

    Returns
    -------
    dict (standard result format)
    """
    if method_name is None:
        method_name = "reverse_sweep" if reverse else "sweep"

    depot = nodes[0]
    customers = list(range(1, len(nodes)))

    # Sort by polar angle (with offset, wrapped to [-pi, pi))
    def sort_key(c):
        angle = customer_angle(depot, nodes[c]) + offset
        # wrap to [-pi, pi)
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle <= -math.pi:
            angle += 2 * math.pi
        return angle

    customers.sort(key=sort_key, reverse=reverse)

    D = build_distance_matrix(nodes)
    num_customers = len(nodes) - 1
    routes = _split_into_routes(customers, demands, capacity, num_vehicles)

    return _make_result(routes, D, demands, capacity, num_customers, num_vehicles, method_name,
                        meta={"offset": offset, "reverse": reverse})


def reverse_sweep(nodes, demands, capacity, num_vehicles):
    """Reverse angular sweep."""
    return sweep_decomposition(nodes, demands, capacity, num_vehicles,
                               offset=0.0, reverse=True, method_name="reverse_sweep")


def shifted_sweep_portfolio(nodes, demands, capacity, num_vehicles, n_offsets=4):
    """
    Try n_offsets evenly-spaced angular offsets and return all results.

    Returns
    -------
    list[dict]
    """
    results = []
    for k in range(n_offsets):
        offset = (2 * math.pi * k) / n_offsets
        r = sweep_decomposition(nodes, demands, capacity, num_vehicles,
                                offset=offset, method_name=f"sweep_off{k}")
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# 2. Clarke-Wright savings
# ---------------------------------------------------------------------------

def clarke_wright(nodes, demands, capacity, num_vehicles):
    """
    Clarke-Wright savings heuristic.

    Returns
    -------
    dict (standard result format)
    """
    D = build_distance_matrix(nodes)
    num_customers = len(nodes) - 1
    customers = list(range(1, num_customers + 1))

    # Compute savings s(i,j) = d(0,i) + d(0,j) - d(i,j)
    savings = []
    for i in customers:
        for j in customers:
            if i >= j:
                continue
            s = D[0][i] + D[0][j] - D[i][j]
            savings.append((s, i, j))
    savings.sort(reverse=True)

    # Start: each customer on its own route
    routes = [[c] for c in customers]
    customer_to_route = {c: routes[idx] for idx, c in enumerate(customers)}

    def remap(route):
        for c in route:
            customer_to_route[c] = route

    for _, i, j in savings:
        ri = customer_to_route.get(i)
        rj = customer_to_route.get(j)

        if ri is None or rj is None or ri is rj:
            continue
        if route_load(ri, demands) + route_load(rj, demands) > capacity:
            continue
        if len(routes) <= num_vehicles and len(routes) - 1 == 0:
            # Only one route remaining, skip
            continue
        # Edge endpoints only
        if i not in (ri[0], ri[-1]) or j not in (rj[0], rj[-1]):
            continue

        left = ri[:] if ri[-1] == i else ri[::-1]
        right = rj[:] if rj[0] == j else rj[::-1]
        merged = left + right

        routes.remove(ri)
        routes.remove(rj)
        routes.append(merged)
        remap(merged)

    return _make_result(routes, D, demands, capacity, num_customers, num_vehicles,
                        "clarke_wright")


# ---------------------------------------------------------------------------
# 3. Capacitated k-means (no sklearn)
# ---------------------------------------------------------------------------

def _kmeans_assign(customers, centroids, nodes):
    """
    Assign each customer to closest centroid.
    Returns list[list[int]] of length k.
    """
    k = len(centroids)
    clusters = [[] for _ in range(k)]
    for c in customers:
        cx, cy = nodes[c]
        best = min(range(k), key=lambda ci: (cx - centroids[ci][0]) ** 2 + (cy - centroids[ci][1]) ** 2)
        clusters[best].append(c)
    return clusters


def _kmeans_centroids(clusters, nodes):
    """Recompute centroids from cluster assignments."""
    centroids = []
    for cluster in clusters:
        if not cluster:
            centroids.append((0.0, 0.0))
        else:
            xs = [nodes[c][0] for c in cluster]
            ys = [nodes[c][1] for c in cluster]
            centroids.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return centroids


def _balance_clusters(clusters, demands, capacity):
    """
    Post-k-means balancing: move overloaded cluster members to underloaded ones.
    Simple greedy reallocate -- not globally optimal, but practical.
    """
    changed = True
    while changed:
        changed = False
        for i, ci in enumerate(clusters):
            load_i = sum(demands[c] for c in ci)
            if load_i <= capacity:
                continue
            # Try to move customers to a cluster that can absorb them
            for j, cj in enumerate(clusters):
                if i == j:
                    continue
                load_j = sum(demands[c] for c in cj)
                # Try each customer in ci
                for c in list(ci):
                    d = demands[c]
                    if load_j + d <= capacity and load_i - d >= 0:
                        ci.remove(c)
                        cj.append(c)
                        load_i -= d
                        load_j += d
                        changed = True
                        break
                if load_i <= capacity:
                    break
    return clusters


def capacitated_kmeans(nodes, demands, capacity, num_vehicles, seed=42, max_iter=50):
    """
    Capacitated k-means: cluster customers into num_vehicles clusters,
    then build one route per cluster with a nearest-neighbor tour.

    Parameters
    ----------
    nodes       : list[(float, float)]
    demands     : list[int]
    capacity    : int
    num_vehicles: int
    seed        : int
    max_iter    : int

    Returns
    -------
    dict (standard result format)
    """
    D = build_distance_matrix(nodes)
    num_customers = len(nodes) - 1
    customers = list(range(1, num_customers + 1))
    k = min(num_vehicles, num_customers)

    rng = random.Random(seed)
    # Init centroids from a random sample of customer positions
    init_ids = rng.sample(customers, k)
    centroids = [(nodes[c][0], nodes[c][1]) for c in init_ids]

    clusters = [[] for _ in range(k)]
    for _ in range(max_iter):
        new_clusters = _kmeans_assign(customers, centroids, nodes)
        new_centroids = _kmeans_centroids(new_clusters, nodes)
        if new_centroids == centroids:
            clusters = new_clusters
            break
        centroids = new_centroids
        clusters = new_clusters

    # Balance overloaded clusters
    clusters = _balance_clusters(clusters, demands, capacity)

    # Build nearest-neighbor route within each cluster
    routes = []
    for cluster in clusters:
        if not cluster:
            continue
        if len(cluster) == 1:
            routes.append(cluster[:])
            continue
        # Nearest-neighbor from depot
        route = []
        remaining = set(cluster)
        current = 0
        while remaining:
            nxt = min(remaining, key=lambda c: D[current][c])
            route.append(nxt)
            remaining.remove(nxt)
            current = nxt
        routes.append(route)

    return _make_result(routes, D, demands, capacity, num_customers, num_vehicles,
                        "capacitated_kmeans", meta={"seed": seed})


# ---------------------------------------------------------------------------
# 4. Unified seed portfolio
# ---------------------------------------------------------------------------

def generate_seed_portfolio(nodes, demands, capacity, num_vehicles,
                            sweep_offsets=4, kmeans_seeds=None):
    """
    Generate all candidate decompositions and return them as a list.

    Parameters
    ----------
    nodes        : list[(float, float)]
    demands      : list[int]
    capacity     : int
    num_vehicles : int
    sweep_offsets: int    -- number of shifted sweep angles to try
    kmeans_seeds : list[int] or None

    Returns
    -------
    list[dict]  -- each is a standard result dict (may be invalid)
    """
    if kmeans_seeds is None:
        kmeans_seeds = [42, 7, 123]

    candidates = []

    # Sweep variants
    candidates.append(sweep_decomposition(nodes, demands, capacity, num_vehicles))
    candidates.append(reverse_sweep(nodes, demands, capacity, num_vehicles))
    candidates.extend(shifted_sweep_portfolio(nodes, demands, capacity, num_vehicles, sweep_offsets))

    # Clarke-Wright
    candidates.append(clarke_wright(nodes, demands, capacity, num_vehicles))

    # k-means variants
    for s in kmeans_seeds:
        candidates.append(capacitated_kmeans(nodes, demands, capacity, num_vehicles, seed=s))

    return candidates


def best_valid_candidate(candidates):
    """
    Return the valid candidate with the lowest total distance.
    Falls back to the overall best (valid or not) if none are valid.
    """
    valid = [c for c in candidates if c["valid"]]
    pool = valid if valid else candidates
    return min(pool, key=lambda c: c["total_dist"])
