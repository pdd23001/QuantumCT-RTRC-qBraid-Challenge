"""
repair.py
=========
Inter-route local search: relocate and swap moves.

Both moves respect capacity and only accept improving moves (strictly better
total distance).  After each accepted move, 2-opt is re-applied to the two
touched routes only.

Design notes
------------
- Works on general route sets, not just sweep-style sectors.
- Route pairs are ordered by centroid proximity to try the most promising
  pairs first.
- This module has no quantum dependency.
"""

import math
from common import (
    route_distance,
    route_load,
    total_solution_distance,
    route_centroid,
)
from local_search import two_opt


# ---------------------------------------------------------------------------
# Proximity ordering
# ---------------------------------------------------------------------------

def _route_pair_order(routes, nodes):
    """
    Return all distinct route index pairs (i, j) sorted by centroid distance
    (closest centroids first -- most likely to benefit from inter-route moves).

    Parameters
    ----------
    routes : list[list[int]]
    nodes  : list[(float, float)]

    Returns
    -------
    list[(int, int)]
    """
    centroids = [route_centroid(r, nodes) for r in routes]
    pairs = []
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            cx = centroids[i][0] - centroids[j][0]
            cy = centroids[i][1] - centroids[j][1]
            dist = math.sqrt(cx * cx + cy * cy)
            pairs.append((dist, i, j))
    pairs.sort()
    return [(i, j) for (_, i, j) in pairs]


# ---------------------------------------------------------------------------
# Relocate move: move one customer from route_a to route_b
# ---------------------------------------------------------------------------

def _try_relocate(routes, D, demands, capacity, nodes, max_pair_fraction=1.0):
    """
    Try all one-customer relocations.  Accept first improving move (first-
    improvement strategy).

    Parameters
    ----------
    routes             : list[list[int]]  -- modified in place on improvement
    D                  : list[list[float]]
    demands            : list[int]
    capacity           : int
    nodes              : list[(float, float)]
    max_pair_fraction  : float in (0,1]  -- fraction of route pairs to check

    Returns
    -------
    bool  -- True if an improvement was found
    """
    pairs = _route_pair_order(routes, nodes)
    n_check = max(1, int(len(pairs) * max_pair_fraction))
    pairs = pairs[:n_check]

    for i, j in pairs:
        ri = routes[i]
        rj = routes[j]

        # Try moving each customer from ri -> rj
        for direction in [(i, j), (j, i)]:
            src_idx, dst_idx = direction
            src = routes[src_idx]
            dst = routes[dst_idx]

            if not src:
                continue

            d_src_before = route_distance(src, D)
            d_dst_before = route_distance(dst, D)
            base = d_src_before + d_dst_before

            for pos, cust in enumerate(src):
                if route_load(dst, demands) + demands[cust] > capacity:
                    continue

                # Remove from src
                new_src = src[:pos] + src[pos + 1:]

                # Find best insertion point in dst
                best_gain = 1e-10  # must be strictly positive
                best_insert = -1
                d_new_src = route_distance(new_src, D) if new_src else 0.0

                for ins in range(len(dst) + 1):
                    new_dst = dst[:ins] + [cust] + dst[ins:]
                    d_new_dst = route_distance(new_dst, D)
                    gain = base - (d_new_src + d_new_dst)
                    if gain > best_gain:
                        best_gain = gain
                        best_insert = ins

                if best_insert >= 0:
                    # Accept move
                    new_src_opt = two_opt(new_src, D) if new_src else []
                    new_dst_raw = dst[:best_insert] + [cust] + dst[best_insert:]
                    new_dst_opt = two_opt(new_dst_raw, D)

                    routes[src_idx] = new_src_opt
                    routes[dst_idx] = new_dst_opt
                    # Remove empty routes
                    routes[:] = [r for r in routes if r]
                    return True

    return False


# ---------------------------------------------------------------------------
# Swap move: exchange one customer between route_a and route_b
# ---------------------------------------------------------------------------

def _try_swap(routes, D, demands, capacity, nodes, max_pair_fraction=1.0):
    """
    Try all one-for-one customer swaps between route pairs.
    Accept first improving move.

    Returns
    -------
    bool
    """
    pairs = _route_pair_order(routes, nodes)
    n_check = max(1, int(len(pairs) * max_pair_fraction))
    pairs = pairs[:n_check]

    for i, j in pairs:
        ri = routes[i]
        rj = routes[j]

        if not ri or not rj:
            continue

        d_ri = route_distance(ri, D)
        d_rj = route_distance(rj, D)
        base = d_ri + d_rj

        for ci_pos, ci in enumerate(ri):
            for cj_pos, cj in enumerate(rj):
                # Capacity check after swap
                new_load_i = route_load(ri, demands) - demands[ci] + demands[cj]
                new_load_j = route_load(rj, demands) - demands[cj] + demands[ci]
                if new_load_i > capacity or new_load_j > capacity:
                    continue

                new_ri = ri[:ci_pos] + [cj] + ri[ci_pos + 1:]
                new_rj = rj[:cj_pos] + [ci] + rj[cj_pos + 1:]

                d_new_ri = route_distance(new_ri, D)
                d_new_rj = route_distance(new_rj, D)

                if base - (d_new_ri + d_new_rj) > 1e-10:
                    # Accept: apply 2-opt and update
                    routes[i] = two_opt(new_ri, D)
                    routes[j] = two_opt(new_rj, D)
                    return True

    return False


# ---------------------------------------------------------------------------
# Public repair interface
# ---------------------------------------------------------------------------

def repair_routes(routes, D, demands, capacity, nodes,
                  max_iter=50, use_relocate=True, use_swap=True,
                  max_pair_fraction=1.0):
    """
    Iteratively apply relocate and swap moves until no improvement is found
    or max_iter is reached.

    Parameters
    ----------
    routes            : list[list[int]]  -- modified in place
    D                 : list[list[float]]
    demands           : list[int]
    capacity          : int
    nodes             : list[(float, float)]
    max_iter          : int
    use_relocate      : bool
    use_swap          : bool
    max_pair_fraction : float  -- limit route pairs checked (1.0 = all)

    Returns
    -------
    list[list[int]]   -- improved routes (same object, also returned for clarity)
    float             -- distance gain (>= 0)
    """
    before = total_solution_distance(routes, D)

    for _ in range(max_iter):
        improved = False

        if use_relocate:
            if _try_relocate(routes, D, demands, capacity, nodes, max_pair_fraction):
                improved = True
                continue  # restart with relocate

        if use_swap:
            if _try_swap(routes, D, demands, capacity, nodes, max_pair_fraction):
                improved = True
                continue

        if not improved:
            break

    after = total_solution_distance(routes, D)
    gain = before - after
    return routes, max(0.0, round(gain, 6))
