"""
local_search.py
===============
Classical intra-route local search improvements.

Implemented
-----------
- two_opt      : standard 2-opt improvement for a single route
- two_opt_all  : apply 2-opt to every route in a solution
- three_opt    : bounded 3-opt improvement (optional, slower)
- nearest_neighbor_route : greedy nearest-neighbor route builder
"""

from common import route_distance


# ---------------------------------------------------------------------------
# 2-opt
# ---------------------------------------------------------------------------

def two_opt(route, D, max_iter=None):
    """
    Improve a single route with 2-opt.  Only accepts improving moves.

    Parameters
    ----------
    route    : list[int]  -- customer IDs (no depot at ends)
    D        : list[list[float]]  -- distance matrix, depot = index 0
    max_iter : int or None  -- maximum passes; None = run until no improvement

    Returns
    -------
    list[int] -- improved route (new list, original unmodified)
    """
    best = route[:]
    improved = True
    iters = 0

    while improved:
        if max_iter is not None and iters >= max_iter:
            break
        improved = False
        iters += 1
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                # Current cost of edge (i, i+1) + edge (j, j+1)
                # After reversal: edge (i, j) + edge (i+1, j+1)
                a = best[i]
                b = best[i + 1]
                c = best[j]
                d = best[j + 1] if j + 1 < len(best) else 0  # next node (0 = depot)

                prev_a = best[i - 1] if i > 0 else 0  # unused in standard 2-opt
                # Standard 2-opt: reverse the segment best[i+1 .. j]

                before = D[a][b] + D[c][d]
                after = D[a][c] + D[b][d]

                if after < before - 1e-10:
                    best[i + 1:j + 1] = best[i + 1:j + 1][::-1]
                    improved = True

    return best


def two_opt_all(routes, D, max_iter=None):
    """
    Apply 2-opt to every route in a solution.

    Parameters
    ----------
    routes   : list[list[int]]
    D        : list[list[float]]
    max_iter : int or None

    Returns
    -------
    list[list[int]]  -- new route list with improved routes
    """
    return [two_opt(r, D, max_iter=max_iter) for r in routes]


# ---------------------------------------------------------------------------
# 3-opt (bounded / simplified)
# ---------------------------------------------------------------------------

def three_opt(route, D, max_iter=3):
    """
    Simplified 3-opt: try all triple edge swaps that reduce route cost.
    Only the 2-opt-style reconnections are checked (not full 8-option 3-opt)
    to keep complexity manageable.  For routes with n > 20, this can be slow;
    caller should gate on route length.

    Parameters
    ----------
    route    : list[int]
    D        : list[list[float]]
    max_iter : int  -- maximum outer passes

    Returns
    -------
    list[int]
    """
    best = route[:]
    n = len(best)

    if n < 4:
        return best

    def segment_cost(r):
        return route_distance(r, D)

    for _ in range(max_iter):
        improved = False
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    # Segment borders (depot = 0)
                    A = best[i]
                    B = best[i + 1]
                    C = best[j]
                    E = best[j + 1]
                    F = best[k]
                    G = best[k + 1] if k + 1 < n else 0

                    d0 = D[A][B] + D[C][E] + D[F][G]  # current

                    # Candidate reconnection: reverse segment j+1..k
                    d1 = D[A][B] + D[C][F] + D[E][G]
                    if d1 < d0 - 1e-10:
                        best[j + 1:k + 1] = best[j + 1:k + 1][::-1]
                        improved = True
                        continue

                    # Candidate reconnection: reverse segment i+1..j
                    d2 = D[A][C] + D[B][E] + D[F][G]
                    if d2 < d0 - 1e-10:
                        best[i + 1:j + 1] = best[i + 1:j + 1][::-1]
                        improved = True
                        continue

        if not improved:
            break

    return best


def three_opt_all(routes, D, max_route_len=20, max_iter=3):
    """
    Apply 3-opt to routes shorter than max_route_len; 2-opt for longer ones.

    Parameters
    ----------
    routes        : list[list[int]]
    D             : list[list[float]]
    max_route_len : int
    max_iter      : int

    Returns
    -------
    list[list[int]]
    """
    result = []
    for r in routes:
        if len(r) <= max_route_len:
            result.append(three_opt(r, D, max_iter=max_iter))
        else:
            result.append(two_opt(r, D))
    return result


# ---------------------------------------------------------------------------
# Nearest-neighbor route builder (standalone utility)
# ---------------------------------------------------------------------------

def nearest_neighbor_route(customers, D):
    """
    Build a single route visiting all given customers using a
    nearest-neighbor greedy heuristic starting from the depot (index 0).

    Parameters
    ----------
    customers : list[int]  -- customer IDs to visit
    D         : list[list[float]]

    Returns
    -------
    list[int]  -- ordered customer IDs
    """
    if not customers:
        return []
    remaining = list(customers)
    route = []
    current = 0  # depot
    while remaining:
        nxt = min(remaining, key=lambda c: D[current][c])
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return route


# ---------------------------------------------------------------------------
# Cleanup pass
# ---------------------------------------------------------------------------

def cleanup_solution(routes, D, use_three_opt=False, three_opt_threshold=15):
    """
    Apply 2-opt (and optionally 3-opt) to all routes.

    Parameters
    ----------
    routes              : list[list[int]]
    D                   : list[list[float]]
    use_three_opt       : bool
    three_opt_threshold : int  -- apply 3-opt only if route len <= this

    Returns
    -------
    list[list[int]]
    """
    if use_three_opt:
        return three_opt_all(routes, D, max_route_len=three_opt_threshold)
    return two_opt_all(routes, D)
