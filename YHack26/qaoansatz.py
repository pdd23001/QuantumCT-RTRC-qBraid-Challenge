import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, FrozenSet, Set

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

try:
    import pulp
except ImportError:
    raise ImportError("Install pulp first: pip install pulp")


# ============================================================
# Instances
# ============================================================

INSTANCES = {
    "instance_1": {
        "Nv": 5,
        "C": 4,
        "depot": (0, 0),
        "customers": {
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
    }
}


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class Route:
    customers: Tuple[int, ...]
    cost: float
    covered: FrozenSet[int]


# ============================================================
# Geometry helpers
# ============================================================

def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.dist(a, b)


def build_nodes(customers: Dict[int, Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
    nodes = {0: (0.0, 0.0)}
    nodes.update(customers)
    return nodes


def route_distance(route: Tuple[int, ...], nodes: Dict[int, Tuple[float, float]]) -> float:
    if not route:
        return 0.0

    total = euclidean(nodes[0], nodes[route[0]])
    for i in range(len(route) - 1):
        total += euclidean(nodes[route[i]], nodes[route[i + 1]])
    total += euclidean(nodes[route[-1]], nodes[0])
    return total


def canonical_route_key(route: Route):
    seq = route.customers
    rev = tuple(reversed(seq))
    return min(seq, rev)


def route_from_perm(perm: Tuple[int, ...], nodes: Dict[int, Tuple[float, float]]) -> Route:
    return Route(
        customers=perm,
        cost=route_distance(perm, nodes),
        covered=frozenset(perm),
    )


def polar_angle(pt: Tuple[float, float]) -> float:
    return math.atan2(pt[1], pt[0])


# ============================================================
# Local search repair for a single route ordering
# ============================================================

def two_opt_once(route: Tuple[int, ...], nodes: Dict[int, Tuple[float, float]]) -> Tuple[Tuple[int, ...], bool]:
    best = route
    best_cost = route_distance(best, nodes)
    n = len(route)

    for i in range(n):
        for j in range(i + 1, n):
            cand = route[:i] + tuple(reversed(route[i:j + 1])) + route[j + 1:]
            cand_cost = route_distance(cand, nodes)
            if cand_cost + 1e-12 < best_cost:
                return cand, True

    return best, False


def relocate_once(route: Tuple[int, ...], nodes: Dict[int, Tuple[float, float]]) -> Tuple[Tuple[int, ...], bool]:
    best = route
    best_cost = route_distance(best, nodes)
    n = len(route)

    for i in range(n):
        node = route[i]
        rest = list(route[:i] + route[i + 1:])
        for j in range(len(rest) + 1):
            cand_list = rest[:j] + [node] + rest[j:]
            cand = tuple(cand_list)
            if cand == route:
                continue
            cand_cost = route_distance(cand, nodes)
            if cand_cost + 1e-12 < best_cost:
                return cand, True

    return best, False


def swap_once(route: Tuple[int, ...], nodes: Dict[int, Tuple[float, float]]) -> Tuple[Tuple[int, ...], bool]:
    best = route
    best_cost = route_distance(best, nodes)
    n = len(route)

    for i in range(n):
        for j in range(i + 1, n):
            cand_list = list(route)
            cand_list[i], cand_list[j] = cand_list[j], cand_list[i]
            cand = tuple(cand_list)
            cand_cost = route_distance(cand, nodes)
            if cand_cost + 1e-12 < best_cost:
                return cand, True

    return best, False


def improve_route_local_search(
    route: Tuple[int, ...],
    nodes: Dict[int, Tuple[float, float]],
    max_rounds: int = 10,
) -> Tuple[int, ...]:
    """
    Tiny local search:
      - 2-opt
      - relocate
      - swap

    Repeats until no improvement or max_rounds reached.
    """
    current = route
    rounds = 0

    while rounds < max_rounds:
        rounds += 1
        improved = False

        for move in (two_opt_once, relocate_once, swap_once):
            candidate, did_improve = move(current, nodes)
            if did_improve:
                current = candidate
                improved = True
                break

        if not improved:
            break

    return current


# ============================================================
# Initial route pool
# ============================================================

def initial_feasible_routes(
    customers: Dict[int, Tuple[float, float]],
    Nv: int,
    C: int,
) -> List[Route]:
    """
    Build an initial feasible route pool.

    Includes:
      - singleton routes
      - pair routes
      - a geometry-aware feasible incumbent
    """
    nodes = build_nodes(customers)
    cust_ids = sorted(customers.keys())
    n = len(cust_ids)

    if n > Nv * C:
        raise ValueError(
            f"Instance infeasible: {n} customers but Nv*C = {Nv*C} capacity slots."
        )

    routes: List[Route] = []

    # Singletons
    for c in cust_ids:
        routes.append(route_from_perm((c,), nodes))

    # Pair routes
    if C >= 2:
        for i in range(len(cust_ids)):
            for j in range(i + 1, len(cust_ids)):
                a, b = cust_ids[i], cust_ids[j]
                routes.append(route_from_perm((a, b), nodes))
                routes.append(route_from_perm((b, a), nodes))

    # Geometry-aware grouping to build a feasible incumbent
    unassigned = set(cust_ids)
    groups = []

    seeds = sorted(
        cust_ids,
        key=lambda c: euclidean(nodes[0], nodes[c]),
        reverse=True
    )[:Nv]

    for s in seeds:
        if s in unassigned:
            groups.append([s])
            unassigned.remove(s)

    while len(groups) < min(Nv, n) and unassigned:
        c = next(iter(unassigned))
        groups.append([c])
        unassigned.remove(c)

    while unassigned:
        best_choice = None
        best_dist = float("inf")
        for c in unassigned:
            for g_idx, group in enumerate(groups):
                if len(group) >= C:
                    continue
                d = min(euclidean(nodes[c], nodes[u]) for u in group)
                if d < best_dist:
                    best_dist = d
                    best_choice = (c, g_idx)

        if best_choice is None:
            raise ValueError("Could not construct initial feasible grouped solution.")

        c, g_idx = best_choice
        groups[g_idx].append(c)
        unassigned.remove(c)

    for group in groups:
        remaining = set(group)
        start = min(remaining, key=lambda c: euclidean(nodes[0], nodes[c]))
        order = [start]
        remaining.remove(start)

        while remaining:
            last = order[-1]
            nxt = min(remaining, key=lambda c: euclidean(nodes[last], nodes[c]))
            order.append(nxt)
            remaining.remove(nxt)

        improved = improve_route_local_search(tuple(order), nodes)
        routes.append(route_from_perm(improved, nodes))
        routes.append(route_from_perm(tuple(reversed(improved)), nodes))

    return list(set(routes))


# ============================================================
# Restricted Master Problem (LP relaxation)
# ============================================================

def solve_rmp_lp(routes: List[Route], customer_ids: List[int], Nv: int):
    model = pulp.LpProblem("RMP_LP", pulp.LpMinimize)

    x = [
        pulp.LpVariable(f"x_{r}", lowBound=0, upBound=1, cat="Continuous")
        for r in range(len(routes))
    ]

    model += pulp.lpSum(routes[r].cost * x[r] for r in range(len(routes)))

    cover_cons = {}
    for i in customer_ids:
        cname = f"cover_{i}"
        model += pulp.lpSum(x[r] for r in range(len(routes)) if i in routes[r].covered) == 1, cname
        cover_cons[i] = model.constraints[cname]

    model += pulp.lpSum(x[r] for r in range(len(routes))) <= Nv, "vehicle_limit"
    vehicle_con = model.constraints["vehicle_limit"]

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"RMP LP solve failed: {status}")

    dual_customer = {i: cover_cons[i].pi for i in customer_ids}
    dual_vehicle = vehicle_con.pi
    obj = pulp.value(model.objective)

    return obj, dual_customer, dual_vehicle


# ============================================================
# Final Master Problem (integer)
# ============================================================

def solve_master_ip(routes: List[Route], customer_ids: List[int], Nv: int):
    model = pulp.LpProblem("MASTER_IP", pulp.LpMinimize)

    x = [
        pulp.LpVariable(f"x_{r}", lowBound=0, upBound=1, cat="Binary")
        for r in range(len(routes))
    ]

    model += pulp.lpSum(routes[r].cost * x[r] for r in range(len(routes)))

    for i in customer_ids:
        model += pulp.lpSum(x[r] for r in range(len(routes)) if i in routes[r].covered) == 1

    model += pulp.lpSum(x[r] for r in range(len(routes))) <= Nv

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"Master IP solve failed: {status}")

    selected = [r for r in range(len(routes)) if pulp.value(x[r]) > 0.5]
    total_cost = pulp.value(model.objective)
    return total_cost, selected


# ============================================================
# Reduced cost
# ============================================================

def reduced_cost(route: Route, dual_customer: Dict[int, float], dual_vehicle: float) -> float:
    return route.cost - sum(dual_customer[i] for i in route.covered) - dual_vehicle


# ============================================================
# Smart candidate generation (NO brute force)
# ============================================================

def build_knn(customers: Dict[int, Tuple[float, float]], k: int) -> Dict[int, List[int]]:
    nodes = build_nodes(customers)
    cust_ids = sorted(customers.keys())
    knn = {}

    for i in cust_ids:
        nbrs = sorted(
            [j for j in cust_ids if j != i],
            key=lambda j: euclidean(nodes[i], nodes[j])
        )
        knn[i] = nbrs[:k]

    return knn


def greedy_order_from_subset(
    subset: Set[int],
    nodes: Dict[int, Tuple[float, float]]
) -> Tuple[int, ...]:
    remaining = set(subset)
    start = min(remaining, key=lambda c: euclidean(nodes[0], nodes[c]))
    order = [start]
    remaining.remove(start)

    while remaining:
        last = order[-1]
        nxt = min(remaining, key=lambda c: euclidean(nodes[last], nodes[c]))
        order.append(nxt)
        remaining.remove(nxt)

    return tuple(order)


def dual_guided_expand(
    seed: int,
    customers: Dict[int, Tuple[float, float]],
    dual_customer: Dict[int, float],
    C: int,
    knn: Dict[int, List[int]],
    variant: int = 0,
) -> Set[int]:
    """
    Build a promising subset around a seed using neighbor geometry + dual scores.
    """
    nodes = build_nodes(customers)
    chosen = [seed]
    chosen_set = {seed}

    while len(chosen) < C:
        candidates = set()
        for u in chosen:
            candidates.update(knn[u])

        candidates = [c for c in candidates if c not in chosen_set]
        if not candidates:
            candidates = [c for c in customers if c not in chosen_set]

        if not candidates:
            break

        def score(c):
            near = min(euclidean(nodes[c], nodes[u]) for u in chosen)
            depot = euclidean(nodes[0], nodes[c])

            if variant == 0:
                return dual_customer.get(c, 0.0) - 0.8 * near - 0.2 * depot
            elif variant == 1:
                return dual_customer.get(c, 0.0) - 0.5 * near - 0.5 * depot
            elif variant == 2:
                return dual_customer.get(c, 0.0) - 1.0 * near
            else:
                return dual_customer.get(c, 0.0) - 0.3 * near - 0.7 * depot

        nxt = max(candidates, key=score)
        chosen.append(nxt)
        chosen_set.add(nxt)

    return chosen_set


def subset_variants(
    subset: Set[int],
    nodes: Dict[int, Tuple[float, float]],
) -> List[Tuple[int, ...]]:
    """
    Build a few cheap orderings without enumerating all permutations.
    """
    if not subset:
        return []

    base = greedy_order_from_subset(subset, nodes)
    variants = {base, tuple(reversed(base))}

    if len(base) >= 3:
        # angle-based ordering
        angle_order = tuple(sorted(subset, key=lambda c: polar_angle(nodes[c])))
        variants.add(angle_order)
        variants.add(tuple(reversed(angle_order)))

        # farthest-from-depot first then greedy continuation
        start = max(subset, key=lambda c: euclidean(nodes[0], nodes[c]))
        remaining = set(subset)
        order = [start]
        remaining.remove(start)
        while remaining:
            last = order[-1]
            nxt = min(remaining, key=lambda c: euclidean(nodes[last], nodes[c]))
            order.append(nxt)
            remaining.remove(nxt)
        variants.add(tuple(order))
        variants.add(tuple(reversed(order)))

    return list(variants)


def generate_smart_candidate_routes(
    customers: Dict[int, Tuple[float, float]],
    dual_customer: Dict[int, float],
    dual_vehicle: float,
    C: int,
    route_pool_canonical: Set[Tuple[int, ...]],
    neighborhood_k: int = 6,
    max_generated_routes: int = 1500,
    ls_max_rounds: int = 10,
) -> List[Route]:
    """
    Generate a smart, limited set of candidate routes using:
      - dual-guided seed expansion
      - geometric neighborhoods
      - sweep windows
      - small route ordering variants
      - local search repair on each ordering

    This replaces exhaustive enumeration.
    """
    del dual_vehicle  # not needed in generation directly

    nodes = build_nodes(customers)
    cust_ids = sorted(customers.keys())
    knn = build_knn(customers, k=min(neighborhood_k, max(1, len(cust_ids) - 1)))

    generated: List[Route] = []
    seen_canonical_subsets = set()
    seen_route_keys = set()

    def try_add_subset(subset: Set[int]):
        if not subset:
            return

        frozen_subset = frozenset(subset)
        if frozen_subset in seen_canonical_subsets:
            return
        seen_canonical_subsets.add(frozen_subset)

        for perm in subset_variants(subset, nodes):
            improved_perm = improve_route_local_search(
                perm,
                nodes,
                max_rounds=ls_max_rounds,
            )
            route = route_from_perm(improved_perm, nodes)
            key = canonical_route_key(route)

            if key in route_pool_canonical or key in seen_route_keys:
                continue

            seen_route_keys.add(key)
            generated.append(route)

    # --------------------------------------------------------
    # 1) Seed-based dual-guided expansions
    # --------------------------------------------------------
    seeds = sorted(
        cust_ids,
        key=lambda c: (
            dual_customer.get(c, 0.0),
            -euclidean(nodes[0], nodes[c])
        ),
        reverse=True
    )

    for seed in seeds:
        for variant in range(4):
            subset = dual_guided_expand(
                seed=seed,
                customers=customers,
                dual_customer=dual_customer,
                C=C,
                knn=knn,
                variant=variant,
            )

            for take in range(1, min(C, len(subset)) + 1):
                ordered = sorted(
                    subset,
                    key=lambda c: (
                        dual_customer.get(c, 0.0) - 0.2 * euclidean(nodes[0], nodes[c])
                    ),
                    reverse=True
                )
                try_add_subset(set(ordered[:take]))

            if len(generated) >= max_generated_routes:
                break
        if len(generated) >= max_generated_routes:
            break

    # --------------------------------------------------------
    # 2) Sweep windows by angle
    # --------------------------------------------------------
    angle_sorted = sorted(cust_ids, key=lambda c: polar_angle(nodes[c]))
    m = len(angle_sorted)
    for start in range(m):
        for width in range(2, min(C, m) + 1):
            subset = {angle_sorted[(start + t) % m] for t in range(width)}
            try_add_subset(subset)
            if len(generated) >= max_generated_routes:
                break
        if len(generated) >= max_generated_routes:
            break

    # --------------------------------------------------------
    # 3) KNN-centered local subsets
    # --------------------------------------------------------
    for center in cust_ids:
        local = [center] + knn[center][:max(0, C - 1)]
        for width in range(2, min(C, len(local)) + 1):
            subset = set(local[:width])
            try_add_subset(subset)
            if len(generated) >= max_generated_routes:
                break
        if len(generated) >= max_generated_routes:
            break

    return generated


# ============================================================
# QAOAnsatz-style route-space encoding in Qiskit
# ============================================================

def num_qubits_for_routes(num_routes: int) -> int:
    return max(1, math.ceil(math.log2(max(1, num_routes))))


def build_route_space_mixer(routes: List[Route]) -> np.ndarray:
    n = len(routes)
    Hm = np.zeros((n, n), dtype=np.complex128)

    for i in range(n):
        set_i = set(routes[i].covered)
        seq_i = routes[i].customers

        for j in range(i + 1, n):
            set_j = set(routes[j].covered)
            seq_j = routes[j].customers

            sym_diff = len(set_i.symmetric_difference(set_j))
            same_set = (set_i == set_j)

            connect = False
            if same_set and seq_i != seq_j:
                connect = True
            elif sym_diff <= 2:
                connect = True

            if connect:
                Hm[i, j] = 1.0
                Hm[j, i] = 1.0

    return Hm


def embed_square_matrix(mat_small: np.ndarray, dim_big: int, pad_diag: float = 0.0) -> np.ndarray:
    out = np.zeros((dim_big, dim_big), dtype=np.complex128)
    n = mat_small.shape[0]
    out[:n, :n] = mat_small
    if pad_diag != 0.0:
        for i in range(n, dim_big):
            out[i, i] = pad_diag
    return out


def build_cost_hamiltonian(candidate_routes: List[Route], dual_customer, dual_vehicle) -> np.ndarray:
    c = np.array(
        [reduced_cost(r, dual_customer, dual_vehicle) for r in candidate_routes],
        dtype=float
    )
    return np.diag(c.astype(np.complex128))


def build_initial_state(candidate_routes: List[Route], n_qubits: int) -> np.ndarray:
    dim = 2 ** n_qubits
    psi = np.zeros(dim, dtype=np.complex128)

    singleton_indices = [i for i, r in enumerate(candidate_routes) if len(r.customers) == 1]
    active = singleton_indices if singleton_indices else list(range(len(candidate_routes)))

    amp = 1.0 / math.sqrt(len(active))
    for idx in active:
        psi[idx] = amp

    return psi


def make_qaoansatz_circuit(
    candidate_routes: List[Route],
    dual_customer: Dict[int, float],
    dual_vehicle: float,
    gammas: List[float],
    betas: List[float],
) -> QuantumCircuit:
    assert len(gammas) == len(betas)
    p = len(gammas)

    num_routes = len(candidate_routes)
    n_qubits = num_qubits_for_routes(num_routes)
    dim = 2 ** n_qubits

    Hc_small = build_cost_hamiltonian(candidate_routes, dual_customer, dual_vehicle)
    Hm_small = build_route_space_mixer(candidate_routes)

    Hc = embed_square_matrix(Hc_small, dim_big=dim, pad_diag=50.0)
    Hm = embed_square_matrix(Hm_small, dim_big=dim, pad_diag=0.0)

    psi0 = build_initial_state(candidate_routes, n_qubits)

    qc = QuantumCircuit(n_qubits)
    qc.initialize(psi0, list(range(n_qubits)))

    for gamma, beta in zip(gammas, betas):
        Uc = expm(-1j * gamma * Hc)
        Um = expm(-1j * beta * Hm)

        qc.unitary(Operator(Uc), list(range(n_qubits)), label="Phase")
        qc.unitary(Operator(Um), list(range(n_qubits)), label="Mixer")

    return qc


def statevector_probabilities(qc: QuantumCircuit) -> np.ndarray:
    psi = Statevector.from_instruction(qc)
    return np.abs(psi.data) ** 2


def qaoansatz_pricing_qiskit(
    candidate_routes: List[Route],
    dual_customer: Dict[int, float],
    dual_vehicle: float,
    p: int = 2,
    maxiter: int = 150,
    top_k: int = 15,
    seed: int = 0,
):
    if not candidate_routes:
        return {
            "params": {"gammas": [], "betas": []},
            "ranked_routes": [],
            "best_expectation": float("inf"),
            "circuit": None,
        }

    rng = np.random.default_rng(seed)

    costs = np.array(
        [reduced_cost(r, dual_customer, dual_vehicle) for r in candidate_routes],
        dtype=float
    )

    def expected_cost(params: np.ndarray) -> float:
        gammas = params[:p].tolist()
        betas = params[p:].tolist()

        qc = make_qaoansatz_circuit(
            candidate_routes=candidate_routes,
            dual_customer=dual_customer,
            dual_vehicle=dual_vehicle,
            gammas=gammas,
            betas=betas,
        )
        probs = statevector_probabilities(qc)
        probs_routes = probs[:len(candidate_routes)]
        return float(np.dot(probs_routes, costs))

    x0 = np.concatenate([
        rng.uniform(0.0, 1.0, size=p),
        rng.uniform(0.0, 1.0, size=p),
    ])

    res = minimize(
        expected_cost,
        x0,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.5, "disp": False},
    )

    gammas = res.x[:p].tolist()
    betas = res.x[p:].tolist()

    qc = make_qaoansatz_circuit(
        candidate_routes=candidate_routes,
        dual_customer=dual_customer,
        dual_vehicle=dual_vehicle,
        gammas=gammas,
        betas=betas,
    )
    probs = statevector_probabilities(qc)[:len(candidate_routes)]

    ranked = sorted(
        [
            (idx, probs[idx], costs[idx], candidate_routes[idx])
            for idx in range(len(candidate_routes))
        ],
        key=lambda x: (x[2], -x[1])
    )

    return {
        "params": {
            "gammas": gammas,
            "betas": betas,
        },
        "ranked_routes": ranked[:top_k],
        "best_expectation": float(res.fun),
        "circuit": qc,
    }


# ============================================================
# Column generation + smart pricing
# ============================================================

def column_generation_qaoansatz_qiskit(
    customers: Dict[int, Tuple[float, float]],
    Nv: int,
    C: int,
    p: int = 2,
    top_k_add: int = 15,
    max_cg_iters: int = 50,
    pricing_maxiter: int = 150,
    max_candidates: int = 128,
    neighborhood_k: int = 6,
    max_generated_routes: int = 1500,
    ls_max_rounds: int = 10,
    seed: int = 0,
    verbose: bool = True,
):
    if len(customers) > Nv * C:
        raise ValueError(
            f"Instance is globally infeasible: {len(customers)} customers > Nv*C = {Nv*C}"
        )

    customer_ids = sorted(customers.keys())

    route_pool = list(initial_feasible_routes(customers, Nv=Nv, C=C))
    route_pool_set = set(route_pool)
    route_pool_canonical = {canonical_route_key(r) for r in route_pool}

    if verbose:
        print(f"Initial route pool size: {len(route_pool)}")

    last_qc = None

    for it in range(1, max_cg_iters + 1):
        lp_obj, dual_customer, dual_vehicle = solve_rmp_lp(route_pool, customer_ids, Nv)

        candidates_all = generate_smart_candidate_routes(
            customers=customers,
            dual_customer=dual_customer,
            dual_vehicle=dual_vehicle,
            C=C,
            route_pool_canonical=route_pool_canonical,
            neighborhood_k=neighborhood_k,
            max_generated_routes=max_generated_routes,
            ls_max_rounds=ls_max_rounds,
        )

        candidates_all = sorted(
            candidates_all,
            key=lambda r: reduced_cost(r, dual_customer, dual_vehicle)
        )

        candidates = candidates_all[:max_candidates]

        if verbose:
            print(f"\nCG iteration {it}")
            print(f"  LP objective            : {lp_obj:.4f}")
            print(f"  dual_vehicle            : {dual_vehicle:.4f}")
            print(f"  generated candidates    : {len(candidates_all)}")
            print(f"  qiskit candidate subset : {len(candidates)}")

        pricing = qaoansatz_pricing_qiskit(
            candidate_routes=candidates,
            dual_customer=dual_customer,
            dual_vehicle=dual_vehicle,
            p=p,
            maxiter=pricing_maxiter,
            top_k=top_k_add,
            seed=seed + it,
        )

        last_qc = pricing["circuit"]
        ranked = pricing["ranked_routes"]
        improving = [(idx, prob, rc, route) for idx, prob, rc, route in ranked if rc < -1e-8]

        if verbose:
            if ranked:
                print(f"  best expectation        : {pricing['best_expectation']:.4f}")
                print(f"  top reduced costs       : {[round(x[2], 4) for x in ranked]}")
            else:
                print("  No pricing candidates left.")

        if not improving:
            if verbose:
                print("  No negative reduced-cost route found. Converged.")
            break

        added = 0
        for _, prob, rc, route in improving:
            key = canonical_route_key(route)
            if route not in route_pool_set and key not in route_pool_canonical:
                route_pool.append(route)
                route_pool_set.add(route)
                route_pool_canonical.add(key)
                added += 1
                if verbose:
                    print(
                        f"  Added route {route.customers} "
                        f"(prob={prob:.4f}, reduced_cost={rc:.4f}, cost={route.cost:.4f})"
                    )

        if added == 0:
            if verbose:
                print("  No new routes added. Stopping.")
            break

    total_cost, selected_idx = solve_master_ip(route_pool, customer_ids, Nv)
    selected_routes = [route_pool[i] for i in selected_idx]

    return {
        "total_cost": total_cost,
        "selected_routes": selected_routes,
        "route_pool_size": len(route_pool),
        "last_pricing_circuit": last_qc,
    }


# ============================================================
# Pretty printing
# ============================================================

def print_solution(name: str, result: dict):
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    print(f"Final total distance: {result['total_cost']:.4f}")
    print(f"Route pool size     : {result['route_pool_size']}")
    print("Selected routes:")
    for r in result["selected_routes"]:
        path = " -> ".join(str(x) for x in r.customers)
        print(f"  0 -> {path} -> 0    cost={r.cost:.4f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    for name, inst in INSTANCES.items():
        print(f"\nRunning {name}...")
        result = column_generation_qaoansatz_qiskit(
            customers=inst["customers"],
            Nv=inst["Nv"],
            C=inst["C"],
            p=2,
            top_k_add=15,
            max_cg_iters=50,
            pricing_maxiter=150,
            max_candidates=128,
            neighborhood_k=6,
            max_generated_routes=1500,
            ls_max_rounds=10,
            seed=42,
            verbose=True,
        )
        print_solution(name, result)