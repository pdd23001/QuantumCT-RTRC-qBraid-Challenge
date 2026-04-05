import math
import itertools
from typing import Dict, Tuple, List, Set, Optional

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

# Try Aer first
try:
    from qiskit_aer import Aer
    HAS_AER = True
except Exception:
    HAS_AER = False


class QiskitDQICVRP:
    """
    Genuine quantum, Qiskit-based DQI-style CVRP prototype.

    Core idea:
    - Work on a small local neighborhood of q customers at a time.
    - Represent subset choice by q qubits.
    - Prepare a quantum state whose amplitudes are biased toward
      better feasible subsets (DQI-style biased sampling).
    - Sample promising subsets quantumly.
    - Classically order each chosen subset into an exact mini-route.
    - Repeat until all customers are assigned.

    This is a practical DQI-style pricing/subset sampler, not the
    full paper's reversible syndrome-decoding construction.
    """

    def __init__(
        self,
        customers: Dict[int, Tuple[float, float]],
        Nv: int,
        C: int,
        depot: Tuple[float, float] = (0.0, 0.0),
        neighborhood_qubits: int = 6,
        shots: int = 2048,
        alpha: float = 3.0,
        seed: int = 0,
    ):
        self.customers = customers
        self.Nv = Nv
        self.C = C
        self.depot = depot
        self.q = neighborhood_qubits
        self.shots = shots
        self.alpha = alpha
        self.seed = seed

        self.coords = {0: depot, **customers}
        self.customer_ids = sorted(customers.keys())
        self.nodes = [0] + self.customer_ids
        self.n = len(self.customer_ids)

        self.dist = {
            (i, j): self._euclidean(i, j)
            for i in self.nodes for j in self.nodes
        }

    # =========================================================
    # Geometry
    # =========================================================
    def _euclidean(self, i: int, j: int) -> float:
        x1, y1 = self.coords[i]
        x2, y2 = self.coords[j]
        return math.hypot(x1 - x2, y1 - y2)

    def route_cost(self, route: List[int]) -> float:
        if not route:
            return 0.0
        total = self.dist[(0, route[0])]
        for a, b in zip(route[:-1], route[1:]):
            total += self.dist[(a, b)]
        total += self.dist[(route[-1], 0)]
        return total

    # =========================================================
    # Exact ordering for a chosen subset: Held-Karp
    # =========================================================
    def best_order_for_subset(self, subset: List[int]) -> Tuple[List[int], float]:
        subset = list(subset)
        m = len(subset)
        if m == 0:
            return [], 0.0
        if m == 1:
            return subset[:], self.route_cost(subset)

        idx_to_customer = {i: subset[i] for i in range(m)}
        DP = {}
        parent = {}

        for j in range(m):
            mask = 1 << j
            cj = idx_to_customer[j]
            DP[(mask, j)] = self.dist[(0, cj)]
            parent[(mask, j)] = None

        for mask in range(1, 1 << m):
            for j in range(m):
                if not (mask & (1 << j)):
                    continue
                if (mask, j) not in DP:
                    continue
                for nxt in range(m):
                    if mask & (1 << nxt):
                        continue
                    new_mask = mask | (1 << nxt)
                    cj = idx_to_customer[j]
                    cn = idx_to_customer[nxt]
                    cand = DP[(mask, j)] + self.dist[(cj, cn)]
                    if (new_mask, nxt) not in DP or cand < DP[(new_mask, nxt)]:
                        DP[(new_mask, nxt)] = cand
                        parent[(new_mask, nxt)] = j

        full = (1 << m) - 1
        best_cost = float("inf")
        best_last = None

        for j in range(m):
            cj = idx_to_customer[j]
            cand = DP[(full, j)] + self.dist[(cj, 0)]
            if cand < best_cost:
                best_cost = cand
                best_last = j

        order_idx = []
        mask = full
        j = best_last
        while j is not None:
            order_idx.append(j)
            prev = parent[(mask, j)]
            mask ^= (1 << j)
            j = prev
        order_idx.reverse()

        order = [idx_to_customer[i] for i in order_idx]
        return order, best_cost

    # =========================================================
    # Local neighborhood
    # =========================================================
    def nearest_neighbors(self, seed: int, pool: Set[int], k: int) -> List[int]:
        arr = list(pool)
        arr.sort(key=lambda c: self.dist[(seed, c)])
        return arr[:k]

    # =========================================================
    # Classical surrogate score for a subset
    # This is the "f(x)" stand-in for local CVRP pricing.
    # =========================================================
    def subset_score(self, subset: List[int]) -> float:
        if len(subset) == 0:
            return -1e9
        if len(subset) > self.C:
            return -1e9

        order, route_c = self.best_order_for_subset(subset)

        pair_dispersion = 0.0
        for i, j in itertools.combinations(subset, 2):
            pair_dispersion += self.dist[(i, j)]

        depot_pull = sum(self.dist[(0, i)] for i in subset)

        # Higher is better
        score = (
            8.0 * len(subset)
            - 1.0 * route_c
            - 0.07 * pair_dispersion
            - 0.03 * depot_pull
        )
        return score

    # =========================================================
    # DQI-style polynomial bias
    # We use a positive polynomial transform P(f)
    # to amplify better local subsets.
    # =========================================================
    def polynomial_bias(self, normalized_score: float) -> float:
        # normalized_score should be in [0, 1]
        x = max(0.0, min(1.0, normalized_score))
        return (x ** self.alpha) + 1e-12

    # =========================================================
    # Build amplitude vector over all 2^q subset bitstrings
    # =========================================================
    def build_amplitudes_for_neighborhood(
        self,
        neighborhood: List[int],
        must_include: Optional[int] = None,
        target_size: Optional[int] = None,
    ) -> np.ndarray:
        q = len(neighborhood)
        size = 1 << q
        weights = np.zeros(size, dtype=float)

        scored = []

        for mask in range(size):
            subset = []
            for b in range(q):
                if mask & (1 << b):
                    subset.append(neighborhood[b])

            if must_include is not None and must_include not in subset:
                continue
            if len(subset) == 0:
                continue
            if len(subset) > self.C:
                continue
            if target_size is not None and len(subset) != target_size:
                continue

            score = self.subset_score(subset)
            scored.append((mask, score))

        if not scored:
            return weights

        vals = [s for _, s in scored]
        mn, mx = min(vals), max(vals)

        if abs(mx - mn) < 1e-12:
            for mask, _ in scored:
                weights[mask] = 1.0
        else:
            for mask, score in scored:
                ns = (score - mn) / (mx - mn)
                weights[mask] = self.polynomial_bias(ns)

        # DQI-style state amplitudes proportional to P(f)
        norm = np.linalg.norm(weights)
        if norm < 1e-15:
            return weights
        return weights / norm

    # =========================================================
    # Genuine quantum sampling with Qiskit
    # =========================================================
    def quantum_sample_subset(
        self,
        neighborhood: List[int],
        must_include: Optional[int] = None,
        target_size: Optional[int] = None,
    ) -> List[int]:
        q = len(neighborhood)
        amps = self.build_amplitudes_for_neighborhood(
            neighborhood=neighborhood,
            must_include=must_include,
            target_size=target_size,
        )

        if np.linalg.norm(amps) < 1e-15:
            return [must_include] if must_include is not None else []

        qc = QuantumCircuit(q, q)
        qc.initialize(amps, list(range(q)))
        qc.measure(range(q), range(q))

        counts = None

        if HAS_AER:
            backend = Aer.get_backend("aer_simulator")
            tqc = transpile(qc, backend, seed_transpiler=self.seed)
            result = backend.run(
                tqc,
                shots=self.shots,
                seed_simulator=self.seed
            ).result()
            counts = result.get_counts()
        else:
            # fallback exact statevector sample
            bare = QuantumCircuit(q)
            bare.initialize(amps, list(range(q)))
            sv = Statevector.from_instruction(bare)
            probs = sv.probabilities_dict()
            counts = {
                k: int(round(v * self.shots))
                for k, v in probs.items()
                if v > 1e-12
            }

        if not counts:
            return [must_include] if must_include is not None else []

        # Most frequent measured subset
        best_bitstring = max(counts.items(), key=lambda kv: kv[1])[0]

        # Qiskit bitstrings are little-endian relative to qubit index
        subset = []
        bits = best_bitstring[::-1]
        for idx, bit in enumerate(bits):
            if bit == "1":
                subset.append(neighborhood[idx])

        # Safety cleanup
        if must_include is not None and must_include not in subset:
            subset = [must_include] + [c for c in subset if c != must_include]

        if len(subset) > self.C:
            subset = subset[:self.C]

        return subset

    # =========================================================
    # Build a full CVRP solution route-by-route
    # =========================================================
    def build_solution(self) -> List[List[int]]:
        remaining = set(self.customer_ids)
        routes = []

        # Because your example has Nv=5, C=4, n=20 exactly,
        # we target route size 4 on each round.
        while remaining:
            seed = max(remaining, key=lambda c: self.dist[(0, c)])

            neighborhood = [seed]
            neighborhood += self.nearest_neighbors(
                seed,
                remaining - {seed},
                self.q - 1
            )
            neighborhood = list(dict.fromkeys(neighborhood))

            subset = self.quantum_sample_subset(
                neighborhood=neighborhood,
                must_include=seed,
                target_size=min(self.C, len(remaining), len(neighborhood))
            )

            subset = [c for c in subset if c in remaining]

            # If the sampled subset is too small, fill greedily
            if len(subset) < min(self.C, len(remaining)):
                missing = min(self.C, len(remaining)) - len(subset)
                candidates = [c for c in neighborhood if c in remaining and c not in subset]
                candidates.sort(key=lambda c: self.dist[(seed, c)])
                subset += candidates[:missing]

            # Final exact ordering
            order, _ = self.best_order_for_subset(subset)
            routes.append(order)

            for c in order:
                remaining.remove(c)

        return routes

    # =========================================================
    # Optional classical cleanup
    # =========================================================
    def two_opt(self, route: List[int]) -> List[int]:
        best = route[:]
        improved = True
        while improved:
            improved = False
            base = self.route_cost(best)
            n = len(best)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    cand = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    c = self.route_cost(cand)
                    if c + 1e-9 < base:
                        best = cand
                        base = c
                        improved = True
        return best

    def improve_routes(self, routes: List[List[int]], passes: int = 10) -> List[List[int]]:
        routes = [self.two_opt(r) for r in routes]

        for _ in range(passes):
            changed = False
            for a in range(len(routes)):
                for b in range(a + 1, len(routes)):
                    ra, rb = routes[a], routes[b]
                    base = self.route_cost(ra) + self.route_cost(rb)
                    best_pair = None
                    best_delta = 0.0

                    for ia, ca in enumerate(ra):
                        for ib, cb in enumerate(rb):
                            na = ra[:]
                            nb = rb[:]
                            na[ia], nb[ib] = cb, ca
                            na, cna = self.best_order_for_subset(na)
                            nb, cnb = self.best_order_for_subset(nb)
                            new_cost = cna + cnb
                            delta = base - new_cost
                            if delta > best_delta + 1e-9:
                                best_delta = delta
                                best_pair = (na, nb)

                    if best_pair is not None:
                        routes[a], routes[b] = best_pair
                        changed = True

            routes = [self.two_opt(r) for r in routes]
            if not changed:
                break

        return routes

    # =========================================================
    # Solve
    # =========================================================
    def solve(self, do_classical_cleanup: bool = True, verbose: bool = True):
        routes = self.build_solution()
        init_cost = sum(self.route_cost(r) for r in routes)

        if do_classical_cleanup:
            routes = self.improve_routes(routes)

        final_cost = sum(self.route_cost(r) for r in routes)

        if verbose:
            print("Quantum DQI-style CVRP solution")
            print(f"Neighborhood qubits per quantum call: {self.q}")
            print(f"Shots per quantum call: {self.shots}")
            print()
            for k, r in enumerate(routes, start=1):
                print(f"Vehicle {k}: [0, " + ", ".join(map(str, r)) + ", 0] "
                      f"load={len(r)} cost={self.route_cost(r):.4f}")
            print(f"\nInitial total cost: {init_cost:.4f}")
            print(f"Final total cost:   {final_cost:.4f}")

        return routes, final_cost


# ============================================================
# Example instance
# ============================================================
if __name__ == "__main__":
    customers = {
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
        20: (0, -2)
    }

    solver = QiskitDQICVRP(
        customers=customers,
        Nv=5,
        C=4,
        depot=(0, 0),
        neighborhood_qubits=6,   # real quantum register size per pricing call
        shots=2048,
        alpha=3.0,
        seed=0,
    )

    routes, total_cost = solver.solve(do_classical_cleanup=True, verbose=True)