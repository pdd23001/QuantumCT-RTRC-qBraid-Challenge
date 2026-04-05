import math
import itertools
from typing import Dict, Tuple, List, Set, Optional

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

try:
    from qiskit_aer import Aer
    HAS_AER = True
except Exception:
    HAS_AER = False


class QiskitDQICVRP:
    """
    DQI (Decoded Quantum Interferometry) based CVRP solver.

    Core DQI loop per route:
      1. Express subset selection as a QUBO using Clarke-Wright savings.
      2. Build a poly-depth quantum circuit:
              |0>^q -> H^q -> PhaseOracle(gamma) -> H^q  (x layers)
         The phase oracle encodes e^{i*gamma*f(x)} via Rz and CNOT+Rz+CNOT
         gadgets — O(q^2) gates total, no classical 2^q enumeration.
      3. Sample from the circuit (shots measurements).
      4. Classical decoding: score every distinct bitstring with the full
         subset_score function, return the best feasible subset.

    The critical difference from the previous version:
      - Old: classically compute all 2^q amplitudes, load with `initialize`
             (O(2^q) gates, eliminates any quantum advantage)
      - New: encode the objective as a shallow quantum circuit (O(q^2) gates),
             then decode classically — the genuine DQI structure
    """

    def __init__(
        self,
        customers: Dict[int, Tuple[float, float]],
        Nv: int,
        C: int,
        depot: Tuple[float, float] = (0.0, 0.0),
        neighborhood_qubits: int = 6,
        shots: int = 2048,
        gamma: float = 0.8,
        layers: int = 1,
        seed: int = 0,
    ):
        self.customers = customers
        self.Nv = Nv
        self.C = C
        self.depot = depot
        self.q = neighborhood_qubits
        self.shots = shots
        self.gamma = gamma
        self.layers = layers
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
    # Exact ordering for a chosen subset: Held-Karp TSP
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
    # Full scoring function (used for classical decoding only)
    # =========================================================
    def subset_score(self, subset: List[int]) -> float:
        if len(subset) == 0:
            return -1e9
        if len(subset) > self.C:
            return -1e9

        _, route_c = self.best_order_for_subset(subset)

        pair_dispersion = sum(
            self.dist[(i, j)] for i, j in itertools.combinations(subset, 2)
        )
        depot_pull = sum(self.dist[(0, i)] for i in subset)

        return (
            8.0 * len(subset)
            - 1.0 * route_c
            - 0.07 * pair_dispersion
            - 0.03 * depot_pull
        )

    # =========================================================
    # QUBO encoding of the subset selection objective
    # =========================================================
    def _build_qubo_coefficients(
        self,
        neighborhood: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode subset scoring as a QUBO: f(x) = sum_i h_i x_i + sum_{i<j} J_ij x_i x_j

        Linear terms h_i:
            8.0 reward for including customer i
            minus depot round-trip cost (approximates individual routing cost)
            plus soft capacity penalty linear part: lambda*(1 - 2*C)

        Quadratic terms J_ij:
            Clarke-Wright savings: dist(0,i) + dist(0,j) - dist(i,j)
            This is the classical routing benefit of combining customers i,j
            in the same vehicle — it's the natural quadratic term for CVRP.
            minus pair dispersion penalty
            minus soft capacity penalty quadratic part: 2*lambda

        The soft capacity penalty is (sum_i x_i - C)^2 expanded in binary:
            = (1 - 2C)*sum_i x_i + 2*sum_{i<j} x_i x_j  + C^2 (constant)
        """
        q = len(neighborhood)
        h = np.zeros(q)
        J = np.zeros((q, q))
        lam = 5.0

        for i, ci in enumerate(neighborhood):
            h[i] = 8.0 - self.dist[(0, ci)]
            h[i] += lam * (1.0 - 2.0 * self.C)

        for i in range(q):
            for j in range(i + 1, q):
                ci, cj = neighborhood[i], neighborhood[j]
                savings = (
                    self.dist[(0, ci)] + self.dist[(0, cj)] - self.dist[(ci, cj)]
                )
                J[i, j] = savings - 0.07 * self.dist[(ci, cj)] - 2.0 * lam

        return h, J

    def _qubo_to_ising(
        self,
        h: np.ndarray,
        J: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Convert QUBO (x_i in {0,1}) to Ising (z_i in {-1,+1}).
        Substitution: x_i = (1 - z_i) / 2

        Returns (offset, h_z, J_z) such that:
            f_qubo(x) = offset + sum_i h_z_i z_i + sum_{i<j} J_z_ij z_i z_j
        """
        q = len(h)
        offset = 0.0
        h_z = np.zeros(q)
        J_z = np.zeros((q, q))

        for i in range(q):
            offset += h[i] / 2.0
            h_z[i] -= h[i] / 2.0

        for i in range(q):
            for j in range(i + 1, q):
                if abs(J[i, j]) < 1e-15:
                    continue
                offset += J[i, j] / 4.0
                h_z[i] -= J[i, j] / 4.0
                h_z[j] -= J[i, j] / 4.0
                J_z[i, j] += J[i, j] / 4.0

        return offset, h_z, J_z

    # =========================================================
    # DQI circuit: poly-depth, no classical 2^q enumeration
    # =========================================================
    def _build_dqi_circuit(self, neighborhood: List[int]) -> QuantumCircuit:
        """
        Build the DQI circuit for the given neighborhood.

        Structure (one layer):
            |0>^q -> H^q -> PhaseOracle(gamma) -> H^q -> Measure

        Multiple layers alternate Phase -> H without resetting, creating
        deeper interference patterns.

        Phase oracle implements e^{i*gamma*(sum h_z_i Z_i + sum J_z_ij Z_i Z_j)}:

            Single-qubit terms h_z_i Z_i:
                e^{i*phi*Z} = Rz(-2*phi)
                (Qiskit convention: Rz(theta) = diag(e^{-i*theta/2}, e^{i*theta/2}))

            Two-qubit terms J_z_ij Z_i Z_j:
                e^{i*phi*Z_i*Z_j} = CNOT(i,j) . Rz(-2*phi, j) . CNOT(i,j)
                Verified: ZZ eigenvalue is encoded in qubit j after CNOT,
                so the Rz applies the correct phase to all four basis states.

        Total circuit depth: O(q^2) for dense QUBO — poly in q, not 2^q.
        """
        q = len(neighborhood)
        h_q, J_q = self._build_qubo_coefficients(neighborhood)
        _, h_z, J_z = self._qubo_to_ising(h_q, J_q)

        # Normalize Ising coefficients to [-1, 1] so gamma controls the
        # maximum phase rotation directly (gamma=pi/4 => max rotation pi/2)
        scale = max(np.max(np.abs(h_z)), np.max(np.abs(J_z)), 1e-12)
        h_z /= scale
        J_z /= scale

        qc = QuantumCircuit(q, q)
        qc.h(range(q))  # uniform superposition |+>^q

        for _ in range(self.layers):
            # Phase oracle: e^{i*gamma*(sum h_z_i Z_i + sum J_z_ij Z_i Z_j)}
            for i in range(q):
                if abs(h_z[i]) > 1e-12:
                    qc.rz(-2.0 * self.gamma * h_z[i], i)

            for i in range(q):
                for j in range(i + 1, q):
                    if abs(J_z[i, j]) < 1e-12:
                        continue
                    qc.cx(i, j)
                    qc.rz(-2.0 * self.gamma * J_z[i, j], j)
                    qc.cx(i, j)

            # Interference / mixing: Walsh-Hadamard transform H^q
            # This is the "interferometry" step that concentrates amplitude
            # on high-scoring subsets via quantum interference.
            qc.h(range(q))

        qc.measure(range(q), range(q))
        return qc

    # =========================================================
    # Circuit execution helper
    # =========================================================
    def _run_circuit(self, qc: QuantumCircuit) -> dict:
        if HAS_AER:
            backend = Aer.get_backend("aer_simulator")
            tqc = transpile(qc, backend, seed_transpiler=self.seed)
            result = backend.run(
                tqc, shots=self.shots, seed_simulator=self.seed
            ).result()
            return result.get_counts()

        # Fallback: exact statevector simulation
        qc_copy = qc.copy()
        qc_copy.remove_final_measurements(inplace=True)
        sv = Statevector.from_instruction(qc_copy)
        probs = sv.probabilities_dict()
        return {
            k: int(round(v * self.shots))
            for k, v in probs.items()
            if v > 1e-12
        }

    # =========================================================
    # DQI quantum sampling + classical decoding
    # =========================================================
    def quantum_sample_subset(
        self,
        neighborhood: List[int],
        must_include: Optional[int] = None,
    ) -> List[int]:
        """
        Run the DQI circuit and apply classical decoding.

        Classical decoding (the 'Decoded' in DQI):
            Evaluate every distinct measured bitstring using the full
            subset_score function and return the best feasible subset.
            This is fundamentally different from taking the most-frequent
            bitstring — it uses quantum sampling to explore the solution
            space and classical evaluation to select the best candidate.
        """
        qc = self._build_dqi_circuit(neighborhood)
        counts = self._run_circuit(qc)

        if not counts:
            return [must_include] if must_include is not None else []

        best_subset: Optional[List[int]] = None
        best_score = -float("inf")

        for bitstring in counts:
            # Qiskit: bitstring[-1] = qubit 0 (little-endian), so reverse
            bits = bitstring[::-1]
            raw = [neighborhood[idx] for idx, bit in enumerate(bits) if bit == "1"]

            decoded = self._decode(raw, must_include)

            score = self.subset_score(decoded)
            if score > best_score:
                best_score = score
                best_subset = decoded

        return best_subset if best_subset else (
            [must_include] if must_include is not None else []
        )

    def _decode(
        self,
        candidate: List[int],
        must_include: Optional[int],
    ) -> List[int]:
        """
        Classical decoding: repair constraint violations in a measured bitstring.

        1. Enforce must_include
        2. Enforce capacity (|subset| <= C) by removing the customers
           farthest from the seed (must_include), keeping the most
           routing-beneficial cluster.
        """
        candidate = list(candidate)

        if must_include is not None and must_include not in candidate:
            candidate = [must_include] + candidate

        if len(candidate) > self.C:
            if must_include is not None:
                optional = [c for c in candidate if c != must_include]
                optional.sort(key=lambda c: self.dist[(must_include, c)])
                candidate = [must_include] + optional[: self.C - 1]
            else:
                candidate = candidate[: self.C]

        return candidate

    # =========================================================
    # Build a full CVRP solution route-by-route
    # =========================================================
    def build_solution(self) -> List[List[int]]:
        remaining = set(self.customer_ids)
        routes = []

        while remaining:
            seed = max(remaining, key=lambda c: self.dist[(0, c)])

            neighborhood = [seed] + self.nearest_neighbors(
                seed, remaining - {seed}, self.q - 1
            )
            neighborhood = list(dict.fromkeys(neighborhood))

            subset = self.quantum_sample_subset(
                neighborhood=neighborhood,
                must_include=seed,
            )
            subset = [c for c in subset if c in remaining]

            if len(subset) < min(self.C, len(remaining)):
                missing = min(self.C, len(remaining)) - len(subset)
                candidates = [
                    c for c in neighborhood if c in remaining and c not in subset
                ]
                candidates.sort(key=lambda c: self.dist[(seed, c)])
                subset += candidates[:missing]

            order, _ = self.best_order_for_subset(subset)
            routes.append(order)
            for c in order:
                remaining.remove(c)

        return routes

    # =========================================================
    # Classical cleanup
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
            print("DQI (Decoded Quantum Interferometry) CVRP solution")
            print(f"Neighborhood qubits per DQI call: {self.q}")
            print(f"DQI layers: {self.layers}  gamma: {self.gamma}")
            print(f"Shots per quantum call: {self.shots}")
            print()
            for k, r in enumerate(routes, start=1):
                print(
                    f"Vehicle {k}: [0, " + ", ".join(map(str, r)) + ", 0] "
                    f"load={len(r)} cost={self.route_cost(r):.4f}"
                )
            print(f"\nInitial total cost: {init_cost:.4f}")
            print(f"Final total cost:   {final_cost:.4f}")

        return routes, final_cost


# ============================================================
# Example instance
# ============================================================
if __name__ == "__main__":
    customers = {
        1:  (-2,  2), 2:  (-5,  8), 3:  ( 2,  3), 4:  ( 5,  7), 5:  ( 2,  4),
        6:  ( 2, -3), 7:  (-4,  1), 8:  ( 0,  6), 9:  ( 3, -2), 10: (-1,  5),
        11: ( 6,  1), 12: (-3,  4), 13: ( 4,  3), 14: (-6,  2), 15: ( 1,  7),
        16: ( 5, -1), 17: (-2, -4), 18: ( 3,  6), 19: (-5,  5), 20: ( 0, -2),
    }

    solver = QiskitDQICVRP(
        customers=customers,
        Nv=5,
        C=4,
        depot=(0, 0),
        neighborhood_qubits=6,
        shots=2048,
        gamma=0.8,
        layers=1,
        seed=0,
    )

    routes, total_cost = solver.solve(do_classical_cleanup=True, verbose=True)
