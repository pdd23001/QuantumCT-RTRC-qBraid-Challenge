"""
quantum_improve.py
==================
Local neighborhood QAOA improvement for individual routes.

Architecture
------------
For each route we:
1. Select a small contiguous neighborhood of k customers.
2. Keep left-anchor and right-anchor nodes fixed (may be depot = 0).
3. Build an ANCHORED position-QUBO on those k customers.
4. Run QAOA (or brute-force for tiny k) and decode the best valid result.
5. Splice improved ordering back into the route only if it reduces cost.



QUBO design (anchored)
-----------------------
- Variables x[i, s] = 1 if the i-th neighborhood customer is at position s.
- n = k customers -> k^2 binary variables.
- Anchor costs:
    * from left_anchor to customer at position 0
    * from customer at position k-1 to right_anchor
- Standard one-hot row / column constraints.
- Objective = total intra-segment + anchor connection distance.

Scaling rule
------------
- max_local_qaoa_qubits (default 25) controls k.
- k = floor(sqrt(max_local_qaoa_qubits)).

Simulator backend
-----------------
- Uses qiskit_aer's AerSimulator (or Aer's StatevectorSimulator) for
  fast statevector simulation when QAOA is triggered.
- Falls back to qiskit.primitives.StatevectorEstimator if Aer unavailable.

Dependencies
------------
- qiskit
- qiskit_aer (recommended for speed)
- scipy.optimize.minimize
- numpy
"""

import time
import math
import itertools
import numpy as np

from common import route_distance

# ---------------------------------------------------------------------------
# Lazy imports: Qiskit and Aer
# ---------------------------------------------------------------------------

_qiskit_available = False
_aer_available = False

try:
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp
    _qiskit_available = True
except ImportError:
    pass

if _qiskit_available:
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
        from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
        _aer_available = True
    except ImportError:
        pass

    if not _aer_available:
        try:
            from qiskit.primitives import StatevectorEstimator, StatevectorSampler
        except ImportError:
            _qiskit_available = False

    try:
        from scipy.optimize import minimize as scipy_minimize
    except ImportError:
        _qiskit_available = False


# ---------------------------------------------------------------------------
# QUBO / Ising helpers (anchored version)
# ---------------------------------------------------------------------------

def _build_anchored_qubo(neighborhood, D, left_anchor, right_anchor):
    """
    Build upper-triangular QUBO for ordering `neighborhood` customers given
    fixed left and right anchor nodes.

    Variables: x[i, s] -- customer i at position s (both 0-indexed).
    n = len(neighborhood) -> n^2 variables; qubit k = i*n + s.
    """
    n = len(neighborhood)
    N = n * n
    Q = np.zeros((N, N))

    def idx(i, s):
        return i * n + s

    dists = [D[a][b]
             for a in [left_anchor] + list(neighborhood)
             for b in [left_anchor, right_anchor] + list(neighborhood)
             if a != b]
    A = (max(dists) if dists else 1.0) * n * 2.0

    # Row constraint: each customer in exactly one position
    for i in range(n):
        for s in range(n):
            Q[idx(i, s), idx(i, s)] -= A
        for s in range(n):
            for t in range(s + 1, n):
                Q[idx(i, s), idx(i, t)] += 2 * A

    # Column constraint: each position holds exactly one customer
    for s in range(n):
        for i in range(n):
            Q[idx(i, s), idx(i, s)] -= A
        for i in range(n):
            for j in range(i + 1, n):
                Q[idx(i, s), idx(j, s)] += 2 * A

    # Objective: anchor -> pos 0
    for i, ci in enumerate(neighborhood):
        Q[idx(i, 0), idx(i, 0)] += D[left_anchor][ci]

    # Objective: pos n-1 -> right_anchor
    for i, ci in enumerate(neighborhood):
        Q[idx(i, n - 1), idx(i, n - 1)] += D[ci][right_anchor]

    # Objective: intra-segment transitions
    for i, ci in enumerate(neighborhood):
        for j, cj in enumerate(neighborhood):
            if i == j:
                continue
            for s in range(n - 1):
                a_idx = idx(i, s)
                b_idx = idx(j, s + 1)
                lo, hi = min(a_idx, b_idx), max(a_idx, b_idx)
                Q[lo, hi] += D[ci][cj]

    return Q


def _qubo_to_ising(Q):
    """Convert upper-triangular QUBO to Ising SparsePauliOp."""
    n = Q.shape[0]
    constant = 0.0
    h = np.zeros(n)
    J = {}

    for k in range(n):
        constant += Q[k, k] / 2.0
        h[k] -= Q[k, k] / 2.0

    for k in range(n):
        for l in range(k + 1, n):
            if abs(Q[k, l]) > 1e-12:
                constant += Q[k, l] / 4.0
                h[k] -= Q[k, l] / 4.0
                h[l] -= Q[k, l] / 4.0
                J[(k, l)] = Q[k, l] / 4.0

    pauli_list = []
    for k in range(n):
        if abs(h[k]) > 1e-12:
            lbl = ["I"] * n
            lbl[n - 1 - k] = "Z"
            pauli_list.append(("".join(lbl), h[k]))

    for (k, l), coef in J.items():
        if abs(coef) > 1e-12:
            lbl = ["I"] * n
            lbl[n - 1 - k] = "Z"
            lbl[n - 1 - l] = "Z"
            pauli_list.append(("".join(lbl), coef))

    if not pauli_list:
        pauli_list = [("I" * n, 0.0)]

    return SparsePauliOp.from_list(pauli_list).simplify(), constant


def _decode_bitstring(bitstring, n, neighborhood, D, left_anchor, right_anchor):
    """
    Decode a QAOA measurement bitstring into an ordered customer list.
    Returns (ordered_customers, segment_cost) or (None, inf) on violation.
    """
    bits = [int(b) for b in reversed(bitstring)]
    while len(bits) < n * n:
        bits.append(0)

    x = np.zeros((n, n), dtype=int)
    for i in range(n):
        for s in range(n):
            x[i, s] = bits[i * n + s]

    if not all(x[i, :].sum() == 1 for i in range(n)):
        return None, float("inf")
    if not all(x[:, s].sum() == 1 for s in range(n)):
        return None, float("inf")

    order = [None] * n
    for i in range(n):
        s = int(np.argmax(x[i, :]))
        order[s] = neighborhood[i]

    cost = D[left_anchor][order[0]]
    for k in range(n - 1):
        cost += D[order[k]][order[k + 1]]
    cost += D[order[-1]][right_anchor]

    return order, cost


# ---------------------------------------------------------------------------
# Neighborhood selection
# ---------------------------------------------------------------------------

def _worst_edges_neighborhood(route, D, k):
    """
    Select a contiguous window of k customers centred on the worst-cost
    contiguous segment.

    Returns (start_idx, end_idx) as slice into `route`.
    """
    n = len(route)
    if n <= k:
        return 0, n

    full = [0] + route + [0]
    # Sum of k consecutive interior edge costs as sliding window
    best_start = 0
    best_cost = -1.0
    for start in range(n - k + 1):
        # window covers full[start+1 .. start+k+1], i.e., k+1 edges
        win_cost = sum(D[full[start + t]][full[start + t + 1]] for t in range(k + 1))
        if win_cost > best_cost:
            best_cost = win_cost
            best_start = start

    end = best_start + k
    return best_start, end


def _random_neighborhood(route, k, rng):
    """Return a random contiguous neighborhood of exactly k customers."""
    n = len(route)
    if n <= k:
        return 0, n
    start = rng.randint(0, n - k)
    return start, start + k





# ---------------------------------------------------------------------------
# QAOA optimizer (only called for k >= 7 with Aer)
# ---------------------------------------------------------------------------

def _get_estimator_and_sampler():
    """Return (estimator, sampler) using Aer if available, else statevector."""
    if _aer_available:
        estimator = AerEstimatorV2()
        sampler = AerSamplerV2()
        return estimator, sampler
    else:
        from qiskit.primitives import StatevectorEstimator, StatevectorSampler  # noqa
        return StatevectorEstimator(), StatevectorSampler()


def _qaoa_optimize_neighborhood(neighborhood, D, left_anchor, right_anchor,
                                 reps=2, shots=2048, restarts=2,
                                 cobyla_maxiter=150, seed=None):
    """
    Run QAOA on the ordering of `neighborhood` customers with fixed anchors.
    Only called when len(neighborhood) >= 7 (otherwise brute force is used).

    Returns
    -------
    (best_order, best_cost, n_qubits, n_gates, elapsed_s)
    """
    n = len(neighborhood)
    t0 = time.time()

    Q = _build_anchored_qubo(neighborhood, D, left_anchor, right_anchor)
    H, _offset = _qubo_to_ising(Q)

    ansatz = QAOAAnsatz(H, reps=reps)
    from qiskit import transpile
    if _aer_available:
        from qiskit_aer import AerSimulator
        ansatz = transpile(ansatz, backend=AerSimulator(), optimization_level=1)
    else:
        ansatz = transpile(ansatz, basis_gates=['rx', 'ry', 'rz', 'cx', 'u1', 'u2', 'u3', 'p', 'u'], optimization_level=1)
    estimator, sampler = _get_estimator_and_sampler()

    rng = np.random.default_rng(seed)

    # Cost function wraps estimator call
    if _aer_available:
        def cost_fn(params):
            job = estimator.run([(ansatz, H, params)])
            result = job.result()
            return float(result[0].data.evs)
    else:
        def cost_fn(params):
            result = estimator.run([(ansatz, H, params)]).result()
            return float(result[0].data.evs)

    best_params = None
    best_energy = float("inf")
    for _ in range(restarts):
        x0 = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
        res = scipy_minimize(cost_fn, x0, method="COBYLA",
                             options={"maxiter": cobyla_maxiter})
        if res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x

    # Sample the optimized circuit
    ansatz_meas = ansatz.measure_all(inplace=False)
    bound = ansatz_meas.assign_parameters(best_params)

    if _aer_available:
        counts = sampler.run([bound], shots=shots).result()[0].data.meas.get_counts()
    else:
        counts = sampler.run([bound], shots=shots).result()[0].data.meas.get_counts()

    best_order = None
    best_cost = float("inf")
    for bs in counts:
        order, cost = _decode_bitstring(bs, n, neighborhood, D, left_anchor, right_anchor)
        if order is not None and cost < best_cost:
            best_cost = cost
            best_order = order



    elapsed = time.time() - t0
    n_qubits = ansatz.num_qubits
    n_gates = sum(ansatz.decompose().count_ops().values())

    return best_order, best_cost, n_qubits, n_gates, elapsed


# ---------------------------------------------------------------------------
# Public interface: improve one route
# ---------------------------------------------------------------------------

def improve_route_qaoa(route, D,
                       max_local_qaoa_qubits=25,
                       strategy="worst_edges",
                       reps=2,
                       shots=2048,
                       restarts=2,
                       cobyla_maxiter=150,
                       seed=None,
                       rng=None):
    """
    Attempt to improve a single route using anchored local optimization.

    This purely uses QAOA simulation for any neighborhood size up to the qubit budget.

    Parameters
    ----------
    route                  : list[int]  -- customer IDs, no depot
    D                      : list[list[float]]
    max_local_qaoa_qubits  : int        -- qubit budget; k = floor(sqrt(budget))
    strategy               : str        -- "worst_edges" or "random"
    reps                   : int        -- QAOA layers
    shots                  : int
    restarts               : int        -- random restarts for COBYLA
    cobyla_maxiter         : int        -- COBYLA iterations per restart
    seed                   : int or None
    rng                    : random.Random instance (for "random" strategy)
    brute_force_threshold  : int        -- k <= this uses brute force (default 6)

    Returns
    -------
    improved_route  : list[int]
    gain            : float   -- distance reduction (>= 0)
    meta            : dict
    """
    if not _qiskit_available and not True:  # classical path always available
        pass

    if len(route) < 2:
        return route[:], 0.0, {"skipped": "route_too_short"}

    k = max(2, int(math.isqrt(max_local_qaoa_qubits)))
    k = min(k, len(route))

    if strategy == "worst_edges":
        start, end = _worst_edges_neighborhood(route, D, k)
    else:
        import random as _random
        _rng = rng if rng is not None else _random.Random(seed)
        start, end = _random_neighborhood(route, k, _rng)

    neighborhood = route[start:end]
    left_anchor = route[start - 1] if start > 0 else 0
    right_anchor = route[end] if end < len(route) else 0

    # Current cost of the segment (including anchor connections)
    seg_cost_before = D[left_anchor][neighborhood[0]]
    for i in range(len(neighborhood) - 1):
        seg_cost_before += D[neighborhood[i]][neighborhood[i + 1]]
    seg_cost_before += D[neighborhood[-1]][right_anchor]

    if not _qiskit_available:
        return route[:], 0.0, {"skipped": "qiskit_not_available"}
    best_order, best_cost, n_qubits, n_gates, elapsed = _qaoa_optimize_neighborhood(
        neighborhood, D, left_anchor, right_anchor,
        reps=reps, shots=shots, restarts=restarts,
        cobyla_maxiter=cobyla_maxiter, seed=seed,
    )
    meta = {
        "n_qubits": n_qubits,
        "n_gates": n_gates,
        "elapsed_s": round(elapsed, 3),
        "neighborhood_size": k,
        "strategy": strategy,
        "qaoa_reps": reps,
        "method": "qaoa",
    }

    if best_order is None:
        return route[:], 0.0, {**meta, "outcome": "no_valid_result"}

    gain = seg_cost_before - best_cost
    if gain <= 1e-10:
        return route[:], 0.0, {**meta, "outcome": "no_improvement"}

    new_route = route[:start] + best_order + route[end:]
    meta["outcome"] = "improved"
    return new_route, round(gain, 6), meta


# ---------------------------------------------------------------------------
# Apply to all routes
# ---------------------------------------------------------------------------

def improve_all_routes_qaoa(routes, D,
                             max_local_qaoa_qubits=25,
                             strategy="worst_edges",
                             reps=2,
                             shots=2048,
                             restarts=2,
                             cobyla_maxiter=150,
                             seed=None):
    """
    Apply improve_route_qaoa to every route.

    Returns
    -------
    improved_routes : list[list[int]]
    total_gain      : float
    all_meta        : list[dict]
    """
    improved_routes = []
    total_gain = 0.0
    all_meta = []

    for i, r in enumerate(routes):
        r_seed = seed + i if seed is not None else None
        new_r, gain, meta = improve_route_qaoa(
            r, D,
            max_local_qaoa_qubits=max_local_qaoa_qubits,
            strategy=strategy,
            reps=reps,
            shots=shots,
            restarts=restarts,
            cobyla_maxiter=cobyla_maxiter,
            seed=r_seed,
        )
        improved_routes.append(new_r)
        total_gain += gain
        all_meta.append(meta)

    return improved_routes, round(total_gain, 6), all_meta


# ---------------------------------------------------------------------------
# Standalone QAOA demo for one neighborhood (notebook helper)
# ---------------------------------------------------------------------------

def demo_qaoa_neighborhood(neighborhood, D, left_anchor, right_anchor,
                            reps=2, shots=2048, restarts=2, seed=42):
    """
    Run QAOA on a single neighborhood and return full diagnostics.
    Intended for notebook demonstration of the quantum subroutine.

    Returns
    -------
    dict with keys: order, cost, n_qubits, n_gates, elapsed_s, qubo_size
    """
    n = len(neighborhood)
    t0 = time.time()
    best_order, best_cost, n_qubits, n_gates, elapsed = _qaoa_optimize_neighborhood(
        neighborhood, D, left_anchor, right_anchor,
        reps=reps, shots=shots, restarts=restarts, seed=seed,
    )
    return {
        "order": best_order,
        "cost": best_cost,
        "n_qubits": n_qubits,
        "n_gates": n_gates,
        "elapsed_s": elapsed,
        "qubo_size": n * n,
    }
