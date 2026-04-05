#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import math
import time
import warnings
warnings.filterwarnings('ignore')
from itertools import permutations

from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimator, SamplerV2 as AerSampler
from scipy.optimize import minimize


# ── Utility Functions ──

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def build_distance_matrix(nodes):
    n = len(nodes)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i,j] = euclidean_distance(nodes[i], nodes[j])
    return D


# In[63]:


# ── Phase 1: Sweep Algorithm (Classical Clustering) ──

def sweep_decomposition(nodes, Nv, C, n_angles=16):
    """Cluster customers by polar angle from depot, trying multiple starting angles.
    Picks the best rotation using nearest-neighbor TSP estimate (scalable)."""
    depot = nodes[0]
    D_local = build_distance_matrix(nodes)

    # Precompute polar angles for all customers
    customer_angles = []
    for i in range(1, len(nodes)):
        angle = math.atan2(nodes[i][1] - depot[1], nodes[i][0] - depot[0])
        customer_angles.append((i, angle))

    best_clusters = None
    best_cost = float('inf')

    for k in range(n_angles):
        offset = 2 * math.pi * k / n_angles

        # Sort by shifted angle
        shifted = sorted(customer_angles,
                         key=lambda x: (x[1] - offset) % (2 * math.pi))

        # Pack into clusters respecting capacity
        clusters = []
        for start in range(0, len(shifted), C):
            if len(clusters) < Nv:
                chunk = shifted[start : start + C]
                clusters.append([c[0] for c in chunk])

        # Evaluate with nearest-neighbor TSP heuristic — O(n²) per cluster
        cost = 0
        for cluster in clusters:
            if len(cluster) == 1:
                cost += 2 * D_local[0, cluster[0]]
            else:
                visited = []
                current = 0  # start at depot
                remaining = set(cluster)
                while remaining:
                    nearest = min(remaining, key=lambda c: D_local[current, c])
                    cost += D_local[current, nearest]
                    current = nearest
                    remaining.remove(nearest)
                cost += D_local[current, 0]  # return to depot

        if cost < best_cost:
            best_cost = cost
            best_clusters = clusters

    return best_clusters

def clarke_wright_clusters(nodes, Nv, C):
    """Cluster customers using Clarke & Wright savings algorithm.
    Returns list of clusters (same format as sweep_decomposition)."""
    n = len(nodes)
    D = build_distance_matrix(nodes)

    # 1. Start: every customer gets its own route
    routes = [[i] for i in range(1, n)]

    # 2. Compute savings for all pairs: s(i,j) = D[0,i] + D[0,j] - D[i,j]
    savings = []
    for i in range(1, n):
        for j in range(i + 1, n):
            s = D[0, i] + D[0, j] - D[i, j]
            if s > 0:
                savings.append((s, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])

    # 3. Greedily merge routes by highest savings
    def find_route(cust):
        for idx, r in enumerate(routes):
            if cust in r:
                return idx
        return -1

    for s, i, j in savings:
        ri, rj = find_route(i), find_route(j)

        # Must be different routes
        if ri == rj:
            continue

        # Both must be at the edge (first or last) of their route
        r_i, r_j = routes[ri], routes[rj]
        if not (r_i[0] == i or r_i[-1] == i):
            continue
        if not (r_j[0] == j or r_j[-1] == j):
            continue

        # Capacity check
        if len(r_i) + len(r_j) > C:
            continue

        # Orient and merge: make i the tail of r_i, j the head of r_j
        if r_i[-1] == i and r_j[0] == j:
            merged = r_i + r_j
        elif r_i[-1] == i and r_j[-1] == j:
            merged = r_i + r_j[::-1]
        elif r_i[0] == i and r_j[0] == j:
            merged = r_i[::-1] + r_j
        elif r_i[0] == i and r_j[-1] == j:
            merged = r_j + r_i
        else:
            continue

        # Apply merge
        for idx in sorted([ri, rj], reverse=True):
            routes.pop(idx)
        routes.append(merged)

    # 4. If too many routes remain, force-merge the cheapest pairs
    while len(routes) > Nv:
        best_cost, best_pair = float('inf'), None
        for a in range(len(routes)):
            for b in range(a + 1, len(routes)):
                if len(routes[a]) + len(routes[b]) <= C:
                    cost = D[routes[a][-1], routes[b][0]] - D[routes[a][-1], 0] - D[0, routes[b][0]]
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (a, b)
        if best_pair:
            a, b = best_pair
            routes[a] = routes[a] + routes[b]
            routes.pop(b)
        else:
            break

    return routes

def agglomerative_clusters(nodes, Nv, C):
    """Capacitated agglomerative clustering — merges by route cost reduction."""
    n = len(nodes)
    D = build_distance_matrix(nodes)
    clusters = [[i] for i in range(1, n)]

    def nn_cost(cluster):
        """Nearest-neighbor tour cost: depot → cluster → depot."""
        if len(cluster) == 1:
            return 2 * D[0, cluster[0]]
        cost, current, remaining = 0, 0, set(cluster)
        while remaining:
            nxt = min(remaining, key=lambda c: D[current, c])
            cost += D[current, nxt]
            current = nxt
            remaining.remove(nxt)
        return cost + D[current, 0]

    while len(clusters) > 1:
        best_benefit, best_pair = -float('inf'), None

        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if len(clusters[a]) + len(clusters[b]) > C:
                    continue
                benefit = nn_cost(clusters[a]) + nn_cost(clusters[b]) - nn_cost(clusters[a] + clusters[b])
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_pair = (a, b)

        if best_pair is None:
            break  # No valid merges (capacity)
        if best_benefit <= 0 and len(clusters) <= Nv:
            break  # No benefit and within vehicle limit

        a, b = best_pair
        clusters[a] = clusters[a] + clusters[b]
        clusters.pop(b)

    return clusters


# In[64]:


def nn_route_cost(cluster, nodes):
    """Nearest-neighbor route cost for a cluster (depot=0)."""
    if not cluster:
        return 0
    route = [0]
    remaining = list(cluster)
    while remaining:
        last = route[-1]
        nearest = min(remaining, key=lambda c: math.sqrt(
            (nodes[last][0]-nodes[c][0])**2 + (nodes[last][1]-nodes[c][1])**2))
        route.append(nearest)
        remaining.remove(nearest)
    # Return to depot
    cost = sum(math.sqrt((nodes[route[i]][0]-nodes[route[i+1]][0])**2 +
                          (nodes[route[i]][1]-nodes[route[i+1]][1])**2)
               for i in range(len(route)-1))
    cost += math.sqrt((nodes[route[-1]][0]-nodes[0][0])**2 +
                       (nodes[route[-1]][1]-nodes[0][1])**2)
    return cost

def refine_clusters_swap(clusters, nodes, C):
    """Improve clusters via inter-cluster pairwise swaps."""
    clusters = [list(c) for c in clusters]
    improved = True
    iteration = 0

    while improved:
        improved = False
        iteration += 1

        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                old_cost = nn_route_cost(clusters[i], nodes) + nn_route_cost(clusters[j], nodes)

                # Try swaps: exchange customer a from cluster i with customer b from cluster j
                best_gain = 0
                best_swap = None

                for ai, a in enumerate(clusters[i]):
                    for bi, b in enumerate(clusters[j]):
                        # Swap
                        ci_new = clusters[i][:ai] + [b] + clusters[i][ai+1:]
                        cj_new = clusters[j][:bi] + [a] + clusters[j][bi+1:]

                        if len(ci_new) > C or len(cj_new) > C:
                            continue

                        new_cost = nn_route_cost(ci_new, nodes) + nn_route_cost(cj_new, nodes)
                        gain = old_cost - new_cost

                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (ai, bi, ci_new, cj_new)

                # Try moves: move customer from larger to smaller cluster
                for ai, a in enumerate(clusters[i]):
                    if len(clusters[j]) < C:
                        ci_new = clusters[i][:ai] + clusters[i][ai+1:]
                        cj_new = clusters[j] + [a]
                        if ci_new:  # don't empty a cluster
                            new_cost = nn_route_cost(ci_new, nodes) + nn_route_cost(cj_new, nodes)
                            gain = old_cost - new_cost
                            if gain > best_gain:
                                best_gain = gain
                                best_swap = (-1, -1, ci_new, cj_new)

                for bi, b in enumerate(clusters[j]):
                    if len(clusters[i]) < C:
                        cj_new = clusters[j][:bi] + clusters[j][bi+1:]
                        ci_new = clusters[i] + [b]
                        if cj_new:
                            new_cost = nn_route_cost(ci_new, nodes) + nn_route_cost(cj_new, nodes)
                            gain = old_cost - new_cost
                            if gain > best_gain:
                                best_gain = gain
                                best_swap = (-1, -1, ci_new, cj_new)

                if best_swap and best_gain > 1e-10:
                    clusters[i] = best_swap[2]
                    clusters[j] = best_swap[3]
                    improved = True
                    print(f"  Swap iter {iteration}: clusters {i},{j} improved by {best_gain:.2f}")

    print(f"  Swap refinement done in {iteration} iterations")
    return clusters


# In[65]:


# ── Phase 2: TSP via QAOA (Quantum Routing) ──

def build_tsp_qubo(cluster_indices, dist_matrix):
    """Build upper-triangular QUBO for TSP (position formulation).
    Variables x[i,s] = 1 if cluster node i is at route position s.
    n customers → n² binary variables / qubits."""
    n = len(cluster_indices)
    N = n * n
    Q = np.zeros((N, N))
    idx = lambda i, s: i * n + s

    # Penalty coefficient (must dominate route costs)
    max_d = max(dist_matrix[a, b]
                for a in [0] + cluster_indices
                for b in [0] + cluster_indices if a != b)
    A = max_d * n * 1.5

    # Constraint: each customer visited exactly once  Σ_s x[i,s] = 1
    for i in range(n):
        for s in range(n):
            Q[idx(i,s), idx(i,s)] -= A
        for s in range(n):
            for t in range(s+1, n):
                Q[idx(i,s), idx(i,t)] += 2 * A

    # Constraint: each position filled by exactly one customer  Σ_i x[i,s] = 1
    for s in range(n):
        for i in range(n):
            Q[idx(i,s), idx(i,s)] -= A
        for i in range(n):
            for j in range(i+1, n):
                Q[idx(i,s), idx(j,s)] += 2 * A

    # Objective: route distance (depot → pos 0, transitions, pos n-1 → depot)
    for i, ci in enumerate(cluster_indices):
        Q[idx(i,0), idx(i,0)]     += dist_matrix[0, ci]   # depot → first
        Q[idx(i,n-1), idx(i,n-1)] += dist_matrix[ci, 0]   # last → depot
        for j, cj in enumerate(cluster_indices):
            if i != j:
                for s in range(n - 1):
                    a, b = idx(i, s), idx(j, s+1)
                    if a <= b:
                        Q[a, b] += dist_matrix[ci, cj]
                    else:
                        Q[b, a] += dist_matrix[ci, cj]
    return Q


def qubo_to_ising(Q):
    """Convert upper-triangular QUBO matrix to a SparsePauliOp Ising Hamiltonian.
    x_k = (1 - Z_k) / 2"""
    n = Q.shape[0]
    constant = 0.0
    h = np.zeros(n)
    J = {}

    for k in range(n):
        constant += Q[k, k] / 2.0
        h[k]     -= Q[k, k] / 2.0

    for k in range(n):
        for l in range(k+1, n):
            if abs(Q[k, l]) > 1e-12:
                constant += Q[k, l] / 4.0
                h[k]     -= Q[k, l] / 4.0
                h[l]     -= Q[k, l] / 4.0
                J[(k, l)] = Q[k, l] / 4.0

    # Build Pauli list  (qiskit little-endian: rightmost char = qubit 0)
    pauli_list = []
    for k in range(n):
        if abs(h[k]) > 1e-12:
            lbl = ['I'] * n
            lbl[n - 1 - k] = 'Z'
            pauli_list.append((''.join(lbl), h[k]))

    for (k, l), coef in J.items():
        if abs(coef) > 1e-12:
            lbl = ['I'] * n
            lbl[n - 1 - k] = 'Z'
            lbl[n - 1 - l] = 'Z'
            pauli_list.append((''.join(lbl), coef))

    if not pauli_list:
        pauli_list = [('I' * n, 0.0)]

    return SparsePauliOp.from_list(pauli_list).simplify(), constant


def decode_bitstring(bitstring, n, cluster_indices, dist_matrix):
    """Decode a measurement bitstring → (ordered_customer_ids, route_distance).
    Returns (None, inf) when the bitstring violates a constraint."""
    bits = [int(b) for b in reversed(bitstring)]
    while len(bits) < n * n:
        bits.append(0)

    x = np.zeros((n, n), dtype=int)
    for i in range(n):
        for s in range(n):
            x[i, s] = bits[i * n + s]

    if not all(x[i, :].sum() == 1 for i in range(n)):
        return None, float('inf')
    if not all(x[:, s].sum() == 1 for s in range(n)):
        return None, float('inf')

    order = [0] * n
    for i in range(n):
        order[int(np.argmax(x[i, :]))] = cluster_indices[i]

    d = dist_matrix[0, order[0]]
    for k in range(n - 1):
        d += dist_matrix[order[k], order[k+1]]
    d += dist_matrix[order[-1], 0]
    return order, d


from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import SamplerV2 as AerSampler
from scipy.optimize import minimize
import numpy as np
import time
from itertools import permutations

# ============================================================
# DQI SOLVER — drop-in replacement for solve_tsp_qaoa
# Same inputs, same outputs — swap at call site trivially
# ============================================================

def build_dqi_circuit(H_pauli, n_qubits, reps=2):
    """
    Build DQI circuit for a given cost Hamiltonian.

    Structure per layer:
      - Phase separation: e^{iγH} — same as QAOA, encodes cost
      - Decoder block: structured single-qubit + entangling rotations
        designed to amplify feasible low-cost solutions

    Key difference from QAOA: the mixer is NOT uniform Rx rotations.
    Instead we use a layered structure of Ry + CNOT that creates
    interference patterns aligned with the constraint structure.
    """
    # Parameters: one gamma per layer (phase), 
    # one theta per qubit per layer (decoder)
    gammas = ParameterVector('γ', reps)
    thetas = ParameterVector('θ', reps * n_qubits)

    qc = QuantumCircuit(n_qubits)

    # Initial state: uniform superposition (same as QAOA)
    qc.h(range(n_qubits))

    for layer in range(reps):
        # ── Phase separation ──────────────────────────────
        # e^{iγH}: for each ZZ term apply Rzz, for each Z term apply Rz
        gamma = gammas[layer]
        for pauli_str, coeff in H_pauli.to_list():
            qubits_involved = [
                n_qubits - 1 - i
                for i, c in enumerate(pauli_str) if c == 'Z'
            ]
            if len(qubits_involved) == 1:
                qc.rz(2 * gamma * coeff, qubits_involved[0])
            elif len(qubits_involved) == 2:
                q0, q1 = qubits_involved
                qc.cx(q0, q1)
                qc.rz(2 * gamma * coeff, q1)
                qc.cx(q0, q1)

        # ── Decoder block ─────────────────────────────────
        # This is what makes DQI different from QAOA.
        # Instead of uniform Rx mixer, we use:
        # 1. Per-qubit Ry rotations (learnable, breaks symmetry)
        # 2. Nearest-neighbor CNOT chain (creates entanglement
        #    that encodes constraint structure)
        # 3. Another round of Ry (allows interference to develop)
        for q in range(n_qubits):
            theta = thetas[layer * n_qubits + q]
            qc.ry(theta, q)

        # Entangling layer — CNOT chain couples adjacent qubits
        # In position encoding x[i,s], adjacent qubits share
        # either the same customer or the same position
        # This coupling creates the interference that amplifies
        # valid assignments (where each row/col sums to 1)
        for q in range(0, n_qubits - 1, 2):
            qc.cx(q, q + 1)
        for q in range(1, n_qubits - 1, 2):
            qc.cx(q, q + 1)

        # Second Ry after entanglement
        for q in range(n_qubits):
            theta = thetas[layer * n_qubits + q]
            qc.ry(theta / 2, q)

    qc.measure_all()
    return qc, gammas, thetas


def solve_tsp_dqi(cluster_indices, dist_matrix, reps=2, shots=8192, restarts=5):
    """
    DQI solver — same interface as solve_tsp_qaoa.
    Returns (route, distance, n_qubits, n_gates, exec_time, best_params)

    Drop-in replacement: change solve_tsp_qaoa → solve_tsp_dqi
    in Cell 5 of your teammate's notebook.
    """
    n = len(cluster_indices)

    # Trivial case
    if n == 1:
        d = dist_matrix[0, cluster_indices[0]] * 2
        return list(cluster_indices), d, 0, 0, 0.0, None

    t0 = time.time()

    # Build QUBO and convert to Ising Hamiltonian
    # (reuse your teammate's existing functions)
    Q = build_tsp_qubo(cluster_indices, dist_matrix)
    H, offset = qubo_to_ising(Q)
    n_qubits = Q.shape[0]

    # Build DQI circuit
    qc, gammas, thetas = build_dqi_circuit(H, n_qubits, reps=reps)
    n_params = reps + reps * n_qubits  # gammas + thetas

    sampler = AerSampler()

    def dqi_cost(params):
        """
        Cost function for DQI parameter optimization.
        Unlike QAOA which minimizes expectation value of H,
        DQI minimizes the expected route distance directly
        from sampled bitstrings — more direct signal.
        """
        gamma_vals = params[:reps]
        theta_vals = params[reps:]

        # Bind parameters
        param_dict = {}
        for i, g in enumerate(gammas):
            param_dict[g] = gamma_vals[i]
        for i, t in enumerate(thetas):
            param_dict[t] = theta_vals[i]

        bound = qc.assign_parameters(param_dict)
        result = sampler.run([bound], shots=256).result()
        counts = result[0].data.meas.get_counts()

        # Evaluate cost as average distance of valid routes
        total_cost = 0
        total_weight = 0
        for bs, count in counts.items():
            route, d = decode_bitstring(bs, n, cluster_indices, dist_matrix)
            if route is not None:
                total_cost += d * count
                total_weight += count

        # If no valid routes found, return large penalty
        if total_weight == 0:
            return 1e6
        return total_cost / total_weight

    # Multi-restart optimization
    best_params = None
    best_cost = float('inf')

    print(f"    Starting DQI optimization ({restarts} restarts, {n_qubits} qubits)...")

    for r in range(restarts):
        if r == 0:
            # Structured initialization:
            # Small gammas (don't over-rotate phases initially)
            # Thetas near π/4 (balanced superposition in decoder)
            x0 = np.array(
                [0.3] * reps +           # gammas
                [np.pi/4] * (reps * n_qubits)  # thetas
            )
            init_type = "Structured"
        else:
            x0 = np.random.uniform(0, np.pi, n_params)
            init_type = "Random    "

        res = minimize(
            dqi_cost, x0,
            method='COBYLA',
            options={'maxiter': 200, 'rhobeg': 0.5}
        )
        print(f"      Restart {r} [{init_type}]: Cost = {res.fun:.3f} | Iters = {res.nfev}")

        if res.fun < best_cost:
            best_cost = res.fun
            best_params = res.x

    # Final high-shot sample with best parameters
    gamma_vals = best_params[:reps]
    theta_vals = best_params[reps:]
    param_dict = {}
    for i, g in enumerate(gammas):
        param_dict[g] = gamma_vals[i]
    for i, t in enumerate(thetas):
        param_dict[t] = theta_vals[i]

    bound = qc.assign_parameters(param_dict)
    counts = sampler.run([bound], shots=shots).result()[0].data.meas.get_counts()

    # Pick best valid route from all samples
    best_route, best_dist = None, float('inf')
    for bs, _ in sorted(counts.items(), key=lambda x: -x[1]):
        route, d = decode_bitstring(bs, n, cluster_indices, dist_matrix)
        if route is not None and d < best_dist:
            best_dist = d
            best_route = route

    # Brute force fallback if DQI found nothing valid
    valid_count = sum(
        1 for bs in counts
        if decode_bitstring(bs, n, cluster_indices, dist_matrix)[0] is not None
    )
    print(f"    DQI valid routes found: {valid_count} / {len(counts)} unique bitstrings")

    # Brute force fallback if DQI found nothing valid
    if best_route is None:
        print("    Warning: DQI found no valid route, using brute force")
        for perm in permutations(cluster_indices):
            d = dist_matrix[0, perm[0]]
            for k in range(len(perm) - 1):
                d += dist_matrix[perm[k], perm[k+1]]
            d += dist_matrix[perm[-1], 0]
            if d < best_dist:
                best_dist = d
                best_route = list(perm)

    elapsed = time.time() - t0
    decomposed = qc.decompose()
    n_gates = sum(decomposed.count_ops().values())

    return best_route, best_dist, n_qubits, n_gates, elapsed, best_params



# ── Instance Library ──

instances = {
    1: {"Nv": 2, "C": 5, "nodes": [(0,0), (-2,2), (-5,8), (2,3)]},
    2: {"Nv": 2, "C": 2, "nodes": [(0,0), (-2,2), (-5,8), (2,3)]},
    3: {"Nv": 3, "C": 2, "nodes": [(0,0), (-2,2), (-5,8), (2,3), (5,7), (2,4), (2,-3)]},
    4: {"Nv": 4, "C": 3, "nodes": [(0,0), (-2,2), (-5,8), (6,3), (4,4), (3,2),
                                     (0,2), (-2,3), (-4,3), (2,3), (2,7), (-2,5), (-1,4)]},
}


# In[66]:


import json

# ── Configuration ──
read_from_file = True
selected_file_id = "instance_1"   # used when read_from_file = True
selected_id = 3                   # used when read_from_file = False

if read_from_file:
    with open('instances_2.json', 'r') as f:
        all_instances = json.load(f)
    config = next((inst for inst in all_instances if inst['instance_id'] == selected_file_id), None)
    assert config is not None, f"Instance '{selected_file_id}' not found in instances.json"
    Nv = config['Nv']
    C  = config['C']
    nodes = [(c['x'], c['y']) for c in config['customers']]
    selected_id = selected_file_id
else:
    config = instances[selected_id]
    Nv = config['Nv']
    C  = config['C']
    nodes = config['nodes']

print(f"Loaded Instance {selected_id}: {len(nodes)-1} customers, {Nv} vehicles, Capacity {C}, Max qubits: {C**2}")


# In[67]:


# 1. Distance matrix
D = build_distance_matrix(nodes)

# 2. Classical Decomposition
clusters = sweep_decomposition(nodes, Nv, C)

# optional refinement
clusters = refine_clusters_swap(clusters, nodes, C)

# --- FEASIBILITY CHECK (MUST BE AFTER ALL MODIFICATIONS) ---
all_customers = set(range(1, len(nodes)))
assigned = set(c for cluster in clusters for c in cluster)

if assigned != all_customers:
    print("❌ ERROR: Missing or duplicate customers in clustering")
else:
    print("✅ All customers assigned exactly once")
# Capacity safety check — split any oversized clusters
fixed = []
for cluster in clusters:
    if len(cluster) <= C:
        fixed.append(cluster)
    else:
        # Split into chunks of size C
        for i in range(0, len(cluster), C):
            chunk = cluster[i:i+C]
            if chunk:
                fixed.append(chunk)
# Trim back to Nv vehicles by merging smallest if needed
while len(fixed) > Nv:
    fixed.sort(key=len)
    merged = False

    for i in range(len(fixed)-1):
        if len(fixed[i]) + len(fixed[i+1]) <= C:
            fixed[i+1] = fixed[i] + fixed[i+1]
            fixed.pop(i)
            merged = True
            break

    if not merged:
        break  # cannot merge safely → keep extra routes
clusters = fixed
if len(clusters) > Nv:
    print(f"⚠️ Using {len(clusters)} vehicles (limit was {Nv})")
# --- FINAL FEASIBILITY CHECK ---
all_customers = set(range(1, len(nodes)))
assigned = set(c for cluster in clusters for c in cluster)

if assigned != all_customers:
    print("❌ FINAL ERROR: Missing or duplicate customers after fixing/merging")
else:
    print("✅ Final clustering is valid")

print(f"Sweep clusters: {clusters}\n")

# 3. Quantum Routing (QAOA per cluster) with warm-starting
total_distance = 0
routes = []
max_qubits = 0
total_gates = 0
total_time  = 0

for i, cluster in enumerate(clusters):
    route, dist, nq, ng, et, _ = solve_tsp_dqi(
    cluster, D, reps=2, restarts=5, shots=8192
)
    full_route = [0] + route + [0]
    routes.append(full_route)
    total_distance += dist
    max_qubits = max(max_qubits, nq)
    total_gates += ng
    total_time  += et
    print(f"  r{i+1}: {', '.join(map(str, full_route))}  |  Distance: {dist:.2f}  |  Qubits: {nq}")

print(f"\n{'='*40}")
print(f"Instance {selected_id}  —  Total Distance: {total_distance:.2f}")
print(f"Max Qubits: {max_qubits}  |  Total Gates: {total_gates}  |  Time: {total_time:.2f}s")


# In[68]:


# ── Feasibility Check (with Flow Conservation) ──

print("=" * 50)
print("FEASIBILITY CHECK")
print("=" * 50)

all_customers = set(range(1, len(nodes)))
visited = []
feasible = True

# Track edges for flow analysis
in_degree  = {i: 0 for i in range(len(nodes))}
out_degree = {i: 0 for i in range(len(nodes))}

for i, route in enumerate(routes):
    # 1. Route starts and ends at depot
    if route[0] != 0 or route[-1] != 0:
        print(f"  ✗ Route {i+1} does not start/end at depot: {route}")
        feasible = False
    else:
        print(f"  ✓ Route {i+1} starts and ends at depot")

    # 2. Capacity constraint
    customers_in_route = [c for c in route if c != 0]
    if len(customers_in_route) > C:
        print(f"  ✗ Route {i+1} exceeds capacity: {len(customers_in_route)} > {C}")
        feasible = False
    else:
        print(f"  ✓ Route {i+1} capacity OK ({len(customers_in_route)}/{C})")

    # 3. Route continuity — no repeated nodes within a single route
    if len(customers_in_route) != len(set(customers_in_route)):
        print(f"  ✗ Route {i+1} has repeated customers: {route}")
        feasible = False
    else:
        print(f"  ✓ Route {i+1} no internal repeats")

    # Accumulate edges for flow
    for k in range(len(route) - 1):
        out_degree[route[k]]   += 1
        in_degree[route[k+1]]  += 1

    visited.extend(customers_in_route)

# 4. Every customer visited exactly once
visited_set = set(visited)
missing    = all_customers - visited_set
duplicates = [c for c in visited if visited.count(c) > 1]

if missing:
    print(f"  ✗ Missing customers: {missing}")
    feasible = False
else:
    print(f"  ✓ All {len(all_customers)} customers visited")

if duplicates:
    print(f"  ✗ Duplicate visits: {set(duplicates)}")
    feasible = False
else:
    print(f"  ✓ No duplicate visits")

# 5. Number of vehicles
if len(routes) > Nv:
    print(f"  ✗ Too many vehicles: {len(routes)} > {Nv}")
    feasible = False
else:
    print(f"  ✓ Vehicles used: {len(routes)}/{Nv}")

# 6. Flow conservation
print(f"\n{'─'*50}")
print("FLOW CONSERVATION")
print(f"{'─'*50}")

# Depot: out-degree = in-degree = number of routes
depot_out_ok = out_degree[0] == len(routes)
depot_in_ok  = in_degree[0]  == len(routes)
if depot_out_ok and depot_in_ok:
    print(f"  ✓ Depot flow balanced: {out_degree[0]} out, {in_degree[0]} in ({len(routes)} routes)")
else:
    print(f"  ✗ Depot flow imbalanced: {out_degree[0]} out, {in_degree[0]} in (expected {len(routes)})")
    feasible = False

# Customers: in-degree = out-degree = 1
flow_violations = []
for c in all_customers:
    if in_degree[c] != 1 or out_degree[c] != 1:
        flow_violations.append((c, in_degree[c], out_degree[c]))

if not flow_violations:
    print(f"  ✓ All customers: in-degree = out-degree = 1")
else:
    for c, ind, outd in flow_violations:
        print(f"  ✗ Customer {c}: in-degree={ind}, out-degree={outd} (expected 1,1)")
    feasible = False

# 7. Verify total distance
print(f"\n{'─'*50}")
print("DISTANCE VERIFICATION")
print(f"{'─'*50}")
recomputed = 0
for i, route in enumerate(routes):
    rd = sum(D[route[k], route[k+1]] for k in range(len(route) - 1))
    recomputed += rd
    print(f"  → Route {i+1}: {' → '.join(map(str, route))}  dist={rd:.2f}")

dist_match = abs(recomputed - total_distance) < 1e-6
if dist_match:
    print(f"  ✓ Total distance verified: {recomputed:.2f}")
else:
    print(f"  ✗ Distance mismatch: reported={total_distance:.2f}, recomputed={recomputed:.2f}")
    feasible = False

print(f"\n{'='*50}")
print(f"{'✓ SOLUTION FEASIBLE' if feasible else '✗ SOLUTION INFEASIBLE'}")
print(f"{'='*50}")


# In[69]:


# ── Run All Instances + Feasibility Check ──

from collections import defaultdict

summary_rows = []

for inst_id, config in instances.items():
    Nv_inst    = config["Nv"]
    C_inst     = config["C"]
    nodes_inst = config["nodes"]
    n_cust     = len(nodes_inst) - 1

    print(f"\n{'█'*60}")
    print(f"  INSTANCE {inst_id}:  {n_cust} customers, {Nv_inst} vehicles, capacity {C_inst}")
    print(f"{'█'*60}")

    # Distance matrix
    D_inst = build_distance_matrix(nodes_inst)

    # Phase 1: Sweep clustering
    # Phase 1: Sweep clustering
    # Phase 1: Sweep clustering
    clusters_inst = sweep_decomposition(nodes_inst, Nv_inst, C_inst)
    # --- FEASIBILITY CHECK ---
    all_customers = set(range(1, len(nodes_inst)))
    assigned = set(c for cluster in clusters_inst for c in cluster)

    if assigned != all_customers:
        print(f"❌ Instance {inst_id}: bad clustering (missing/duplicate customers)")

    # Capacity safety fix — split any cluster exceeding capacity
    fixed = []
    for cluster in clusters_inst:
        if len(cluster) <= C_inst:
            fixed.append(cluster)
        else:
            for i in range(0, len(cluster), C_inst):
                chunk = cluster[i:i+C_inst]
                if chunk:
                    fixed.append(chunk)
    # If splitting created more clusters than vehicles, merge smallest pairs
    while len(fixed) > Nv_inst:
        fixed.sort(key=len)
        merged = False

        for i in range(len(fixed) - 1):
            if len(fixed[i]) + len(fixed[i+1]) <= C_inst:
                fixed[i+1] = fixed[i] + fixed[i+1]
                fixed.pop(i)
                merged = True
                break

        if not merged:
            print(f"⚠️ Instance {inst_id}: cannot merge without exceeding capacity")
            break
    clusters_inst = fixed

    print(f"  Clusters: {clusters_inst}\n")

    # Capacity safety fix
    fixed = []
    for cluster in clusters_inst:
        if len(cluster) <= C_inst:
            fixed.append(cluster)
        else:
            for i in range(0, len(cluster), C_inst):
                chunk = cluster[i:i+C_inst]
                if chunk:
                    fixed.append(chunk)
    while len(fixed) > Nv_inst:
        fixed.sort(key=len)
        merged = False

        for i in range(len(fixed) - 1):
            if len(fixed[i]) + len(fixed[i+1]) <= C_inst:
                fixed[i+1] = fixed[i] + fixed[i+1]
                fixed.pop(i)
                merged = True
                break

        if not merged:
            print(f"⚠️ Instance {inst_id}: cannot merge without exceeding capacity")
            break
    clusters_inst = fixed

    print(f"  Clusters: {clusters_inst}\n")

    # Phase 2: QAOA routing per cluster
    routes_inst = []
    td, mq, tg, tt = 0, 0, 0, 0.0

    for i, cluster in enumerate(clusters_inst):
        route, dist, nq, ng, et, _ = solve_tsp_dqi(
    cluster, D, reps=2, restarts=5, shots=8192)
        full_route = [0] + route + [0]
        routes_inst.append(full_route)
        td += dist
        mq = max(mq, nq)
        tg += ng
        tt += et
        print(f"  r{i+1}: {' → '.join(map(str, full_route))}  |  dist={dist:.2f}  |  qubits={nq}")

    print(f"\n  Total Distance: {td:.2f}  |  Max Qubits: {mq}  |  Gates: {tg}  |  Time: {tt:.2f}s")

    # ── Feasibility Check ──
    print(f"\n  {'─'*50}")
    print(f"  FEASIBILITY CHECK")
    print(f"  {'─'*50}")

    all_customers = set(range(1, len(nodes_inst)))
    visited = []
    feasible = True
    in_deg  = defaultdict(int)
    out_deg = defaultdict(int)

    for i, route in enumerate(routes_inst):
        custs = [c for c in route if c != 0]

        # Depot start/end
        if route[0] != 0 or route[-1] != 0:
            print(f"    ✗ Route {i+1}: doesn't start/end at depot")
            feasible = False

        # Capacity
        if len(custs) > C_inst:
            print(f"    ✗ Route {i+1}: capacity exceeded ({len(custs)} > {C_inst})")
            feasible = False

        # Internal repeats
        if len(custs) != len(set(custs)):
            print(f"    ✗ Route {i+1}: repeated customers")
            feasible = False

        # Edges for flow
        for k in range(len(route) - 1):
            out_deg[route[k]]   += 1
            in_deg[route[k+1]]  += 1

        visited.extend(custs)

    # All customers visited exactly once
    visited_set = set(visited)
    missing     = all_customers - visited_set
    duplicates  = set(c for c in visited if visited.count(c) > 1)
    if missing:
        print(f"    ✗ Missing customers: {missing}")
        feasible = False
    if duplicates:
        print(f"    ✗ Duplicate visits: {duplicates}")
        feasible = False

    # Vehicle count
    if len(routes_inst) > Nv_inst:
        print(f"    ✗ Too many vehicles: {len(routes_inst)} > {Nv_inst}")
        feasible = False

    # Flow: depot
    if out_deg[0] != len(routes_inst) or in_deg[0] != len(routes_inst):
        print(f"    ✗ Depot flow imbalanced: out={out_deg[0]}, in={in_deg[0]}")
        feasible = False

    # Flow: customers
    flow_bad = [(c, in_deg[c], out_deg[c]) for c in all_customers
                if in_deg[c] != 1 or out_deg[c] != 1]
    if flow_bad:
        for c, i_d, o_d in flow_bad:
            print(f"    ✗ Customer {c}: in={i_d}, out={o_d}")
        feasible = False

    # Distance verification
    recomputed = sum(
        sum(D_inst[r[k], r[k+1]] for k in range(len(r)-1))
        for r in routes_inst
    )
    if abs(recomputed - td) > 1e-6:
        print(f"    ✗ Distance mismatch: reported={td:.2f}, recomputed={recomputed:.2f}")
        feasible = False

    status = "✓ FEASIBLE" if feasible else "✗ INFEASIBLE"
    print(f"\n  {status}")
    summary_rows.append((inst_id, n_cust, Nv_inst, C_inst, td, mq, tg, tt, feasible))

# ── Summary Table ──
print(f"\n\n{'='*90}")
print(f"{'SUMMARY':^90}")
print(f"{'='*90}")
print(f"{'Inst':>5} {'Cust':>5} {'Veh':>4} {'Cap':>4} {'Distance':>10} {'Qubits':>7} {'Gates':>7} {'Time(s)':>8} {'Status':>12}")
print(f"{'─'*90}")
for inst_id, nc, nv, cap, dist, qb, gt, tm, feas in summary_rows:
    tag = "✓ FEASIBLE" if feas else "✗ INFEAS."
    print(f"{inst_id:>5} {nc:>5} {nv:>4} {cap:>4} {dist:>10.2f} {qb:>7} {gt:>7} {tm:>8.2f} {tag:>12}")
print(f"{'='*90}")
all_ok = all(r[-1] for r in summary_rows)
print(f"\nAll instances feasible: {'✓ YES' if all_ok else '✗ NO'}")


# In[70]: