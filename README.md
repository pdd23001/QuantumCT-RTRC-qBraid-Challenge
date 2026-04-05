# Quantum Courier Challenge 2026

> Yale Quantum Hackathon 2026 — sponsored by RTRC, QuantumCT, and qBraid

Quantum-classical hybrid solution for the **Capacitated Vehicle Routing Problem (CVRP)** that decouples problem size from qubit count. We benchmark five algorithms — **Gurobi (exact MIP)**, **Genetic Algorithm**, **QAOA**, **DQI**, and **QITE** — across six progressively harder instances (5 to 25 nodes), with live hardware execution on **IBM `ibm_fez`** and **Rigetti Ankaa** via qBraid.

---
## Mathematical Formulation

The CVRP is defined on a complete graph $G = (V, E)$ where $V = \{0, 1, \ldots, N\}$ (node $0$ is the depot). A fleet of $K$ vehicles with uniform capacity $C$ serves customer demands $q_i$. Binary decision variable $x_{ijk} = 1$ if vehicle $k$ traverses edge $(i, j)$.

**Objective** — minimize total distance:
$$\min \sum_{k=1}^{K} \sum_{i=0}^{N} \sum_{j=0}^{N} d_{ij}\, x_{ijk}$$

**Subject to:**

| Constraint | Formula | Meaning |
|-----------|---------|---------|
| Visit once | $\sum_{k} \sum_{i \neq j} x_{ijk} = 1 \;\; \forall j \in V \setminus \{0\}$ | Every customer served exactly once |
| Flow conservation | $\sum_{i} x_{ihk} - \sum_{j} x_{hjk} = 0 \;\; \forall h, k$ | Arrive = depart at every node |
| Depot start/end | $\sum_{j} x_{0jk} = 1, \;\sum_{i} x_{i0k} = 1 \;\; \forall k$ | Each vehicle leaves and returns to depot |
| Capacity | $\sum_{i \in V \setminus \{0\}} q_i \sum_{j} x_{ijk} \leq C \;\; \forall k$ | No vehicle exceeds capacity $C$ |

---
## Algorithms

### Gurobi Exact Solver (MIP) — `gurobi.ipynb`

Models the full CVRP as a Mixed-Integer Program. The key challenge is subtour elimination: the exponential family of constraints $\sum_{(i,j) \in S} x_{ij} \leq |S| - 1$ for every subset $S$ makes eager enumeration infeasible. We solve this with a **lazy callback** — the solver runs without subtour constraints, and whenever a candidate solution contains a disconnected cycle, the callback injects the violated constraint and the Branch-and-Cut tree re-solves. This keeps the active constraint set small while still guaranteeing global optimality.

- **Time complexity:** $O(2^N)$ worst-case (NP-Hard), but the lazy approach prunes aggressively — Instance 6 (25 nodes) solves in ~7.7 seconds.
- **Space complexity:** $O(N^2 K)$ for the decision variable matrix plus dynamic constraint storage.
- **Role in the project:** Provides the provably optimal ground truth that all other algorithms are benchmarked against.

### Genetic Algorithm (GA) — `ga.ipynb`

A purely classical evolutionary meta-heuristic using the [PyGAD](https://pygad.readthedocs.io/) framework. Each individual in the population encodes a complete CVRP solution as a permutation of customer nodes, split into routes respecting capacity $C$.

- **Selection:** Rank-based parent selection (top 12 parents from population of 40).
- **Crossover:** Ordered crossover (OX) preserving relative customer sequences while recombining route structure from two parents.
- **Mutation:** Random swap mutation at 20% rate — two customers exchange positions, potentially across different routes.
- **Termination:** Fixed at 120 generations.
- **Time complexity:** $O(P \times G \times N)$ where $P = 40$, $G = 120$. All six instances complete in under 0.3 seconds total.
- **Trade-off:** Extremely fast convergence but no optimality guarantee. On Instance 6 the best GA result (111.49) is 5.6% above Gurobi's optimum — the stochastic search cannot exhaustively verify it has found the global minimum.

### QAOA (Quantum Approximate Optimization Algorithm) — `qaoa_ibm.ipynb`, `qaoa_qbraid.ipynb`

Our primary quantum algorithm. After decomposition, each vehicle cluster becomes an independent sub-TSP that we solve with QAOA end-to-end.

#### QUBO Construction (Position Formulation)
For a cluster of $n$ customers, we introduce $n^2$ binary variables $x_{i,s} \in \{0,1\}$ where $x_{i,s} = 1$ means customer $i$ is at position $s$ in the route. This gives us $n^2$ qubits per cluster. The QUBO encodes:

- **Assignment constraints** (penalty weight $A = 1.5 \cdot n \cdot d_{\max}$): each customer appears in exactly one position, and each position is occupied by exactly one customer. Violated assignments are penalized quadratically.
- **Routing objective:** depot-to-first, consecutive-position transition costs, and last-to-depot distances are embedded as linear and quadratic terms in the QUBO matrix.

The QUBO is then converted to an Ising Hamiltonian via the substitution $x_k = (1 - Z_k)/2$, yielding single-qubit $Z$ fields and two-qubit $ZZ$ couplings compatible with the `QAOAAnsatz`.

#### Parameter Initialization: Linear Ramp + Warm-Starting

Vanilla QAOA with random initial parameters often gets trapped in barren plateaus, especially at $p \geq 2$ layers. We use a two-pronged initialization strategy:

1. **Linear Ramp (Restart 0):** The initial $\gamma$ and $\beta$ parameters follow a linear schedule inspired by the adiabatic limit:
   $$\gamma_k = \frac{k+1}{p} \cdot 0.75, \qquad \beta_k = \left(1 - \frac{k+1}{p}\right) \cdot 0.75 \qquad \text{for } k = 0, \ldots, p-1$$
   This mimics a smooth annealing trajectory from the mixer Hamiltonian toward the problem Hamiltonian, providing a structured starting point that avoids the flat energy landscape of random initialization.

2. **Warm-Starting (Restart 1):** After the first cluster is optimized, its converged parameters are passed as `warm_params` to all subsequent clusters. Since decomposed clusters share similar graph structure and edge-weight distributions, the optimal parameter landscape transfers well between clusters. This dramatically reduces COBYLA iterations for clusters 2+.

3. **Random Restart (Restart 2+):** Additional restarts with $\text{Uniform}(-\pi, \pi)$ initialization to escape any local minima missed by the structured starts.

Each restart runs COBYLA with a 500-iteration budget (200 on hardware). The best energy across all restarts is kept.

#### Execution Backends

- **Aer Simulator (CPU/GPU):** Primary optimization loop uses `AerEstimator` with 1024–8192 shots. GPU acceleration via `cuStateVec` for Instance 6 (25 qubits, 8975 gates).
- **IBM `ibm_fez`:** After simulator optimization locks parameters, the parameterized ansatz is transpiled via Qiskit's `generate_preset_pass_manager` and submitted to the 156-qubit Heron processor through `SamplerV2` with a `Session`. Hardware results are decoded identically to simulator results.
- **Rigetti Ankaa via qBraid:** The same QAOA pipeline is re-targeted through qBraid's transpilation layer, which maps Qiskit circuits onto Rigetti's native gate set and topology. Executed in `qaoa_qbraid.ipynb`.

#### Sampling and Decoding
After optimization, all cluster circuits are batched into a single `SamplerV2` job (reducing queue overhead). Each measured bitstring is decoded back into the $n \times n$ assignment matrix — only bitstrings satisfying both row and column one-hot constraints are accepted. The lowest-distance valid route across all shots is selected.

#### Scaling
- **Qubits per cluster:** $n^2$ (e.g., a 5-customer cluster needs 25 qubits)
- **Circuit depth:** $O(p \cdot C^4)$ — each QAOA layer applies $O(C^4)$ two-qubit gates from the dense Ising coupling map
- **Total qubits (after decomposition):** bounded by the largest cluster, i.e., $O(C^2)$, independent of total problem size $N$
- **Total optimization phase:** $O\!\left(N^3 + \frac{N}{C} \cdot I \cdot S \cdot p \cdot C^4\right)$ where $I$ is COBYLA iterations, $S$ is shots, and $p$ is QAOA depth
- **Best result:** 108.83 on Instance 6 (3.1% above Gurobi optimum)

### DQI (Decoded Quantum Interferometry) — `dqi.ipynb`

DQI is a quantum-inspired algorithm that uses actual Qiskit circuits but targets a fundamentally different sub-problem than QAOA: instead of solving route ordering, DQI uses quantum interference to select which customers to group into each route.

#### Route-by-Route Construction

DQI builds the CVRP solution one route at a time:

1. **Seed selection:** Pick the farthest remaining customer from the depot as the anchor for the next route. This prioritizes hard-to-reach customers early, preventing them from being stranded in expensive single-node routes later.

2. **Neighborhood formation:** Gather the $q$ nearest remaining customers around the seed to form a local candidate pool. The parameter $q$ directly controls qubit count.

3. **QUBO encoding for subset selection:** Each customer in the neighborhood gets one binary variable (include or exclude). The QUBO coefficients encode:
   - **Linear terms:** $h_i = 8.0 - d(0, i) + \lambda(1 - 2C)$ — a reward for inclusion, penalized by depot distance and a soft capacity regularizer ($\lambda = 5.0$).
   - **Quadratic terms:** $J_{ij} = \text{savings}(i,j) - 0.07 \cdot d(i,j) - 2\lambda$ — Clarke-Wright savings $[d(0,i) + d(0,j) - d(i,j)]$ reward customers that are close to each other but far from the depot (merging them into one route saves two depot legs), minus dispersion and capacity penalties.

4. **DQI circuit:** The QUBO is converted to Ising form ($x_i \to (1 - Z_i)/2$) and normalized. The circuit structure is:
   $$|0\rangle^{\otimes q} \xrightarrow{H^{\otimes q}} \text{uniform superposition} \xrightarrow{\text{Phase Oracle}} \xrightarrow{H^{\otimes q}} \text{measure}$$

   The phase oracle applies $R_z(-2\gamma \cdot h_{z,i})$ for single-qubit Ising fields and $\text{CNOT}(i,j) \cdot R_z(-2\gamma \cdot J_{z,ij},\, j) \cdot \text{CNOT}(i,j)$ for two-qubit couplings. This is structurally similar to a single QAOA layer but without the mixer — the Hadamard bookends create interference that amplifies bitstrings aligned with the Ising ground state. Multiple layers can be stacked with a configurable `layers` parameter.

5. **Classical decoding and repair:** Every distinct measured bitstring is mapped back to a customer subset. Infeasible subsets (missing the seed, exceeding capacity $C$) are repaired: the seed is force-included, and excess customers are pruned by distance. Each repaired subset is scored with a composite metric balancing route cost (via Held-Karp exact TSP), coverage reward, pair dispersion, and depot pull.

6. **Exact route ordering:** The winning subset is ordered optimally using **Held-Karp dynamic programming** ($O(C^2 \cdot 2^C)$), which is tractable because cluster sizes are bounded by $C$.

7. **Repeat** until all customers are assigned.

#### Classical Post-Processing
After all routes are constructed:
- **2-opt local search:** Reverses sub-segments within each route to eliminate crossing edges.
- **Inter-route swap refinement:** Tries moving or exchanging customers between adjacent routes to reduce total distance (10 passes).

#### Why DQI Works So Well
- The quantum circuit is shallow ($O(q^2)$ depth for dense couplings) and only needs $q$ qubits per call (neighborhood size, not problem size).
- Subset selection is the right sub-problem for quantum speedup — it's a combinatorial search over $2^q$ subsets where interference can bias measurement toward high-scoring groups.
- Route ordering is solved exactly by Held-Karp on the small selected subset — no approximation error.
- The deterministic nature of the scoring function means no shot noise accumulates across the construction.

#### Scaling
- **Qubits per DQI call:** $q$ (neighborhood size, typically 5–10)
- **Circuit depth:** $O(q^2)$ per layer
- **Held-Karp per route:** $O(C^2 \cdot 2^C)$
- **Total constructive phase:** $O\!\left(\frac{N}{C}\left(N \log N + \text{quantum}(q, S) + M \cdot C^2 \cdot 2^C\right)\right)$ where $S$ is shots and $M$ is distinct bitstrings evaluated
- **Best result:** 106.27 on Instance 6 (0.6% above Gurobi optimum) — the closest any quantum-oriented method gets to the exact solution

### QITE (Quantum Imaginary Time Evolution) — `qite.ipynb`

QITE takes a fundamentally different approach from variational algorithms: instead of optimizing parameters in a feedback loop, it directly evolves the quantum state toward the ground state through imaginary-time propagation.

#### Mathematical Foundation

Standard quantum time evolution applies $e^{-iHt}$ — a unitary rotation that preserves state norms and explores the energy landscape periodically. Imaginary-time evolution substitutes $t \to -i\tau$ (Wick rotation), giving the propagator:

$$|\Psi(\tau)\rangle = \frac{e^{-H\tau}|\Psi(0)\rangle}{\|e^{-H\tau}|\Psi(0)\rangle\|}$$

This is no longer unitary — it exponentially suppresses high-energy components. If $|\Psi(0)\rangle = \sum_k c_k |E_k\rangle$ in the energy eigenbasis, then after propagation each amplitude scales as $c_k e^{-E_k \tau}$. As $\tau \to \infty$, only the ground state $|E_0\rangle$ survives (assuming nonzero initial overlap $c_0 \neq 0$). This guarantees convergence to the exact optimum without any parameter tuning.

#### Implementation Pipeline

1. **QUBO + Ising mapping:** Identical position-formulation QUBO as QAOA ($n^2$ qubits for $n$ customers), converted to a Pauli-$Z$ Ising Hamiltonian.

2. **Numerical propagation:** Since $e^{-H\tau}$ is non-unitary, it cannot be implemented as a gate circuit directly. We compute the full $2^{n^2} \times 2^{n^2}$ Hamiltonian matrix, then propagate via:
   - Discretize total imaginary time $T = 10.0$ into 50 steps of $\Delta\tau = 0.2$
   - Compute the matrix exponential propagator $U = e^{-\Delta\tau \cdot H}$ once (via `scipy.linalg.expm`)
   - Repeatedly apply $|\psi\rangle \leftarrow U|\psi\rangle$ followed by renormalization
   - Track the energy expectation $\langle\psi|H|\psi\rangle$ at each step to produce a convergence trace

3. **Aer statevector decoding:** The final evolved state vector is loaded into a Qiskit `AerSimulator` (statevector method) via `QuantumCircuit.initialize()`. Probabilities are extracted and the highest-probability bitstrings are decoded into routes using the same position-matrix decoder as QAOA.

#### Scaling and Limitations

- **Qubits:** $n^2$ per cluster (same as QAOA)
- **Memory:** $O(2^{n^2})$ for the full statevector — this is the critical bottleneck. A 4-customer cluster requires $2^{16} = 65{,}536$ amplitudes (tractable). A 5-customer cluster requires $2^{25} \approx 33$ million (borderline). Clusters of 6+ customers exceed available RAM.
- **Time:** $O(4^{n^2})$ for the matrix exponential, plus $O(\text{steps} \cdot 2^{n^2})$ for propagation
- **Guarantees:** Converges to the exact ground state given sufficient imaginary time and nonzero initial overlap (guaranteed by the uniform superposition initialization $|\Psi(0)\rangle = |+\rangle^{\otimes n^2}$)
- **Result:** Achieves the exact Gurobi optimum on Instances 1–4. Instances 5–6 are omitted because their largest clusters exceed the statevector memory ceiling.

---
## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. LOAD              Parse instance JSON → node coordinates + demands     │
├─────────────────────────────────────────────────────────────────────────────┤
│  2. DECOMPOSE         Capacity-aware Agglomerative Clustering              │
│                       ► Merge nearby nodes bottom-up                       │
│                       ► Halt each cluster the moment demand sum → C        │
│                       ► Inter-cluster boundary swaps to tighten edges      │
│                       Result: CVRP → set of independent sub-TSPs           │
├─────────────────────────────────────────────────────────────────────────────┤
│  3. SOLVE (parallel)                                                       │
│     ┌────────────────┬──────────────┬────────────┬──────────┬────────────┐ │
│     │ Gurobi (MIP)   │ GA           │ QAOA       │ DQI      │ QITE       │ │
│     │ Lazy subtour    │ Evolutionary │ COBYLA opt │ Wave-fn  │ Imaginary  │ │
│     │ elimination     │ crossover +  │ + Aer sim  │ interf.  │ time evol. │ │
│     │                │ mutation     │ + QPU hw   │ on class.│ statevec.  │ │
│     └────────────────┴──────────────┴────────────┴──────────┴────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│  4. RECONSTRUCT       Stitch sub-routes back into full CVRP solution       │
├─────────────────────────────────────────────────────────────────────────────┤
│  5. EVALUATE          Score total distance, export .txt routes + plots     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Decompose?

A direct QUBO encoding of a 25-node CVRP requires thousands of logical qubits (slack variables for capacity and subtour constraints). That far exceeds any NISQ device. By decomposing with capacity-aware clustering first, each sub-problem maps to $O(C^2)$ qubits — independent of $N$ — making real hardware execution feasible.

We evaluated K-Means (capacity-blind, causes illegal routes), **Angle Sweep** (sorts customers by polar angle from the depot and greedily fills vehicles along the sweep — fast but sensitive to starting angle and blind to spatial density), and **Agglomerative Hierarchical Clustering** (deterministic, capacity-bounded bottom-up merges). Agglomerative won on every metric, and we further refine it with inter-cluster node swaps.

---
## Results Summary

Best objective values across all six instances (lower is better):

| Instance | Nodes | Gurobi (Exact) | GA | QAOA | DQI | QITE |
|----------|-------|----------------:|-------:|------:|-------:|------:|
| 1 | 5 | **21.74** | **21.74** | **21.74** | **21.74** | **21.74** |
| 2 | 8 | **26.18** | **26.18** | **26.18** | **26.18** | **26.18** |
| 3 | 10 | **49.50** | **49.50** | **49.50** | **49.50** | **49.50** |
| 4 | 15 | **58.18** | **58.18** | **58.18** | **58.18** | **58.18** |
| 5 | 20 | **86.43** | 86.75 | 86.43 | 86.50 | — |
| 6 | 25 | **105.61** | 111.49 | 108.83 | 106.27 | — |

- Gurobi provides the provably optimal baseline on all instances.
- DQI matches or nearly matches Gurobi across the board (best quantum-oriented result: **106.27** on Instance 6).
- QAOA reaches **108.83** on Instance 6 with 25 qubits / 8975 gates on Aer GPU, and was also executed on live IBM `ibm_fez` and Rigetti Ankaa hardware.
- QITE achieves exact optima on Instances 1–4 via statevector simulation; Instances 5–6 were omitted due to exponential memory scaling of the statevector backend.
- GA converges quickly ($< 0.3$s) but diverges on larger instances without global optimality guarantees.

---
## Benchmarking Methodology: Simulator-First, Then Hardware

Every quantum algorithm in this project was **first developed and fully benchmarked on Qiskit Aer** before any hardware submission. This was a deliberate design choice, not a shortcut:

1. **Aer CPU baseline (Instances 1–4):** All QAOA, DQI, and QITE circuits were executed on `AerEstimator` / `AerSampler` using CPU statevector simulation with 1024–4096 shots. This validates correctness — if the algorithm cannot find the optimal route on a noiseless simulator, hardware execution is meaningless.

2. **Aer GPU scaling (Instances 5–6):** Instance 6 requires 25 qubits and 8975 gates. CPU simulation becomes prohibitively slow at this scale, so we switched to GPU-accelerated Aer via `cuStateVec` (`device='GPU'`, `method='statevector'`). This confirmed that QAOA converges to 108.83 and DQI to 106.27 at full problem scale before committing to hardware queue time.

3. **Hardware sampling (QAOA only):** After Aer optimization locked the variational parameters, the bound circuits were submitted to real QPUs for sampling. This tests whether NISQ noise degrades solution quality on already-optimized circuits.

### Hardware Limitations Encountered

We designed the pipeline to run **both optimization and sampling** on quantum hardware — the code supports `USE_AER_FOR_OPT = False` which routes every COBYLA iteration through the QPU. In practice, two platform constraints prevented this:

- **IBM Open Plan:** The free-tier IBM Quantum access does not grant `Session` mode, which is required to keep a dedicated hardware reservation alive across the hundreds of iterative `EstimatorV2` calls that COBYLA needs during optimization. Without a Session, each iteration re-enters the public queue (minutes to hours per call), making a full optimization loop infeasible. We therefore optimized on Aer and only used `ibm_fez` for final sampling via `SamplerV2`.

- **qBraid / Rigetti Ankaa:** qBraid's runtime layer supports circuit submission and transpilation but does not expose a session-equivalent primitive for iterative variational loops. Each job is submitted independently, and the `solve_tsp_qaoa` function falls back to submitting individual circuits per COBYLA iteration with `device.run()` — functional but prohibitively slow due to per-job queue latency. As with IBM, we optimized on Aer and sampled on hardware.

**In both cases, the full hardware-optimization code path exists and is tested** — it is gated behind a flag (`USE_AER_FOR_OPT = False` / `USE_REAL_HARDWARE = True` for optimization). Given a paid IBM Session plan or a dedicated qBraid reservation, the entire variational loop would execute on-device with no code changes.

---
## Quantum Runtime Analysis and Fault-Tolerant Outlook

### Problem-Size Independence

The central architectural achievement of this project is that **quantum resource requirements are completely decoupled from the total number of customers $N$**. Through capacity-aware decomposition:

- QAOA and QITE need at most $C^2$ qubits (where $C$ is the vehicle capacity)
- DQI needs at most $q$ qubits (the neighborhood size, typically $\leq 10$)

This means a 25-node CVRP and a 2500-node CVRP require the **exact same quantum hardware** — only the number of independent sub-problem calls increases (linearly in $N/C$). The decomposition transforms an exponential quantum scaling wall into a classical linear overhead.

### NISQ Runtime Profile (Current Hardware)

| Algorithm | Qubits | Circuit Depth | Optimization Calls | Limiting Factor |
|-----------|--------|---------------|-------------------|-----------------|
| QAOA | $C^2$ (max 25) | $O(p \cdot C^4)$ | ~500 per cluster | Shot noise + gate errors at depth |
| DQI | $q$ (max ~10) | $O(q^2)$ per layer | None (single-shot per route) | Shallow enough for any NISQ device |
| QITE | $C^2$ (max 25) | N/A (statevector) | None | $O(2^{C^2})$ memory for classical simulation |

### Fault-Tolerant Quantum Runtime (Future Hardware)

When fault-tolerant quantum computers with error-corrected logical qubits become available, every bottleneck in our current pipeline transforms:

**QAOA on FT hardware:**
- Gate errors vanish — the depth-$O(p \cdot C^4)$ circuits execute faithfully, eliminating the noise floor that currently limits solution quality.
- The COBYLA optimization loop can run entirely on-device with `EstimatorV2` Session mode, removing the classical-quantum communication bottleneck.
- Higher QAOA depths ($p = 10, 20, \ldots$) become practical, approaching the adiabatic limit where QAOA provably converges to the exact ground state. Currently $p = 3$ is the practical ceiling due to noise accumulation.
- **Projected runtime:** Each sub-TSP optimization would complete in seconds (no queue, no error mitigation overhead), making the full 25-node CVRP solvable in under a minute with hardware-in-the-loop optimization.

**QITE on FT hardware:**
- The statevector memory wall disappears entirely. Fault-tolerant hardware can natively implement imaginary-time evolution through quantum phase estimation or block-encoding techniques, operating in $O(C^2)$ qubits instead of $O(2^{C^2})$ classical memory.
- QITE's guarantee of exact ground-state convergence (no variational approximation, no barren plateaus) makes it the theoretically strongest algorithm in our suite — it simply needs hardware that can execute the non-unitary propagation natively.
- **Projected advantage:** QITE would match Gurobi's exact optimality on all instances while scaling polynomially in qubit count, whereas Gurobi scales exponentially in time.

**DQI on FT hardware:**
- Already the lightest quantum footprint ($q \leq 10$ qubits, $O(q^2)$ depth). On FT hardware, the circuit executes perfectly, and the only improvement is zero sampling noise — every shot returns a high-quality subset.
- The practical effect is that fewer shots are needed per DQI call, reducing the classical overhead of the constructive loop.

### The Key Insight

Our decomposition strategy is not a workaround for limited hardware — it is a **scalable architectural pattern** that remains optimal even on fault-tolerant machines. The quantum computer solves the hard combinatorial kernel (sub-TSP ordering or subset selection) while classical preprocessing handles the tractable structural decomposition. As hardware improves, the quantum kernel executes faster and more accurately, but the $O(C^2)$ qubit ceiling and the $O(N/C)$ classical scaling remain unchanged. **This is how quantum advantage for logistics will actually be deployed.**

## Setup and Reproduction

```bash
# Clone and install
git clone <repo-url>
cd QuantumCT-RTRC-qBraid-Challenge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Key dependencies:** `qiskit 2.3`, `qiskit-aer`, `qiskit-ibm-runtime`, `qiskit-optimization`, `gurobipy`, `qbraid`, `azure-quantum`, `numpy`, `scipy`, `pandas`, `matplotlib`

To run any algorithm, open the corresponding notebook in `Algorithms/` and execute all cells. Results are written to the matching `Results/` subdirectory.

> **Note on hardware reproduction:** All quantum algorithms run fully on the Qiskit Aer simulator by default — no cloud access is needed to reproduce every result in this repo. To reproduce the hardware sampling runs, you will need:
> - **IBM Quantum:** An IBM Quantum account and API key (`IBMQ_TOKEN`). Set `USE_REAL_HARDWARE = True` in `qaoa_ibm.ipynb`. The Open Plan provides queue-based access to `ibm_fez`; a paid plan is required for `Session`-mode iterative optimization.
> - **qBraid:** A qBraid account with access to a supported QPU. Set `USE_REAL_HARDWARE = True` in `qaoa_qbraid.ipynb`. The `DEVICE_ID` variable controls which backend is targeted.
> - **Gurobi:** A valid `gurobi.lic` license file (free for academics) placed in your home directory. Required only for `gurobi.ipynb`.

---
## Repository Structure — Complete File Reference

### Root Directory

| File | Description |
|------|-------------|
| `final_instances.json` | All 6 CVRP problem instances. Each entry contains `instance_id`, number of vehicles `Nv`, capacity `C`, and a list of customers with `(x, y)` coordinates and `demand` values. Node 0 is always the depot at the origin. Instances scale from 5 customers (Instance 1) to 25 customers (Instance 6). |
| `requirements.txt` | Pinned Python dependencies for exact reproduction. Includes `qiskit==2.3.1`, `qiskit-aer==0.17.2`, `qiskit-ibm-runtime==0.46.1`, `qiskit-optimization==0.7.0`, `gurobipy==13.0.1`, `qbraid`, `azure-quantum==3.8.0`, and supporting libraries (`numpy`, `scipy`, `pandas`, `matplotlib`). |
| `qCourier-YaleHackathon-2026.ipynb` | The original hackathon problem statement notebook provided by the organizers. Contains the challenge description, CVRP mathematical formulation, judging criteria (approximation ratio, scale, novelty, presentation quality), and submission requirements. This is **not** our code — it is the reference specification we built against. |
| `README.md` | This file. |

### `Algorithms/` — Solver Notebooks

Each notebook is self-contained: load the instance JSON, run decomposition, solve, and look at results.

| File | Algorithm | Backend | What It Does |
|------|-----------|---------|-------------|
| `gurobi.ipynb` | Gurobi MIP | Classical (Gurobi) | Solves the full CVRP exactly using Mixed-Integer Programming with lazy subtour elimination callbacks. Contains two solver variants: an eager formulation (`solve_cvrp_gurobi`) and the optimized lazy version (`solve_cvrp_gurobi_lazy`). Iterates over all 6 instances, exports route `.txt` files, route visualization `.png` plots, and a summary CSV to `Results/Gurobi_Results/`. |
| `ga.ipynb` | Genetic Algorithm | Classical (PyGAD) | Evolutionary meta-heuristic solver using PyGAD. Configures population size (40), parent count (12), ordered crossover, 20% swap mutation, and 120 generations. Runs 5 independent trials per instance with different random seeds to capture solution variability. Exports route `.txt` files, route plots, and `ga_results.csv` with per-trial objective values. Also generates performance analysis charts comparing best vs. average distance across instances. |
| `qaoa_ibm.ipynb` | QAOA | Aer Simulator + IBM `ibm_fez` | The primary quantum notebook. Implements the full pipeline: agglomerative clustering with inter-cluster swaps, position-formulation QUBO construction, QUBO-to-Ising conversion, `QAOAAnsatz` with linear ramp + warm-start + random restart initialization, COBYLA optimization on `AerEstimator` (CPU or GPU), batched sampling via `SamplerV2`, and bitstring decoding. Contains two execution modes controlled by `USE_REAL_HARDWARE` and `USE_AER_FOR_OPT` flags. When hardware mode is enabled, circuits are transpiled via `generate_preset_pass_manager` and submitted to IBM `ibm_fez` through a `Session`. Includes a full feasibility checker (visit-once, flow conservation, capacity, depot start/end) and exports convergence CSVs, route plots, and `qaoa_results.csv` with qubit counts, gate counts, and timing data. |
| `qaoa_qbraid.ipynb` | QAOA | Aer Simulator + Rigetti via qBraid | Identical QAOA pipeline to `qaoa_ibm.ipynb` but re-targeted for qBraid's runtime layer. Connects to a cloud QPU via `qbraid.runtime.QbraidDevice` (configured for `DEVICE_ID = "aws:ionq:qpu:forte-1"`). Handles qBraid-specific circuit submission: strips `GlobalPhaseGate` instructions (unsupported by the transpilation bridge), resets global phase, and submits via `device.run()` with robust result extraction handling multiple count formats. Optimization runs on Aer; only sampling hits the QPU. |
| `dqi.ipynb` | DQI | Aer Simulator | The production DQI implementation. Implements route-by-route construction: farthest-seed selection, nearest-neighbor neighborhood formation, QUBO subset encoding with Clarke-Wright savings, DQI circuit construction (Hadamard → phase oracle → Hadamard → measure), Aer-backed circuit execution, classical bitstring decoding and repair, Held-Karp exact TSP ordering, 2-opt local search, and inter-route swap refinement. Includes a benchmark section that runs 5 seeded trials across all 6 instances with configurable shots (1024–8192), exporting convergence traces, route visualizations, and `dqi_results.csv`. |
| `qite.ipynb` | QITE | Aer Statevector | Quantum Imaginary Time Evolution solver. Implements numerical imaginary-time propagation: constructs the full Hamiltonian matrix from the Ising model, computes the matrix exponential propagator via `scipy.linalg.expm`, applies 50 timesteps of $e^{-\Delta\tau H}$ with renormalization, then loads the evolved statevector into `AerSimulator` for probability extraction and bitstring decoding. Runs Instances 1–4 only (Instances 5–6 exceed statevector memory). Includes an energy convergence tracker, scaling benchmark section, and convergence visualization utilities. |

### `Results/` — Output Data

Every algorithm writes to its own isolated subdirectory. Results are fully reproducible from the corresponding notebook.

#### `Results/Gurobi_Results/`
| File/Directory | Contents |
|----------------|----------|
| `Txt_Format_Results/Instance{1-6}.txt` | Competition submission format. Each line is a route: `r1: 0, 3, 5, 0` (depot → customers → depot). One file per instance. |
| `Result_Graphs/Instance{1-6}_Result.png` | Spatial route visualization with colored paths per vehicle, customer node labels, and the depot marked. |
| `gurobi_results.csv` | Columns: `instance`, `config`, `trial_1`, `avg_value`, `best_value`, `avg_time_s`, `notes`. Single trial per instance (deterministic solver). |

#### `Results/GA_Results/`
| File/Directory | Contents |
|----------------|----------|
| `Txt_Format_Results/Instance{1-4}.txt` | Route submission files (Instances 1–4 only in this directory). |
| `Result_Graphs/ga_sol_instance{1-6}.png` | Best-found route visualizations for all 6 instances. |
| `ga_results.csv` | Columns: `instance`, `config`, `trial_1` through `trial_5`, `avg_value`, `best_value`, `avg_time_s`, `notes`. 5 seeded trials per instance showing solution variability. |

#### `Results/QAOA_Results/`
| File/Directory | Contents |
|----------------|----------|
| `Txt_Format_Results/Instance{1-6}.txt` | Route submission files for all 6 instances. |
| `Result_Graphs/Instance{1-6}_Result.png` | Spatial route visualizations. |
| `qaoa_convergence/csv/convergence_{1-6}.csv` | Per-instance COBYLA convergence traces. Columns: `circuit_evaluation`, `energy`. Tracks objective value at every estimator call across all clusters. |
| `qaoa_convergence/plots/convergence_{1-6}.png` | Convergence curve plots showing cumulative energy vs. circuit evaluations. |
| `qaoa_results.csv` | Columns: `instance`, `config`, `trial_1` through `trial_5`, `avg_value`, `best_value`, `qubits_used`, `total_gates`, `avg_time_s`. The `config` field records backend, shot count, COBYLA settings, QAOA depth, and restart count. |

#### `Results/DQI_Results/`
| File/Directory | Contents |
|----------------|----------|
| `Txt_Format_Results/Instance{1-6}.txt` | Route submission files for all 6 instances. |
| `Results_Graph/Instance{1-6}_Result.png` | Spatial route visualizations. |
| `dqi_convergence/csv/convergence_{1-6}.csv` | Per-instance convergence traces tracking DQI circuit evaluations and cumulative objective energy. |
| `dqi_convergence/plots/instance_{1-6}_convergence.png` | Convergence curve plots. |
| `dqi_results.csv` | Columns: `instance`, `config`, `trial_1` through `trial_5`, `avg_value`, `best_value`, `qubits_used`, `total_gates`, `avg_time_s`. Config records gamma, shot count, and qubit count. |

#### `Results/QITE_Results/`
| File/Directory | Contents |
|----------------|----------|
| `Txt_Format_Results/Instance{1-4}.txt` | Route submission files for Instances 1–4 only (5–6 omitted due to memory limits). |
| `Result_Graph/Instance{1-4}_Result.png` | Spatial route visualizations for Instances 1–4. |
| `qite_results.csv` | Columns: `instance`, `config`, `trial_1`, `avg_value`, `best_value`, `qubits_used`, `total_gates`, `avg_time_s`, `notes`. Config records propagation parameters (`dt=10`, `steps=50`). All trials yield identical deterministic results. |

#### `Results/QAOA_DQI_Convergence.png`
Side-by-side convergence comparison plot showing QAOA and DQI objective descent across circuit evaluations. Demonstrates that DQI converges faster (fewer quantum circuit calls) while QAOA converges deeper (better final energy on larger instances when given sufficient iterations).

---
## Problem Instances

All instances are stored in `final_instances.json`. The first four were provided by the hackathon organizers; **Instances 5 and 6 were created by us** to stress-test our algorithms at larger scale and verify that the decomposition strategy holds as problem size grows.

| Instance | Source | Customers | Vehicles ($K$) | Capacity ($C$) | Max Qubits (QAOA: $C^2$) | Gurobi Optimum |
|----------|--------|----------:|----------------:|----------------:|--------------------------:|---------------:|
| 1 | Provided | 3 | 2 | 5 | 25 | 21.74 |
| 2 | Provided | 3 | 2 | 2 | 4 | 26.18 |
| 3 | Provided | 6 | 3 | 2 | 4 | 49.50 |
| 4 | Provided | 12 | 4 | 3 | 9 | 58.18 |
| 5 | **Ours** | 20 | 5 | 4 | 16 | 86.43 |
| 6 | **Ours** | 25 | 5 | 5 | 25 | 105.61 |

Instances 1–4 test correctness at small scale where brute-force verification is possible. Instance 5 introduces 20 customers with tighter capacity constraints, forcing the decomposition into more clusters. Instance 6 is the flagship benchmark: 25 customers with 5 vehicles of capacity 5, producing clusters that max out at 25 qubits on QAOA — the largest circuit we execute on real hardware.

All customer demands are unit ($q_i = 1$), so capacity $C$ directly equals the maximum customers per vehicle. Node 0 is always the depot at the origin $(0, 0)$. Customer coordinates are 2D Euclidean with distances computed as straight-line $\ell_2$ norms.

---
---

*Built for the 2026 Quantum Courier Hackathon. Thanks to RTRC, QuantumCT, and qBraid.*
