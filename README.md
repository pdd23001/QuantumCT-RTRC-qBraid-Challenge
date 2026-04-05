# Quantum Courier Challenge 2026

> Yale Quantum Hackathon 2026 — sponsored by RTRC, QuantumCT, and qBraid

Quantum-classical hybrid solution for the **Capacitated Vehicle Routing Problem (CVRP)** that decouples problem size from qubit count. We benchmark five algorithms — **Gurobi (exact MIP)**, **Genetic Algorithm**, **QAOA**, **DQI**, and **QITE** — across six progressively harder instances (5 to 25 nodes), with live hardware execution on **IBM `ibm_fez`** and **Rigetti Ankaa** via qBraid.

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

We evaluated K-Means (capacity-blind, causes illegal routes), DBSCAN (fractures outliers into degenerate single-node clusters), and **Agglomerative Hierarchical Clustering** (deterministic, capacity-bounded merges). Agglomerative won on every metric.

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

### Gurobi Exact Solver (MIP)
Models the full CVRP as a Mixed-Integer Program with **lazy subtour elimination** callbacks — subtour constraints are only added when the solver produces a disconnected solution, keeping the constraint set manageable. Guarantees global optimality but scales exponentially ($O(2^N)$) in the worst case.

### Genetic Algorithm (GA)
Evolutionary meta-heuristic operating on route permutations. Uses ordered crossover and random mutation across a population of 40 individuals over 120 generations. Runs in $O(P \times G \times N)$ time — very fast, but sacrifices optimality guarantees on larger instances.

### QAOA (Quantum Approximate Optimization Algorithm)
Variational quantum algorithm encoding each sub-TSP as a QUBO Hamiltonian. Optimized with COBYLA over 2–3 QAOA layers. Simulated on Qiskit Aer (CPU/GPU) and executed on real hardware:
- **IBM `ibm_fez`** — direct Qiskit Runtime submission
- **Rigetti Ankaa** — bridged via qBraid transpilation layer

### DQI (Deterministic Quantum-Inspired)
Classical algorithm that emulates quantum superposition and wave-function interference using continuous probability vectors. Avoids sampling noise entirely, producing deterministic results. Achieved **106.27** on Instance 6 — within 0.6% of the Gurobi optimum.

### QITE (Quantum Imaginary Time Evolution)
Evolves the quantum state under imaginary time $\tau$: $|\Psi(\tau)\rangle \propto e^{-H\tau}|\Psi(0)\rangle$. The state naturally decays toward the ground state (optimal solution) without variational parameter tuning. Executed via statevector simulation, which limits scalability to Instances 1–4 due to $O(2^n)$ memory requirements.

---

## Repository Structure

```
├── Algorithms/
│   ├── gurobi.ipynb               # Gurobi MIP solver with lazy subtour elimination
│   ├── ga.ipynb                   # Genetic Algorithm solver
│   ├── qaoa_ibm.ipynb             # QAOA via IBM Runtime (Aer + ibm_fez hardware)
│   ├── qaoa_qbraid.ipynb          # QAOA via qBraid (Rigetti Ankaa hardware)
│   ├── dqi.ipynb                  # Deterministic Quantum-Inspired solver
│   └── qite.ipynb                 # Quantum Imaginary Time Evolution solver
│
├── Results/
│   ├── Gurobi_Results/
│   │   ├── Txt_Format_Results/    # Route outputs (.txt) for submission
│   │   ├── Result_Graphs/         # Route visualizations (.png)
│   │   └── gurobi_results.csv     # Objective values and solve times
│   ├── GA_Results/
│   │   ├── Txt_Format_Results/
│   │   ├── Result_Graphs/
│   │   └── ga_results.csv
│   ├── QAOA_Results/
│   │   ├── Txt_Format_Results/
│   │   ├── Result_Graphs/
│   │   ├── qaoa_convergence/      # Per-instance convergence curves (csv + plots)
│   │   └── qaoa_results.csv       # Trials, qubits, gates, and timing data
│   ├── DQI_Results/
│   │   ├── Txt_Format_Results/
│   │   ├── Results_Graph/
│   │   ├── dqi_convergence/       # Per-instance convergence curves (csv + plots)
│   │   └── dqi_results.csv
│   ├── QITE_Results/
│   │   ├── Txt_Format_Results/    # Instances 1–4 only
│   │   ├── Result_Graph/
│   │   └── qite_results.csv
│   └── QAOA_DQI_Convergence.png   # Side-by-side convergence comparison
│
├── final_instances.json            # All 6 problem instances (coords, demands, capacity)
├── requirements.txt                # Pinned Python dependencies
└── README.md
```

---

## Setup and Reproduction

```bash
# Clone and install
git clone <repo-url>
cd QuantumCT-RTRC-qBraid-Challenge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Key dependencies:** `qiskit 2.3`, `qiskit-aer`, `qiskit-ibm-runtime`, `qiskit-optimization`, `gurobipy`, `qbraid`, `azure-quantum`, `numpy`, `scipy`, `pandas`, `matplotlib`

**Hardware access:** IBM `ibm_fez` requires an IBM Quantum account and active access plan. Rigetti Ankaa access is managed through qBraid. A valid Gurobi license is required for the exact solver (academic licenses are free).

To run any algorithm, open the corresponding notebook in `Algorithms/` and execute all cells. Results are written to the matching `Results/` subdirectory.

---

*Built for the 2026 Quantum Courier Hackathon. Thanks to RTRC, QuantumCT, and qBraid.*
