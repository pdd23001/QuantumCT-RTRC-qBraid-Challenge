# Quantum Courier Challenge — Hybrid CVRP Solver

**Sponsors:** RTX Technology Research Center (RTRC), QuantumCT, and qBraid  
**Event:** Yale Hackathon 2026

---

## Overview

This repository implements a **hybrid quantum-classical solver** for the
Capacitated Vehicle Routing Problem (CVRP), developed for the Yale Quantum
Courier Challenge.

The solver is designed to:
- Keep a valid quantum-classical story for the challenge
- Improve route quality on all four challenge instances
- Scale cleanly to larger synthetic benchmark instances (32–80 customers)
- Use quantum computation where it is actually beneficial (local route improvement)

---

## Architecture

The solver follows a five-stage pipeline:

```
Stage A  →  Seed Portfolio
            Multiple classical decompositions: sweep (standard, reverse,
            shifted x4), Clarke-Wright savings, capacitated k-means (x3 seeds)

Stage B  →  2-opt Cleanup
            Apply 2-opt to every route in every candidate solution

Stage C  →  Inter-route Repair
            Relocate one customer between routes (and swap) until no
            improving move exists. Routes re-optimized with 2-opt after
            each accepted move.

Stage D  →  Quantum Local Improvement
            For each route, select a small contiguous neighborhood of k
            customers with fixed left/right anchors. Build the anchored 
            position-QUBO and optimize the ordering exclusively using QAOA.
            Splice back only if route distance improves.

Stage E  →  Reporting
            Per-stage distance gains, validity, qubit/gate counts, runtime
```

### Why Local QAOA?

Full-cluster QAOA scales as k^2 qubits for k customers; running it on
50-100 customer routes requires thousands of qubits. Instead, we apply
QAOA as a **local improver** on bounded neighborhoods:

| qubit budget | k (neighborhood) | method |
|---|---|---|
| 9–36 | 3–6 | QAOA |
| 49 | 7 | QAOA (49 qubits) |
| 64 | 8 | QAOA (64 qubits) |

Set `max_local_qaoa_qubits` to control the budget. The default sets k=5 for local QAOA optimization runs.

---

## Repository Layout

```
QuantumCT-RTRC-qBraid-Challenge/
├── instances/
│   ├── common.py                   # Geometry, distance matrix, validation
│   ├── decomposition.py            # Seed portfolio (sweep, CW, k-means)
│   ├── local_search.py             # 2-opt, 3-opt route improvement
│   ├── repair.py                   # Inter-route relocate + swap
│   ├── quantum_improve.py          # Anchored local QAOA / brute-force
│   ├── hybrid_solver.py            # Full pipeline orchestration
│   ├── benchmark_VRP_solver.py     # Multi-method benchmark harness (CLI)
│   ├── instances.py                # Random benchmark instance generator
│   ├── setA_random_instances_grouped.json  # 27 synthetic benchmark instances
│   └── benchmark_output/           # Benchmark results and plots
├── notebooks/
│   ├── qaoa_sweep.ipynb            # Thin experimental notebook (imports shared modules)
│   └── gurobi.ipynb                # Exact Gurobi oracle for 4 small instances
├── qCourier-YaleHackathon-2026.ipynb   # Challenge statement
├── README.md
└── .venv/                          # Python virtual environment (Python 3.12)
```

---

## Quick Start

### Environment setup

All circuits run on qBraid-provided simulators. Locally:

```bash
cd QuantumCT-RTRC-qBraid-Challenge
source .venv/bin/activate   # or use .venv/bin/python3.12 directly
```

Required packages (already in `.venv`):
- `qiskit >= 2.0`
- `qiskit-aer`
- `qiskit-algorithms`
- `scipy`
- `numpy`
- `matplotlib`

### Run the benchmark harness

```bash
cd instances/

# Classical-only benchmark (fast, <10s total)
python3.12 benchmark_VRP_solver.py --no-quantum

# Clarke-Wright baseline only
python3.12 benchmark_VRP_solver.py --methods clarke_wright

# Classical seed portfolio + repair
python3.12 benchmark_VRP_solver.py --methods classical_best

# Full hybrid with local QAOA (slow for large instances)
python3.12 benchmark_VRP_solver.py --methods hybrid_qaoa --qubits 25

# All methods
python3.12 benchmark_VRP_solver.py --methods clarke_wright classical_best hybrid_qaoa
```

Output files:
- `benchmark_output/benchmark_results.json` — full per-instance results
- `benchmark_output/benchmark_comparison.json` — method comparison table
- `benchmark_output/method_comparison.png` — comparison bar chart
- `benchmark_output/<id>_<method>_routes.png` — route plots per instance

### Run the hybrid solver programmatically

```python
import sys
sys.path.insert(0, 'instances/')

from common import challenge_instances
from hybrid_solver import HybridSolver

instances = challenge_instances()
for inst in instances:
    solver = HybridSolver(inst, max_local_qaoa_qubits=25, verbose=True)
    result = solver.solve(run_quantum=True)
    print(result['instance_id'], result['total_dist_final'], result['valid_final'])
```

---

## Challenge Instances

The four small instances from the challenge statement:

| Instance | Customers | Vehicles | Capacity |
|----------|-----------|----------|----------|
| Instance1 | 3 | 2 | 5 |
| Instance2 | 3 | 2 | 2 |
| Instance3 | 6 | 3 | 2 |
| Instance4 | 12 | 4 | 3 |

Results from the hybrid solver (local QAOA native improvement, k=5):

| Instance | Distance | Valid | QAOA Gain |
|----------|----------|-------|-----------|
| Instance1 | 21.7445 | True | 0.0000 |
| Instance2 | 26.1817 | True | 0.0000 |
| Instance3 | 49.4988 | True | 0.0000 |
| Instance4 | 59.5355 | True | 0.1251 |

Instance 4 shows quantum-local improvement of 0.1251 distance units over
the classical seed + repair baseline.

---

## Benchmark Results (27 Synthetic Instances)

All instances remain valid across both methods. `classical_best` consistently
improves over raw Clarke-Wright:

| Instance | CW Distance | Classical Best | Improvement |
|----------|-------------|----------------|-------------|
| A_n32_k5 | 1258.57 | ~1210 | ~3.8% |
| A_n80_k10 | 1243.39 | ~1242 | ~0.1% |
| ... | ... | ... | ... |

Run `benchmark_VRP_solver.py` to get the full table.

---

## Metrics Reported

The benchmark harness writes the following per instance per method:

| Field | Description |
|-------|-------------|
| `method` | Solver method name |
| `total_distance` | Total route distance |
| `runtime_s` | Wall-clock time |
| `valid` | All constraints satisfied |
| `vehicles_used` | Number of routes used |
| `seed_best_method` | Which seed won Stage A |
| `cleanup_gain` | Distance reduction from 2-opt |
| `repair_gain` | Distance reduction from relocate/swap |
| `qaoa_gain` | Distance reduction from quantum local step |
| `quantum_ran` | Whether Stage D executed |
| `qubit_estimates` | Qubit count per route neighborhood |
| `neighborhood_sizes` | k per route |
| `route_details` | Per-route load, distance, sequence |

---

## QAOA Submission Requirements

For the challenge submission, the hybrid solver reports for each instance:

- **Number of qubits**: k^2 where k = floor(sqrt(qubit_budget))
- **Gate operations**: Sum of decomposed gate counts across all neighborhoods
- **Execution time**: Total QAOA simulation time per instance

See the resource usage table cell in `notebooks/qaoa_sweep.ipynb`.

---

## Algorithmic Notes

### Seed Portfolio

Running multiple decompositions and picking the best reduces sensitivity to
initial conditions. Clarke-Wright and k-means often find different good
groupings that sweep misses.



### Inter-route Repair Quality

Repair uses centroid proximity to order route pairs, so geographically close
routes are tested first. This keeps the wall-time practical even with many
routes.

---

## Submission Format

Each solved instance follows the required format:

```
r1: 0, 2, 3, 0
r2: 0, 1, 0
```

The `format_routes_text()` function in `common.py` generates this output.

---

## References

- [qBraid SDK Documentation](https://docs.qbraid.com/v2/)
- [Wikipedia: Vehicle Routing Problem](https://en.wikipedia.org/wiki/Vehicle_routing_problem)
- [Qiskit Optimization Tutorial on VRP](https://qiskit-community.github.io/qiskit-optimization/tutorials/07_examples_vehicle_routing.html)
- Clarke, G. & Wright, J.W. (1964). "Scheduling of Vehicles from a Central Depot"
