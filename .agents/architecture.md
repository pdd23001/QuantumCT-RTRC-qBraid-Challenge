# Architecture

## End-to-end pipeline

The current execution flow is:

1. Load instance JSON.
2. Validate against schema and semantic rules.
3. Build Euclidean distance matrix and customer polar coordinates.
4. Run multi-start sweep decomposition.
5. For each vehicle cluster:
   - build a position-based routing QUBO
   - run QAOA if the cluster has more than one customer
   - decode or repair the sampled assignment into a concrete route
   - optionally improve the route with 2-opt
6. Validate the combined set of routes globally.
7. Write route files, metrics, and plots if valid.

## Module ownership

### Entry point and orchestration

- [src/main.py](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/src/main.py)
  - CLI parsing
  - config assembly
  - per-instance solve orchestration
  - output writing and summary rendering

### Data models and config

- [src/models.py](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/src/models.py)
  - typed dataclasses for instances, sweep candidates, cluster problems, sampler handles, and solutions
- [src/config.py](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/src/config.py)
  - solver, sweep, QAOA, penalty, and runtime config
  - per-instance `qaoa` override handling

### Input/output and validation

- [src/io_utils.py](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/src/io_utils.py)
  - instance discovery
  - JSON loading
  - route/metrics writing
  - submission sync
- [src/validation.py](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/src/validation.py)
  - schema validation
  - semantic instance checks
  - final route feasibility checks

### Geometry and classical decomposition

- [src/geometry.py](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/src/geometry.py)
  - Euclidean distance matrix
  - depot-relative polar coordinates
  - route-distance utilities
  - nearest-neighbor scoring helper
- [src/sweep.py](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/src/sweep.py)
  - clockwise and counterclockwise sweep
  - multi-start offsets
  - capacity-aware partitioning
  - cheap scoring to choose the best decomposition before QAOA

### QUBO and quantum solve

- [src/qubo_builder.py](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/src/qubo_builder.py)
  - builds the position-based QUBO with `docplex`
  - converts it to `QuadraticProgram` with `from_docplex_mp`
- [src/runtime.py](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/src/runtime.py)
  - constructs the sampler object for `local_statevector`, `local_aer`, or `qbraid_runtime`
- [src/qaoa_solver.py](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/src/qaoa_solver.py)
  - builds `QAOA`
  - wraps it in `MinimumEigenOptimizer`
  - solves the cluster-local QUBO
  - estimates qubits and gate counts from the compiled optimal circuit

### Decoding and postprocess

- [src/decoder.py](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/src/decoder.py)
  - converts sampled binary assignments into a route
  - accepts exact permutation matrices first
  - repairs near-feasible samples with a linear assignment projection
- [src/postprocess.py](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/src/postprocess.py)
  - optional 2-opt cleanup inside a route

### Reporting

- [src/metrics.py](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/src/metrics.py)
  - aggregates run metadata
- [src/visualize.py](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/src/visualize.py)
  - route plots

## Algorithm details

### Sweep stage

Each customer is ranked by polar angle around the depot. The solver tries multiple circular starting offsets and both directions, then partitions the ordered list into vehicle clusters of size at most `capacity`.

The best sweep candidate is chosen with a cheap classical score based on:

- nearest-neighbor route estimates per cluster
- a mild load-balance penalty

### QAOA encoding

For a cluster of `m` customers, the solver uses `m x m` binary variables:

`y[u, p] = 1` if customer `u` is visited at position `p`

This is a permutation-matrix encoding, not a global edge-variable encoding. The QUBO includes:

- depot-to-first-customer cost
- consecutive customer-to-customer cost
- last-customer-to-depot cost
- row exactly-one penalties
- column exactly-one penalties

### Decoding logic

The QAOA result may include imperfect samples. Decoding therefore:

1. tries to find an exact permutation matrix among samples
2. otherwise repairs the best sample into a valid assignment
3. returns a depot-anchored route

## Design boundaries

- The solver is intentionally modular. If an agent changes one stage, it should keep interfaces between modules stable.
- The final route representation is route-list based, but it can be interpreted as a per-vehicle adjacency matrix after decoding if needed.
- Empty vehicles are allowed and are represented as `0, 0`.

## Places an agent is most likely to extend

- stronger sweep scoring or boundary-customer swap logic
- warm-start QAOA
- better decoder heuristics
- richer qBraid runtime backend selection
- stronger metrics or visualization
