# Quantum Courier Challenge

Hybrid CVRP solver for the qBraid / Qiskit hackathon benchmark. The pipeline is:

1. Classical sweep decomposition to assign customers to vehicles.
2. Cluster-local position-based QUBO for each vehicle.
3. QAOA to optimize visit order inside each cluster.
4. Feasibility-aware decoding plus optional 2-opt cleanup.
5. Validated route-file, metrics, and plot generation.

## Why this layout

The challenge instances have a fixed depot at node `0`, Euclidean travel cost, and capacity defined as a maximum number of customers per vehicle. That makes a sweep-plus-QAOA split practical:

- `sweep.py` handles customer assignment.
- `qubo_builder.py` builds a permutation-style routing QUBO per vehicle cluster.
- `qaoa_solver.py` runs QAOA only on the reduced local routing problems.
- `validation.py` ensures submission files are not emitted unless the combined solution is feasible.

## Tested Python setup

This codebase was verified with Python `3.12`. The local machine used during implementation had Python `3.14` as the default `python3`, which does not currently match the installed Qiskit/qBraid package stack cleanly.

Recommended setup:

```bash
python3.12 -m venv .venv
.venv/bin/pip3.12 install -r requirements.txt
```

## Run the solver

Solve one instance:

```bash
.venv/bin/python3.12 -m src.main --instance data/instances/Instance1.json --mode local_statevector
```

Solve all benchmark instances:

```bash
.venv/bin/python3.12 -m src.main --all --mode local_statevector
```

Faster local debug run:

```bash
.venv/bin/python3.12 -m src.main --all --mode local_statevector --maxiter 5 --shots 256
```

Shot-based local simulation:

```bash
.venv/bin/python3.12 -m src.main --instance data/instances/Instance2.json --mode local_aer --maxiter 5 --shots 256
```

Attempt qBraid-managed runtime discovery:

```bash
.venv/bin/python3.12 -m src.main --instance data/instances/Instance2.json --mode qbraid_runtime
```

`qbraid_runtime` currently expects IBM Runtime credentials to be configured so that qBraid's `QiskitRuntimeProvider` can discover devices. If that is not configured, the CLI fails with a clear error instead of silently falling back to a local simulator.

## CLI flags

Useful overrides:

- `--qaoa-reps`
- `--shots`
- `--optimizer`
- `--maxiter`
- `--multi-start-sweep`
- `--backend-name`
- `--qbraid-channel`
- `--disable-2opt`
- `--row-penalty`
- `--col-penalty`

## Inputs

Benchmark JSON files are in [`data/instances/`](data/instances). The schema is in [`data/schemas/cvrp_instance.schema.json`](data/schemas/cvrp_instance.schema.json).

Each instance uses:

- depot `0`
- Euclidean distances
- `vehicles`
- `capacity`
- customer coordinates

Optional per-instance QAOA overrides can be embedded under `qaoa`.

## Outputs

For each valid solve, the pipeline writes:

- [`outputs/routes/`](outputs/routes)
- [`submission/`](submission)
- [`outputs/metrics/`](outputs/metrics)
- [`outputs/plots/`](outputs/plots)

Metrics include:

- chosen sweep decomposition
- cluster sizes
- per-cluster route and distance
- QAOA settings
- qubit and gate-count estimates from the compiled optimal QAOA circuit
- validation status

## Repo structure

Key files:

- [`src/main.py`](src/main.py)
- [`src/sweep.py`](src/sweep.py)
- [`src/qubo_builder.py`](src/qubo_builder.py)
- [`src/qaoa_solver.py`](src/qaoa_solver.py)
- [`src/runtime.py`](src/runtime.py)
- [`src/decoder.py`](src/decoder.py)
- [`src/validation.py`](src/validation.py)

## Verified locally

The following commands were run successfully in the local Python `3.12` environment:

- `.venv/bin/python3.12 -m src.main --all --mode local_statevector --maxiter 5 --shots 256`
- `.venv/bin/python3.12 -m src.main --instance data/instances/Instance2.json --mode local_aer --maxiter 5 --shots 256`

Those runs produced valid submission files for the four benchmark instances and metrics/plot artifacts in `outputs/`.
