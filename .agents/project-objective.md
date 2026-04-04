# Project Objective

## What this repository is trying to do

Build a generalized quantum-classical hybrid solver for the Quantum Courier Challenge benchmark instances and any other instances that share the same JSON shape.

The solver must:

- accept a depot at node `0`
- minimize total Euclidean travel distance
- respect a fixed number of vehicles
- respect capacity defined as a maximum number of customers per vehicle
- output one route file per solved instance in the required submission format
- remain hybrid or quantum, not classical-only

## Why the hybrid split exists

The full CVRP is too large and awkward to encode directly as one global QUBO for the first implementation. This repository uses a decomposition that matches the challenge structure:

1. Classical stage: assign customers to vehicles with capacity-aware sweep around the depot.
2. Quantum stage: solve the visit order inside each assigned cluster with QAOA.

This reduces the quantum problem size and removes global capacity handling from the QAOA layer.

## Problem formulation used here

The global CVRP is decomposed into:

`CVRP = customer assignment + per-vehicle routing order`

After sweep assigns a cluster `S_k` to vehicle `k`, the remaining subproblem is:

`0 -> permutation(S_k) -> 0`

That is a small TSP-like routing-order problem with fixed depot start and end.

## Non-negotiable repository constraints

- Keep the code generalized. Do not hardcode solver logic to the four benchmark instances.
- Preserve the exact route-file format:
  - `r1: 0, ..., 0`
  - `r2: 0, ..., 0`
- Never silently write invalid submissions.
- Keep QAOA or a quantum-classical hybrid path in the solution. Do not replace the core with a purely classical optimizer.
- When touching runtime code, prefer the installed Qiskit/qBraid APIs rather than guessed wrappers.

## Inputs and outputs

Inputs:

- JSON instance files in [data/instances](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/data/instances)
- schema in [data/schemas/cvrp_instance.schema.json](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/data/schemas/cvrp_instance.schema.json)

Outputs for each valid solve:

- route file in [outputs/routes](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/outputs/routes)
- synced submission file in [submission](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/submission)
- metrics JSON in [outputs/metrics](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/outputs/metrics)
- route plot in [outputs/plots](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/outputs/plots)

## Current verified status

The current solver has already been run successfully in local simulation modes:

- `local_statevector`
- `local_aer`

The `qbraid_runtime` branch is structured and fails clearly when credentials or runtime configuration are missing.
