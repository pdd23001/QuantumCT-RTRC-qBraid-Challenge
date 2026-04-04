# Agent Knowledge Pack

This folder is the shared context for any agent working in this repository.

Read order:

1. [project-objective.md](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/.agents/project-objective.md)
2. [architecture.md](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/.agents/architecture.md)
3. [runbook.md](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/.agents/runbook.md)

What an agent should understand after reading this folder:

- the hackathon objective and non-negotiable constraints
- why the repository uses a sweep-plus-QAOA hybrid design
- how data moves from instance JSON to validated route files
- which modules own which responsibilities
- how to run, debug, and extend the solver without breaking the submission contract

Current implementation summary:

- Entry point: [src/main.py](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/src/main.py)
- Input contract: [data/schemas/cvrp_instance.schema.json](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/data/schemas/cvrp_instance.schema.json)
- Benchmark data: [data/instances](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/data/instances)
- Output targets: [outputs/routes](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/outputs/routes), [outputs/metrics](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/outputs/metrics), [outputs/plots](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/outputs/plots), [submission](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/submission)

Fast orientation:

- The challenge problem is CVRP with depot `0`, Euclidean distance, fixed vehicle count, and capacity interpreted as max customers per vehicle.
- The solver does not build a global `x[i,j,k]` CVRP QUBO. It decomposes the problem classically, then solves smaller routing-order QUBOs per vehicle cluster.
- Route files are only written after global feasibility checks pass.
- `qbraid_runtime` is scaffolded but requires runtime credentials to actually discover remote devices.
