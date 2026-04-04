# Agent Guide

Start with [.agents/README.md](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/.agents/README.md).

This repository contains a hybrid CVRP solver for the Quantum Courier Challenge. The current implementation uses:

- classical sweep decomposition for customer-to-vehicle assignment
- cluster-local position-based QUBOs
- QAOA for per-cluster visit ordering
- deterministic validation before writing submission files

Working norms for agents in this repo:

- Use Python `3.12` with the local virtual environment at `.venv/`.
- Prefer the verified commands in [.agents/runbook.md](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/.agents/runbook.md) instead of inventing new entrypoints.
- Do not bypass [src/validation.py](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/src/validation.py) when changing route-generation logic.
- Keep the solution generalized to arbitrary JSON instances matching [data/schemas/cvrp_instance.schema.json](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/data/schemas/cvrp_instance.schema.json).
- Preserve the hybrid requirement: no classical-only replacement for the QAOA stage.

Primary references:

- [.agents/project-objective.md](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/.agents/project-objective.md)
- [.agents/architecture.md](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/.agents/architecture.md)
- [.agents/runbook.md](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/.agents/runbook.md)
