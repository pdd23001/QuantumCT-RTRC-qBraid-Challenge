# Runbook

## Environment

Use Python `3.12`.

Recommended setup:

```bash
python3.12 -m venv .venv
.venv/bin/pip3.12 install -r requirements.txt
```

Reason: the local machine that produced the current code had Python `3.14` as the default interpreter, but the Qiskit/qBraid stack was verified in `3.12`.

## Primary commands

Solve one instance:

```bash
.venv/bin/python3.12 -m src.main --instance data/instances/Instance1.json --mode local_statevector
```

Solve all benchmark instances:

```bash
.venv/bin/python3.12 -m src.main --all --mode local_statevector
```

Faster local smoke test:

```bash
.venv/bin/python3.12 -m src.main --all --mode local_statevector --maxiter 5 --shots 256
```

Shot-based local simulation:

```bash
.venv/bin/python3.12 -m src.main --instance data/instances/Instance2.json --mode local_aer --maxiter 5 --shots 256
```

Attempt remote runtime discovery:

```bash
.venv/bin/python3.12 -m src.main --instance data/instances/Instance2.json --mode qbraid_runtime
```

## Important CLI flags

- `--mode local_statevector|local_aer|qbraid_runtime`
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

## What to inspect after a run

Routes:

- [outputs/routes](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/outputs/routes)
- [submission](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/submission)

Metrics:

- [outputs/metrics](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/outputs/metrics)

Plots:

- [outputs/plots](/Users/pranaykakkar/Hackathons/QuantumCT-RTRC-qBraid-Challenge/outputs/plots)

## Expected output contract

Each submission file must look like:

```text
r1: 0, 2, 3, 0
r2: 0, 1, 0
```

Route files are only supposed to be emitted after global validation succeeds.

## Known operational notes

- `qbraid_runtime` is not a local fallback mode. It is expected to fail clearly if qBraid or IBM Runtime credentials are not configured.
- Larger cluster sizes increase runtime sharply because the number of qubits is `m^2` for a cluster with `m` customers.
- For local debugging, reduce `--maxiter` and `--shots` before changing the algorithm.
- Empty vehicles are valid and currently render as `r#: 0, 0`.

## If an agent is modifying the solver

- Keep schema compatibility unless the user explicitly asks to change the instance contract.
- Re-run at least one local command after changing the solver path.
- If route generation changes, inspect both the route file and metrics JSON.
- If runtime code changes, check both `local_statevector` and `local_aer` before treating the edit as safe.
