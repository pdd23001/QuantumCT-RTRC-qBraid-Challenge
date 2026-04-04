from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping


@dataclass(frozen=True)
class SweepConfig:
    multi_start: int = 16
    directions: tuple[str, ...] = ("clockwise", "counterclockwise")
    balance_penalty_weight: float = 0.25


@dataclass(frozen=True)
class QAOAConfig:
    reps: int = 1
    optimizer: str = "COBYLA"
    maxiter: int = 100
    shots: int = 2048
    seed: int = 42


@dataclass(frozen=True)
class PenaltyConfig:
    row_exactly_one: float = 20.0
    col_exactly_one: float = 20.0


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str = "local_statevector"
    backend_name: str | None = None
    qbraid_channel: str | None = None
    qbraid_api_key_env: str = "QBRAID_API_KEY"
    ibm_token_env: str = "QISKIT_IBM_TOKEN"
    ibm_instance_env: str = "QISKIT_IBM_INSTANCE"


@dataclass(frozen=True)
class SolverConfig:
    sweep: SweepConfig = SweepConfig()
    qaoa: QAOAConfig = QAOAConfig()
    penalties: PenaltyConfig = PenaltyConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    enable_2opt: bool = True

    def with_instance_overrides(self, overrides: Mapping[str, Any] | None) -> "SolverConfig":
        if not overrides:
            return self

        qaoa = self.qaoa
        if "reps" in overrides:
            qaoa = replace(qaoa, reps=int(overrides["reps"]))
        if "optimizer" in overrides:
            qaoa = replace(qaoa, optimizer=str(overrides["optimizer"]).upper())
        if "maxiter" in overrides:
            qaoa = replace(qaoa, maxiter=int(overrides["maxiter"]))
        if "shots" in overrides:
            qaoa = replace(qaoa, shots=int(overrides["shots"]))

        runtime = self.runtime
        if "sampler_mode" in overrides:
            runtime = replace(runtime, mode=str(overrides["sampler_mode"]))

        return replace(self, qaoa=qaoa, runtime=runtime)
