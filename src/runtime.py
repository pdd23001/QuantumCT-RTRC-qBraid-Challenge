from __future__ import annotations

import os

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from src.config import RuntimeConfig
from src.models import SamplerHandle


class SamplerFactory:
    @staticmethod
    def build(runtime_config: RuntimeConfig, shots: int, seed: int) -> SamplerHandle:
        mode = runtime_config.mode
        if mode == "local_statevector":
            from qiskit.primitives import StatevectorSampler

            return SamplerHandle(
                mode=mode,
                sampler=StatevectorSampler(default_shots=shots, seed=seed),
                provider_name="local",
                backend_name="statevector_sampler",
                metadata={"shots": shots, "seed": seed},
            )

        if mode == "local_aer":
            from qiskit_aer import AerSimulator
            from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

            backend = AerSimulator()
            pass_manager = generate_preset_pass_manager(
                optimization_level=1,
                backend=backend,
            )
            return SamplerHandle(
                mode=mode,
                sampler=AerSamplerV2(default_shots=shots, seed=seed),
                provider_name="local",
                backend_name=backend.name,
                backend=backend,
                transpiler=pass_manager,
                metadata={"shots": shots, "seed": seed},
            )

        if mode == "qbraid_runtime":
            return SamplerFactory._build_qbraid_runtime(runtime_config)

        raise ValueError(f"Unsupported runtime mode: {mode}")

    @staticmethod
    def _build_qbraid_runtime(runtime_config: RuntimeConfig) -> SamplerHandle:
        from qbraid.runtime.ibm import QiskitRuntimeProvider
        from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2

        token = os.getenv(runtime_config.ibm_token_env)
        instance = os.getenv(runtime_config.ibm_instance_env)
        channel = runtime_config.qbraid_channel

        try:
            provider = QiskitRuntimeProvider(
                token=token,
                instance=instance,
                channel=channel,
            )
            devices = list(provider.get_devices())
        except Exception as exc:
            raise RuntimeError(
                "qbraid_runtime mode could not discover Qiskit runtime devices via qBraid. "
                "Configure IBM Runtime credentials or use a local mode."
            ) from exc

        if not devices:
            raise RuntimeError("qbraid_runtime mode found no devices.")

        backend_name = runtime_config.backend_name or getattr(devices[0], "id", None)
        if not backend_name:
            raise RuntimeError("qbraid_runtime mode could not determine a backend id.")

        try:
            backend = provider.runtime_service.backend(backend_name)
            sampler = RuntimeSamplerV2(mode=backend)
            pass_manager = generate_preset_pass_manager(
                optimization_level=1,
                backend=backend,
            )
        except Exception as exc:
            raise RuntimeError(
                f"qbraid_runtime mode discovered backend '{backend_name}' but could not build a sampler."
            ) from exc

        return SamplerHandle(
            mode="qbraid_runtime",
            sampler=sampler,
            provider_name="qbraid",
            backend_name=backend_name,
            backend=backend,
            transpiler=pass_manager,
            metadata={"device_count": len(devices)},
        )
