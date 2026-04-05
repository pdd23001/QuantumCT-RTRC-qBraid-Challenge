import json
import re

with open('notebooks/Hybrid_CVRP_Unified.ipynb', 'r') as f:
    nb = json.load(f)

new_estimator = """def _get_estimator_and_sampler():
    \"\"\"
    STRICT QUANTUM HARDWARE EXECUTION
    No local simulators permitted. Defers to qBraid API or IBM Quantum.
    \"\"\"
    print("\\n[!] Initializing ACTUAL Quantum Computer Connection...")
    
    # User can place tokens here or in their environment
    try:
        from qbraid_qiskit.providers import QbraidProvider
        from qiskit.primitives import BackendEstimatorV2
        
        provider = QbraidProvider(qbraid_api_key="INSERT_QBRAID_API_KEY_HERE")
        # Example hardware allocation: Rigetti Aspen-M-3 or IonQ Aria
        backend = provider.get_backend("ionq_aria_1") 
        print(f"[+] Assigned qBraid Target QPU: {backend.name}")
        estimator = BackendEstimatorV2(backend=backend)
        return estimator, None
    except ImportError:
        pass
        
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as RuntimeEstimator
        service = QiskitRuntimeService(channel="ibm_quantum", token="INSERT_IBM_TOKEN_HERE")
        backend = service.least_busy(operational=True, simulator=False)
        print(f"[+] Assigned IBM Runtime QPU: {backend.name}")
        estimator = RuntimeEstimator(mode=backend)
        return estimator, None
    except ImportError:
        raise ImportError("No Hardware SDK found. Install qbraid-sdk or qiskit-ibm-runtime.")
"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # Replace the def _get_estimator_and_sampler(): ... function completely using regex
        if "def _get_estimator_and_sampler():" in source:
            source = re.sub(
                r'def _get_estimator_and_sampler\(\):.*?(?=def _qaoa_optimize_neighborhood)', 
                new_estimator + '\n\n', 
                source, 
                flags=re.DOTALL
            )
            # Update Transpile logic to target Estimator Backend instead of Aer:
            source = re.sub(r'if _aer_available:.*?(?=estimator, sampler)', 
                            '', source, flags=re.DOTALL)
            source = source.replace("estimator, sampler = _get_estimator_and_sampler()", "estimator, sampler = _get_estimator_and_sampler()\n    from qiskit import transpile\n    try:\n        ansatz = transpile(ansatz, backend=estimator.backend, optimization_level=2)\n    except:\n        ansatz = transpile(ansatz, basis_gates=['rx', 'ry', 'rz', 'rzz', 'cx', 'id'], optimization_level=2)\n")
        cell['source'] = [line + '\n' for line in source.strip('\n').split('\n')]
        
with open('notebooks/Hybrid_CVRP_Unified.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("qBraid QPU Implementation loaded into Jupyter cells!")
