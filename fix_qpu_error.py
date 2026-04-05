import json
import re

with open('notebooks/Hybrid_CVRP_Unified.ipynb', 'r') as f:
    nb = json.load(f)

new_estimator = """def _get_estimator_and_sampler():
    \"\"\"
    STRICT QUANTUM HARDWARE EXECUTION
    No local simulators permitted.
    \"\"\"
    print("\\n[!] Initializing ACTUAL Quantum Computer Connection...")
    
    try:
        # User defined block
        from qbraid_qiskit.providers import QbraidProvider
        from qiskit.primitives import BackendEstimatorV2
        
        provider = QbraidProvider()
        # Ensure your qBraid environment handles the API footprint or insert it manually!
        backend = provider.get_backend("ionq_aria_1") 
        print(f"[+] Assigned qBraid Target QPU: {backend.name}")
        estimator = BackendEstimatorV2(backend=backend)
        return estimator, None
        
    except ImportError:
        raise Exception("FATAL: You cannot connect to a Quantum Computer without the SDK! Please install it via `pip install qbraid qbraid-qiskit` or execute this notebook natively inside a qBraid Lab!")
    except Exception as e:
        raise Exception(f"FATAL: QPU Hardware allocation rejected! Did you input your credentials? Error: {e}")
"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def _get_estimator_and_sampler():" in source:
            source = re.sub(
                r'def _get_estimator_and_sampler\(\):.*?(?=def _qaoa_optimize_neighborhood)', 
                new_estimator + '\n\n', 
                source, 
                flags=re.DOTALL
            )
        cell['source'] = [line + '\n' for line in source.strip('\n').split('\n')]
        
with open('notebooks/Hybrid_CVRP_Unified.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("QPU Error boundaries strictly secured!")
