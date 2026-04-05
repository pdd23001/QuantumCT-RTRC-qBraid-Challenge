import json

with open('notebooks/Hybrid_CVRP_Unified.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # Lower qubit simulation ceilings to avoid RAM throttling
        if 'max_local_qaoa_qubits=25' in source:
            source = source.replace('max_local_qaoa_qubits=25', 'max_local_qaoa_qubits=15')
        cell['source'] = [line + '\n' for line in source.strip('\n').split('\n')]
        
with open('notebooks/Hybrid_CVRP_Unified.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Lowered simulator ceiling fixed")
