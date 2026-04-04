import json
import os
import re

def strip_local_imports(code_text):
    lines = code_text.split('\n')
    out = []
    for line in lines:
        if line.startswith('from common import') or \
           line.startswith('from decomposition import') or \
           line.startswith('from local_search import') or \
           line.startswith('from repair import') or \
           line.startswith('from quantum_improve import') or \
           line.startswith('from hybrid_solver import'):
            continue
        out.append(line)
    return '\n'.join(out)

def read_file(path):
    with open(path, 'r') as f:
        return strip_local_imports(f.read())

files = {
    'common': read_file('instances/common.py'),
    'decomposition': read_file('instances/decomposition.py'),
    'local_search': read_file('instances/local_search.py'),
    'repair': read_file('instances/repair.py'),
    'quantum_improve': read_file('instances/quantum_improve.py'),
    'hybrid_solver': read_file('instances/hybrid_solver.py')
}

cells = []
def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + '\n' for line in text.strip().split('\n')]})

def add_code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + '\n' for line in text.split('\n')]})

add_md("""
# Hybrid Quantum-Classical CVRP Solver (Unified)
## Yale Quantum Courier Challenge 2026
This notebook combines the entire multi-stage solver architecture into a single, standalone runable notebook.
""")

add_md("## Step 0: Standard Library Imports")
add_code("""import time
import math
import itertools
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
""")

add_md("## Step 1: Common Utilities (Geometry & Distances)")
add_code(files['common'])

add_md("## Step 2: Decomposition (Seed Generation)")
add_code(files['decomposition'])

add_md("## Step 3: Local Search (2-Opt Cleanup)")
add_code(files['local_search'])

add_md("## Step 4: Inter-Route Repair (Relocate & Swap)")
add_code(files['repair'])

add_md("## Step 5: Quantum Local Improvement (QAOA)")
add_code(files['quantum_improve'])

add_md("## Step 6: Hybrid Solver Orchestration")
add_code(files['hybrid_solver'])

add_md("## Step 7: Pipeline Evaluation & Execution\nHere we run the solver on all 4 challenge instances, showcasing the strict QAOA optimization loop.")
add_code("""
# Instantiate and solve
instances = challenge_instances()
results = {}

for inst in instances:
    print(f"\\n{'='*50}")
    print(f"Solving {inst['instance_id']}")
    print(f"{'='*50}")
    # Force QAOA for neighborhoods by setting a suitable budget 
    # (By default it dynamically applies k^2 = budget)
    solver = HybridSolver(inst, max_local_qaoa_qubits=25, verbose=True)
    res = solver.solve(run_quantum=True)
    results[inst['instance_id']] = res
    print(f"\\nFINAL VALID: {res['valid_final']} | DIST: {res['total_dist_final']:.4f}")

# Display standard submittable output format
print("\\n\\n" + "="*50)
print("FINAL CHALLENGE SUBMISSION FORMAT")
print("="*50)
for iid, r in results.items():
    print(format_routes_text(r['routes_final'], instance_id=iid))
    print()
""")

add_md("## Step 8: Visualization")
add_code("""
def plot_routes(inst, routes, title=''):
    nodes = inst['nodes']
    depot = nodes[0]
    cust_xy = nodes[1:]
    D_plot = build_distance_matrix(nodes)
    total_d = total_solution_distance(routes, D_plot)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(*depot, c='black', marker='X', s=200, zorder=5, label='Depot')
    ax.text(depot[0]+0.3, depot[1]+0.3, '0', fontsize=10)

    cx = [p[0] for p in cust_xy]
    cy = [p[1] for p in cust_xy]
    ax.scatter(cx, cy, c='steelblue', s=70, zorder=4)
    for idx, (x, y) in enumerate(cust_xy, start=1):
        ax.text(x+0.3, y+0.3, str(idx), fontsize=8)

    colors = plt.cm.tab10.colors
    for ridx, route in enumerate(routes, start=1):
        pts = np.array([depot] + [cust_xy[c-1] for c in route] + [depot])
        ax.plot(pts[:,0], pts[:,1], linewidth=2.2,
                color=colors[ridx % len(colors)], label=f'R{ridx}: {route}')

    ax.set_title(f"{inst['instance_id']} -- {title}\\ndist={total_d:.4f}", fontsize=13)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

for iid, r in results.items():
    inst_plot = next(i for i in instances if i['instance_id'] == iid)
    plot_routes(inst_plot, r['routes_final'], title='Hybrid QAOA Solver')
""")

nb = {
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.0"
  }
 },
 "cells": cells
}

with open('notebooks/Hybrid_CVRP_Unified.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Created notebooks/Hybrid_CVRP_Unified.ipynb successfully.")
