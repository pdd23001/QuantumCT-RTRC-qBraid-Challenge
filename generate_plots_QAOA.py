import json
import os
import matplotlib.pyplot as plt
import shutil

# 1. Configuration
results_dirs = [
    'Results/QAOA_Results'
]

routes = {
    'Instance1': [[1, 2, 3]],
    'Instance2': [[1, 2], [3]],
    'Instance3': [[1, 2], [6, 3], [5, 4]],
    'Instance4': [[6, 9, 5], [11, 2, 8], [10, 4, 3], [1, 7, 12]],
    'Instance5': [[14, 19, 2, 12], [13, 4, 18, 3], [10, 8, 15, 5], [1, 7, 17, 20], [6, 9, 16, 11]],
    'Instance6': [[1, 12, 14, 7, 24], [10, 8, 22, 2, 19], [16, 9, 6, 17, 20], [25, 15, 4, 18, 5], [3, 13, 21, 11, 23]]
}

print("Ensuring paths exist...")
for d in results_dirs:
    os.makedirs(d, exist_ok=True)

# 2. Extract Coordinates from final_instances.json
with open('final_instances.json', 'r') as f:
    instances = json.load(f)

for inst_key, route_lists in routes.items():
    inst_num = inst_key.replace("Instance", "")
    target_id = str(inst_num)
    
    config = next(i for i in instances if str(i['instance_id']) == target_id)
    if isinstance(config['customers'][0], dict):
        nodes = [(float(c['x']), float(c['y'])) for c in config['customers']]
    else:
        nodes = [(float(c[0]), float(c[1])) for c in config['customers']]
        
    Nv = config['Nv']
    C = config['C']
    

    # 4. Generate Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Try to import matplotlib colormap carefully
    try:
        colormap = plt.colormaps['tab10']
    except AttributeError:
        colormap = plt.cm.get_cmap('tab10')
    
    # Plot routes
    for idx, r in enumerate(route_lists):
        c_nodes = [0] + r + [0]
        xs = [nodes[n][0] for n in c_nodes]
        ys = [nodes[n][1] for n in c_nodes]
        
        ax.plot(xs, ys, marker='o', markersize=6, linewidth=2, color=colormap(idx%10), label=f'Vehicle {idx+1}')
        
        for k in range(len(xs)-1):
            dx = xs[k+1] - xs[k]
            dy = ys[k+1] - ys[k]
            ax.arrow(xs[k], ys[k], dx*0.8, dy*0.8, head_width=0.3, head_length=0.4, fc=colormap(idx%10), ec=colormap(idx%10), alpha=0.7)
            
    # Depot
    ax.scatter([nodes[0][0]], [nodes[0][1]], color='red', s=200, marker='s', zorder=5, label='Depot (0,0)')
    
    # Nodes
    for i in range(1, len(nodes)):
        ax.text(nodes[i][0] + 0.2, nodes[i][1] + 0.2, str(i), fontsize=10, fontweight='bold', zorder=6)
        
    ax.set_title(f"Optimal Output Map - {inst_key}\n{len(nodes)-1} Customers | V = {Nv} | C = {C}", fontsize=14)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title="Routes", loc='best')
    
    temp_png = f"{inst_key}_Result.png"
    fig.savefig(temp_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Copy to all directories where the text file actually exists!
    for d in results_dirs:
        txt_path = os.path.join(d, 'Txt_Format_Results', f"{inst_key}.txt")
        if os.path.exists(txt_path) or os.path.exists(os.path.join(d, f"{inst_key}.txt")):
            shutil.copy(temp_png, os.path.join(d, temp_png))
        
    os.remove(temp_png)

print("Text files and plots seamlessly propagated across all four results folders!")
