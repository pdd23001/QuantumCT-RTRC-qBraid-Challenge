import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import matplotlib.pyplot as plt


def euclidean(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    return math.hypot(a['x'] - b['x'], a['y'] - b['y'])


def build_distance(customers: Dict[int, Dict[str, Any]]) -> Dict[Tuple[int, int], float]:
    dist = {}
    for i in customers:
        for j in customers:
            dist[(i, j)] = euclidean(customers[i], customers[j])
    return dist


def route_distance(path: List[int], dist: Dict[Tuple[int, int], float]) -> float:
    return sum(dist[(path[i], path[i + 1])] for i in range(len(path) - 1))


def route_load(path: List[int], customers: Dict[int, Dict[str, Any]]) -> int:
    return sum(customers[cid]['demand'] for cid in path if cid != 0)


def is_start_customer(path: List[int], cid: int) -> bool:
    return len(path) >= 3 and path[1] == cid


def is_end_customer(path: List[int], cid: int) -> bool:
    return len(path) >= 3 and path[-2] == cid


def merge_paths(path1: List[int], i: int, path2: List[int], j: int) -> List[int]:
    seq1 = path1[1:-1]
    seq2 = path2[1:-1]
    if seq1[0] == i:
        seq1 = list(reversed(seq1))
    if seq2[-1] == j:
        seq2 = list(reversed(seq2))
    return [0] + seq1 + seq2 + [0]


def cheapest_insertion(path: List[int], customer_id: int, dist: Dict[Tuple[int, int], float]) -> Tuple[float, int]:
    best_delta = float('inf')
    best_pos = -1
    for pos in range(len(path) - 1):
        a = path[pos]
        b = path[pos + 1]
        delta = dist[(a, customer_id)] + dist[(customer_id, b)] - dist[(a, b)]
        if delta < best_delta:
            best_delta = delta
            best_pos = pos + 1
    return best_delta, best_pos


def try_dissolve_route(
    routes: List[Dict[str, Any]],
    idx: int,
    customers: Dict[int, Dict[str, Any]],
    capacity: int,
    dist: Dict[Tuple[int, int], float],
) -> Optional[List[Dict[str, Any]]]:
    victim = routes[idx]
    victims_customers = [cid for cid in victim['path'] if cid != 0]
    victims_customers.sort(key=lambda cid: (customers[cid]['demand'], len(victims_customers)), reverse=True)

    new_routes = []
    for k, route in enumerate(routes):
        if k == idx:
            continue
        new_routes.append({'path': route['path'][:], 'load': route['load']})

    for cid in victims_customers:
        demand = customers[cid]['demand']
        best_choice = None
        for r_idx, route in enumerate(new_routes):
            if route['load'] + demand > capacity:
                continue
            delta, pos = cheapest_insertion(route['path'], cid, dist)
            slack_after = capacity - (route['load'] + demand)
            key = (delta, slack_after, len(route['path']))
            if best_choice is None or key < best_choice[0]:
                best_choice = (key, r_idx, pos)
        if best_choice is None:
            return None
        _, r_idx, pos = best_choice
        new_routes[r_idx]['path'].insert(pos, cid)
        new_routes[r_idx]['load'] += demand

    return new_routes


def repair_route_count(
    routes: List[Dict[str, Any]],
    customers: Dict[int, Dict[str, Any]],
    capacity: int,
    max_vehicles: int,
    dist: Dict[Tuple[int, int], float],
) -> Tuple[List[Dict[str, Any]], bool]:
    routes = [{'path': r['path'][:], 'load': r['load']} for r in routes]

    improved = True
    while len(routes) > max_vehicles and improved:
        improved = False
        candidate_indices = sorted(
            range(len(routes)),
            key=lambda i: (routes[i]['load'], len(routes[i]['path']))
        )
        for idx in candidate_indices:
            repaired = try_dissolve_route(routes, idx, customers, capacity, dist)
            if repaired is not None:
                routes = repaired
                improved = True
                break

    return routes, len(routes) <= max_vehicles


def clarke_wright_with_repair(instance: Dict[str, Any]) -> Dict[str, Any]:
    customers = {c['customer_id']: c for c in instance['customers']}
    customer_ids = [cid for cid in customers if cid != 0]
    capacity = instance['C']
    max_vehicles = instance['Nv']
    dist = build_distance(customers)

    routes: Dict[int, Dict[str, Any]] = {}
    route_of_customer: Dict[int, int] = {}
    for cid in customer_ids:
        routes[cid] = {'path': [0, cid, 0], 'load': customers[cid]['demand']}
        route_of_customer[cid] = cid

    savings = []
    for i in customer_ids:
        for j in customer_ids:
            if i < j:
                s = dist[(0, i)] + dist[(0, j)] - dist[(i, j)]
                savings.append((s, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))

    for _, i, j in savings:
        ri = route_of_customer[i]
        rj = route_of_customer[j]
        if ri == rj:
            continue

        path_i = routes[ri]['path']
        path_j = routes[rj]['path']
        if not ((is_start_customer(path_i, i) or is_end_customer(path_i, i)) and (is_start_customer(path_j, j) or is_end_customer(path_j, j))):
            continue

        new_load = routes[ri]['load'] + routes[rj]['load']
        if new_load > capacity:
            continue

        new_path = merge_paths(path_i, i, path_j, j)
        keep = min(ri, rj)
        drop = max(ri, rj)
        routes[keep] = {'path': new_path, 'load': new_load}
        for cid in new_path:
            if cid != 0:
                route_of_customer[cid] = keep
        del routes[drop]

    initial_routes = list(routes.values())
    repaired_routes, repaired_ok = repair_route_count(initial_routes, customers, capacity, max_vehicles, dist)

    for route in repaired_routes:
        route['distance'] = route_distance(route['path'], dist)

    total_distance = sum(route['distance'] for route in repaired_routes)
    loads_valid = all(route['load'] <= capacity for route in repaired_routes)

    return {
        'instance_id': instance['instance_id'],
        'Nv': max_vehicles,
        'C': capacity,
        'initial_num_routes': len(initial_routes),
        'num_routes': len(repaired_routes),
        'vehicle_feasible': repaired_ok,
        'capacity_feasible': loads_valid,
        'valid': repaired_ok and loads_valid,
        'total_distance': total_distance,
        'routes': repaired_routes,
    }


def plot_solution(instance: Dict[str, Any], result: Dict[str, Any], output_dir: Path) -> Path:
    customers = {c['customer_id']: c for c in instance['customers']}
    depot = customers[0]

    plt.figure(figsize=(9, 7))
    plt.scatter([depot['x']], [depot['y']], marker='s', s=120)
    plt.annotate('Depot', (depot['x'], depot['y']), xytext=(6, 6), textcoords='offset points')

    for cid, cust in customers.items():
        if cid == 0:
            continue
        plt.scatter([cust['x']], [cust['y']], s=45)
        plt.annotate(f"{cid}(d={cust['demand']})", (cust['x'], cust['y']), xytext=(4, 4), textcoords='offset points', fontsize=8)

    for idx, route in enumerate(result['routes'], start=1):
        xs = [customers[cid]['x'] for cid in route['path']]
        ys = [customers[cid]['y'] for cid in route['path']]
        plt.plot(xs, ys, linewidth=1.8, label=f"R{idx}: load={route['load']}")

    plt.title(
        f"{result['instance_id']} | routes={result['num_routes']}/{result['Nv']} | "
        f"dist={result['total_distance']:.2f}"
    )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(fontsize=8, loc='best')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    output_path = output_dir / f"{result['instance_id']}_clarke_wright_repair.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    input_path = script_dir / 'demand2_instances.json'
    if not input_path.exists():
        raise SystemExit(f'Input JSON not found: {input_path}')

    with input_path.open('r', encoding='utf-8') as f:
        instances = json.load(f)

    output_dir = script_dir / 'clarke_wright_repair_output'
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for instance in instances:
        result = clarke_wright_with_repair(instance)
        plot_solution(instance, result, output_dir)
        results.append(result)
        print(
            f"{result['instance_id']}: initial_routes={result['initial_num_routes']} -> "
            f"final_routes={result['num_routes']}/{result['Nv']}, "
            f"valid={result['valid']}, total_distance={result['total_distance']:.3f}"
        )

    output_json = output_dir / 'clarke_wright_repair_results.json'
    with output_json.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f'Saved results to: {output_json}')
    for result in results:
        with open (output_dir / f"output_txt/{result['instance_id']}.txt", 'w') as f:
            for idx, route in enumerate(result['routes'], start=1):
                route_str = ' -> '.join(str(cid) for cid in route['path'])
                f.write(f"Route {idx}: {route_str}\n")

if __name__ == '__main__':
    main()
