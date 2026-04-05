import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt


INPUT_JSON = Path('demand2_instances.json')
OUTPUT_DIR = Path('ga_best_of_five_output')
OUTPUT_DIR.mkdir(exist_ok=True)

TRIALS = 5
POP_SIZE = 80
GENERATIONS = 250
ELITE_COUNT = 8
TOURNAMENT_SIZE = 4
MUTATION_RATE = 0.20
SEED_BASE = 20260405


@dataclass
class TrialResult:
    trial: int
    seed: int
    total_distance: float
    rounded_distance: int
    runtime_sec: float
    num_routes: int
    valid: bool
    routes: List[Dict[str, Any]]


def euclidean(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    return math.hypot(a['x'] - b['x'], a['y'] - b['y'])


def build_distance_matrix(customers: Dict[int, Dict[str, Any]]) -> Dict[Tuple[int, int], float]:
    dist = {}
    ids = list(customers.keys())
    for i in ids:
        for j in ids:
            dist[(i, j)] = euclidean(customers[i], customers[j])
    return dist


def decode_routes(permutation: List[int], customers: Dict[int, Dict[str, Any]], capacity: int) -> List[Dict[str, Any]]:
    routes: List[Dict[str, Any]] = []
    current: List[int] = [0]
    current_load = 0

    for cid in permutation:
        demand = customers[cid]['demand']
        if current_load + demand <= capacity:
            current.append(cid)
            current_load += demand
        else:
            current.append(0)
            routes.append({'path': current, 'load': current_load})
            current = [0, cid]
            current_load = demand

    current.append(0)
    routes.append({'path': current, 'load': current_load})
    return routes


def route_distance(path: List[int], dist: Dict[Tuple[int, int], float]) -> float:
    total = 0.0
    for i in range(len(path) - 1):
        total += dist[(path[i], path[i + 1])]
    return total


def evaluate_permutation(
    permutation: List[int],
    customers: Dict[int, Dict[str, Any]],
    dist: Dict[Tuple[int, int], float],
    capacity: int,
    max_vehicles: int,
) -> Tuple[float, List[Dict[str, Any]], bool]:
    routes = decode_routes(permutation, customers, capacity)

    total_distance = 0.0
    for route in routes:
        total_distance += route_distance(route['path'], dist)

    valid_capacity = all(route['load'] <= capacity for route in routes)
    valid_vehicle_count = len(routes) <= max_vehicles
    valid = valid_capacity and valid_vehicle_count

    # Large penalties for infeasibility, smaller for using extra vehicles.
    penalty = 0.0
    if not valid_capacity:
        penalty += 1_000_000
    if not valid_vehicle_count:
        penalty += 100_000 * (len(routes) - max_vehicles)

    return total_distance + penalty, routes, valid


def random_permutation(customer_ids: List[int], rng: random.Random) -> List[int]:
    p = customer_ids[:]
    rng.shuffle(p)
    return p


def ordered_crossover(parent1: List[int], parent2: List[int], rng: random.Random) -> List[int]:
    n = len(parent1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[a:b + 1] = parent1[a:b + 1]

    p2_items = [gene for gene in parent2 if gene not in child]
    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = p2_items[idx]
            idx += 1
    return child


def mutate_swap(permutation: List[int], rng: random.Random, mutation_rate: float) -> None:
    if rng.random() < mutation_rate:
        i, j = rng.sample(range(len(permutation)), 2)
        permutation[i], permutation[j] = permutation[j], permutation[i]


def tournament_select(population: List[List[int]], scored: List[Tuple[float, int]], rng: random.Random) -> List[int]:
    contenders = rng.sample(scored, TOURNAMENT_SIZE)
    contenders.sort(key=lambda x: x[0])
    return population[contenders[0][1]][:]


def nearest_neighbor_seed(customer_ids: List[int], dist: Dict[Tuple[int, int], float]) -> List[int]:
    remaining = set(customer_ids)
    current = 0
    order = []
    while remaining:
        nxt = min(remaining, key=lambda c: dist[(current, c)])
        order.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return order


def run_ga_for_instance(instance: Dict[str, Any], trial_idx: int) -> TrialResult:
    customers = {c['customer_id']: c for c in instance['customers']}
    customer_ids = [cid for cid in customers if cid != 0]
    capacity = int(instance['C'])
    max_vehicles = int(instance['Nv'])
    dist = build_distance_matrix(customers)

    seed = SEED_BASE + trial_idx * 1000 + sum(ord(ch) for ch in instance['instance_id'])
    rng = random.Random(seed)

    start = time.perf_counter()

    population: List[List[int]] = []
    population.append(nearest_neighbor_seed(customer_ids, dist))
    for _ in range(POP_SIZE - 1):
        population.append(random_permutation(customer_ids, rng))

    best_perm = None
    best_score = float('inf')
    best_routes = None
    best_valid = False

    for _ in range(GENERATIONS):
        scored: List[Tuple[float, int]] = []
        for idx, perm in enumerate(population):
            score, routes, valid = evaluate_permutation(perm, customers, dist, capacity, max_vehicles)
            scored.append((score, idx))
            if score < best_score:
                best_score = score
                best_perm = perm[:]
                best_routes = routes
                best_valid = valid

        scored.sort(key=lambda x: x[0])
        next_population: List[List[int]] = []

        for _, idx in scored[:ELITE_COUNT]:
            next_population.append(population[idx][:])

        while len(next_population) < POP_SIZE:
            p1 = tournament_select(population, scored, rng)
            p2 = tournament_select(population, scored, rng)
            child = ordered_crossover(p1, p2, rng)
            mutate_swap(child, rng, MUTATION_RATE)
            next_population.append(child)

        population = next_population

    # Re-evaluate best perm without penalty so reported distance is clean.
    _, final_routes, final_valid = evaluate_permutation(best_perm, customers, dist, capacity, max_vehicles)
    final_distance = sum(route_distance(r['path'], dist) for r in final_routes)
    runtime_sec = time.perf_counter() - start

    for route in final_routes:
        route['distance'] = route_distance(route['path'], dist)

    return TrialResult(
        trial=trial_idx,
        seed=seed,
        total_distance=final_distance,
        rounded_distance=round(final_distance),
        runtime_sec=runtime_sec,
        num_routes=len(final_routes),
        valid=final_valid,
        routes=final_routes,
    )


def plot_instance(instance: Dict[str, Any], trial_result: TrialResult, output_path: Path) -> None:
    customers = {c['customer_id']: c for c in instance['customers']}
    depot = customers[0]

    plt.figure(figsize=(9, 7))

    plt.scatter([depot['x']], [depot['y']], marker='s', s=150, label='Depot')

    for cid, c in customers.items():
        if cid == 0:
            continue
        plt.scatter(c['x'], c['y'], s=55)
        plt.text(c['x'] + 0.15, c['y'] + 0.15, f"{cid}(d={c['demand']})", fontsize=8)

    for idx, route in enumerate(trial_result.routes, start=1):
        path = route['path']
        xs = [customers[cid]['x'] for cid in path]
        ys = [customers[cid]['y'] for cid in path]
        plt.plot(xs, ys, linewidth=1.8, label=f"R{idx} load={route['load']}")

    title = (
        f"{instance['instance_id']} | GA best of {TRIALS} trials\n"
        f"best trial={trial_result.trial} | dist={trial_result.rounded_distance} "
        f"| routes={trial_result.num_routes}/{instance['Nv']} | valid={trial_result.valid}"
    )
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    with INPUT_JSON.open('r', encoding='utf-8') as f:
        instances = json.load(f)

    summary = []
    csv_rows = []

    for instance in instances:
        trial_results = []
        for trial in range(1, TRIALS + 1):
            result = run_ga_for_instance(instance, trial)
            trial_results.append(result)

        # Prefer valid solutions first, then lowest distance.
        best_trial = min(
            trial_results,
            key=lambda r: (not r.valid, r.total_distance, r.num_routes, r.runtime_sec)
        )

        plot_path = OUTPUT_DIR / f"{instance['instance_id']}_ga_best_trial.png"
        plot_instance(instance, best_trial, plot_path)

        row = {
            'instance_id': instance['instance_id'],
            'best_trial': best_trial.trial,
            'best_seed': best_trial.seed,
            'best_distance': best_trial.total_distance,
            'best_distance_rounded': best_trial.rounded_distance,
            'best_runtime_sec': best_trial.runtime_sec,
            'best_num_routes': best_trial.num_routes,
            'vehicle_limit': instance['Nv'],
            'best_valid': best_trial.valid,
            'plot_file': str(plot_path.name),
            'trial_results': [
                {
                    'trial': r.trial,
                    'seed': r.seed,
                    'distance': r.total_distance,
                    'rounded_distance': r.rounded_distance,
                    'runtime_sec': r.runtime_sec,
                    'num_routes': r.num_routes,
                    'valid': r.valid,
                }
                for r in trial_results
            ],
            'best_routes': best_trial.routes,
        }
        summary.append(row)

        csv_rows.append([
            instance['instance_id'],
            *[r.rounded_distance for r in trial_results],
            round(sum(r.total_distance for r in trial_results) / len(trial_results), 3),
            best_trial.trial,
            best_trial.rounded_distance,
            best_trial.num_routes,
            best_trial.valid,
        ])

    with (OUTPUT_DIR / 'ga_best_of_five_results.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / 'ga_best_of_five_table.txt', "w", encoding="utf-8") as f:
        # header
        f.write("Instance\tTrial1\tTrial2\tTrial3\tTrial4\tTrial5\tAverage\tBest\n")

        for r in summary:
            trials = [f"{tr['distance']:.2f}" for tr in r['trial_results']]
            avg = sum(float(tr) for tr in trials) / len(trials)
            best = min(float(tr) for tr in trials)

            line = (
                f"{r['instance_id']}\t"
                f"{trials[0]}\t{trials[1]}\t{trials[2]}\t{trials[3]}\t{trials[4]}\t"
                f"{round(avg, 2)}\t{best}\n"
            )
            f.write(line)
    print(f'Wrote outputs to: {OUTPUT_DIR.resolve()}')
    #Write the results of the routes to a .txt file format Vehicle 1: 0 -> 5 -> 3 -> 0 
    for r in summary:
        with open(OUTPUT_DIR / f"output_txt/{r['instance_id']}.txt", "w", encoding="utf-8") as f:
            for idx, route in enumerate(r['best_routes'], start=1):
                path_str = " -> ".join(str(cid) for cid in route['path'])
                f.write(f"Vehicle {idx}: {path_str}\n")

if __name__ == '__main__':
    main()
