import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from pygad_vrp import OfficialPyGADVRP


def load_instances(path: Path) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('Input JSON must contain a list of instances.')
    return data


def run_trials(
    instances: List[Dict[str, Any]],
    trials: int,
    num_generations: int,
    sol_per_pop: int,
    num_parents_mating: int,
    mutation_probability: float,
    base_seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for instance in instances:
        trial_distances: List[float] = []
        instance_name = instance.get('instance_id', 'unknown_instance')

        for trial_idx in range(1, trials + 1):
            seed = base_seed + trial_idx - 1
            solver = OfficialPyGADVRP(
                instance=instance,
                num_generations=num_generations,
                sol_per_pop=sol_per_pop,
                num_parents_mating=num_parents_mating,
                mutation_probability=mutation_probability,
                seed=seed,
            )
            result = solver.solve()
            distance = float(result['distance'])
            trial_distances.append(distance)
            print(f"[{instance_name}] Trial {trial_idx}/{trials}: distance={distance:.2f}, valid={result['valid']}, seed={seed}")

        row: Dict[str, Any] = {'Instance name': instance_name}
        for i, distance in enumerate(trial_distances, start=1):
            row[f'Trial {i}'] = round(distance, 2)
        row['Average'] = round(mean(trial_distances), 2)
        rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, Any]], output_path: Path, trials: int) -> None:
    fieldnames = ['Instance name'] + [f'Trial {i}' for i in range(1, trials + 1)] + ['Average']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tabbed_txt(rows: List[Dict[str, Any]], output_path: Path, trials: int) -> None:
    headers = ['Instance name'] + [f'Trial {i}' for i in range(1, trials + 1)] + ['Average']
    table_rows: List[List[str]] = [headers]

    for row in rows:
        table_rows.append([str(row.get(col, '')) for col in headers])

    col_widths = [max(len(r[c]) for r in table_rows) for c in range(len(headers))]

    with open(output_path, 'w', encoding='utf-8') as f:
        for row in table_rows:
            padded = [value.ljust(col_widths[idx]) for idx, value in enumerate(row)]
            f.write('\t'.join(padded) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run multiple GA trials per VRP instance and save a distance table.'
    )
    parser.add_argument('--input', type=Path, default=Path('../c5n25.json'), help='Path to the input JSON instances file.')
    parser.add_argument('--trials', type=int, default=5, help='Number of GA trials per instance.')
    parser.add_argument('--base-seed', type=int, default=100, help='Starting seed. Trial i uses base_seed + i - 1.')
    parser.add_argument('--num-generations', type=int, default=400, help='Number of generations for PyGAD.')
    parser.add_argument('--sol-per-pop', type=int, default=80, help='Population size for PyGAD.')
    parser.add_argument('--num-parents-mating', type=int, default=20, help='Parents mating count for PyGAD.')
    parser.add_argument('--mutation-probability', type=float, default=0.2, help='Mutation probability for PyGAD.')
    parser.add_argument('--output-dir', type=Path, default=Path('./ga_trial_outputs'), help='Directory to store CSV/TXT results.')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    instances = load_instances(args.input)
    rows = run_trials(
        instances=instances,
        trials=args.trials,
        num_generations=args.num_generations,
        sol_per_pop=args.sol_per_pop,
        num_parents_mating=args.num_parents_mating,
        mutation_probability=args.mutation_probability,
        base_seed=args.base_seed,
    )

    #csv_path = args.output_dir / f"{args.input.stem}_ga_trial_distances.csv"
    txt_path = args.output_dir / f"{args.input.stem}_ga_trial_distances.txt"

    #write_csv(rows, csv_path, args.trials)
    write_tabbed_txt(rows, txt_path, args.trials)

    #print(f'CSV saved to: {csv_path}')
    print(f'TXT saved to: {txt_path}')


if __name__ == '__main__':
    main()
