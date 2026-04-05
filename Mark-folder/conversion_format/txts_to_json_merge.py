import json
import math
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

LINE_RE = re.compile(r'^\s*Vehicle\s+(\d+)\s*:\s*(.*?)\s*$')


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def parse_txt_solution(txt_path: Path):
    routes = []
    with txt_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = LINE_RE.match(line)
            if not m:
                raise ValueError(f'Invalid line in {txt_path.name}: {line}')
            vehicle = int(m.group(1))
            seq = [int(x.strip()) for x in m.group(2).split('->')]
            routes.append({'vehicle': vehicle, 'route': seq})
    routes.sort(key=lambda x: x['vehicle'])
    return routes


def load_instances(instances_path: Path) -> Dict[str, dict]:
    with instances_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('Instances JSON must be a list of instance objects.')
    return {item['instance_id']: item for item in data}


def build_result(instance_id: str, routes: List[dict], instance_meta: dict | None):
    result = {'id': instance_id, 'routes': []}

    if instance_meta is None:
        result.update({
            'customers': None,
            'vehicles_allowed': None,
            'vehicles_used': len(routes),
            'distance': None,
            'runtime': None,
            'valid': None,
        })
        for r in routes:
            result['routes'].append({
                'vehicle': r['vehicle'],
                'route': r['route'],
                'load': None,
                'distance': None,
            })
        return result

    customers_meta = instance_meta['customers']
    coord = {c['customer_id']: (c['x'], c['y']) for c in customers_meta}
    demand = {c['customer_id']: c['demand'] for c in customers_meta}
    capacity = instance_meta.get('C')
    vehicles_allowed = instance_meta.get('Nv')
    expected_customers = len(customers_meta) - 1

    seen_non_depot = []
    valid = True
    total_distance = 0.0

    for r in routes:
        seq = r['route']
        if not seq or seq[0] != 0 or seq[-1] != 0:
            valid = False
        route_load = 0
        route_distance = 0.0
        for i in range(len(seq) - 1):
            if seq[i] not in coord or seq[i+1] not in coord:
                valid = False
                continue
            route_distance += euclidean(coord[seq[i]], coord[seq[i+1]])
        for node in seq[1:-1]:
            if node == 0:
                valid = False
            if node not in demand:
                valid = False
                continue
            route_load += demand[node]
            seen_non_depot.append(node)
        if capacity is not None and route_load > capacity:
            valid = False
        total_distance += route_distance
        result['routes'].append({
            'vehicle': r['vehicle'],
            'route': seq,
            'load': route_load,
            'distance': round(route_distance, 6),
        })

    if vehicles_allowed is not None and len(routes) > vehicles_allowed:
        valid = False

    expected_set = set(range(1, expected_customers + 1))
    seen_set = set(seen_non_depot)
    if seen_set != expected_set or len(seen_non_depot) != expected_customers:
        valid = False

    result.update({
        'customers': expected_customers,
        'vehicles_allowed': vehicles_allowed,
        'vehicles_used': len(routes),
        'distance': round(total_distance, 6),
        'runtime': None,
        'valid': valid,
    })
    return result


def infer_instance_id(txt_path: Path) -> str:
    return txt_path.stem


def main():
    parser = argparse.ArgumentParser(description='Merge many TXT VRP solutions into one JSON file.')
    parser.add_argument('txt_dir', help='Directory containing per-instance TXT files')
    parser.add_argument('-o', '--output', default='merged_results.json', help='Output JSON file')
    parser.add_argument('--instances', help='Optional grouped instances JSON for enrichment/validation')
    args = parser.parse_args()

    txt_dir = Path(args.txt_dir)
    txt_files = sorted(txt_dir.glob('*.txt'))
    if not txt_files:
        raise FileNotFoundError(f'No .txt files found in {txt_dir}')

    instances = None
    if args.instances:
        instances = load_instances(Path(args.instances))

    merged = []
    for txt_file in txt_files:
        instance_id = infer_instance_id(txt_file)
        routes = parse_txt_solution(txt_file)
        meta = instances.get(instance_id) if instances else None
        merged.append(build_result(instance_id, routes, meta))

    with Path(args.output).open('w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2)

    print(f'Wrote {len(merged)} merged results to {args.output}')


if __name__ == '__main__':
    main()

# This script can be run from the command line as follows:
# python txts_to_json_merge.py path/to/txt_directory -o path/to/output.json --instances path/to/instances.json

