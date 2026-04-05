import json
import argparse
from pathlib import Path


def write_instance_txt(instance, out_dir: Path):
    instance_id = instance.get('id') or instance.get('instance_id') or 'unknown_instance'
    routes = instance.get('routes', [])
    out_path = out_dir / f"{instance_id}.txt"
    with out_path.open('w', encoding='utf-8') as f:
        for idx, route_info in enumerate(routes, start=1):
            route = route_info.get('route', []) if isinstance(route_info, dict) else route_info
            line = f"Vehicle {idx}: " + " -> ".join(str(x) for x in route)
            f.write(line)
            if idx < len(routes):
                f.write('\n')
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Split a benchmark JSON containing many instance results into one TXT file per instance.')
    parser.add_argument('json_file', help='Input JSON file containing a list of benchmark results')
    parser.add_argument('-o', '--out-dir', default='txt_outputs', help='Output directory for TXT files')
    args = parser.parse_args()

    json_path = Path(args.json_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        if 'results' in data and isinstance(data['results'], list):
            results = data['results']
        else:
            results = [data]
    elif isinstance(data, list):
        results = data
    else:
        raise ValueError('Unsupported JSON structure. Expected a list or an object containing results.')

    written = []
    for instance in results:
        written.append(write_instance_txt(instance, out_dir))

    print(f'Wrote {len(written)} TXT files to {out_dir}')
    for p in written[:10]:
        print(p.name)
    if len(written) > 10:
        print('...')


if __name__ == '__main__':
    main()

#Terminal command to run the script:
# python json_to_txt_split.py path/to/benchmark_results.json -o path/to/output_directory
