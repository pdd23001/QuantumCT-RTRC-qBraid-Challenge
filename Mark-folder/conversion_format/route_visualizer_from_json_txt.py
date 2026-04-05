import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np


ROUTE_PATTERN = re.compile(r"^\s*Vehicle\s+(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)


def load_instances(json_path: Path) -> Dict[str, Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected official instances JSON to be a list of instance objects.")

    by_id: Dict[str, Dict[str, Any]] = {}
    for instance in data:
        instance_id = instance["instance_id"]
        by_id[instance_id] = instance
    return by_id



def parse_routes_txt(txt_path: Path) -> List[List[int]]:
    routes: List[List[int]] = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = ROUTE_PATTERN.match(line)
            if not match:
                continue

            route_part = match.group(2)
            nodes = [int(piece.strip()) for piece in route_part.split("->")]
            routes.append(nodes)

    if not routes:
        raise ValueError(f"No routes found in TXT file: {txt_path}")

    return routes



def build_maps(instance: Dict[str, Any]) -> Tuple[Tuple[float, float], Dict[int, Tuple[float, float, int]]]:
    depot = None
    customer_map: Dict[int, Tuple[float, float, int]] = {}

    for c in instance["customers"]:
        cid = int(c["customer_id"])
        x = float(c["x"])
        y = float(c["y"])
        demand = int(c.get("demand", 1))

        if cid == 0:
            depot = (x, y)
        else:
            customer_map[cid] = (x, y, demand)

    if depot is None:
        raise ValueError(f"Instance {instance['instance_id']} has no depot with customer_id=0.")

    return depot, customer_map



def route_distance(route: List[int], coords: Dict[int, Tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(route) - 1):
        a = coords[route[i]]
        b = coords[route[i + 1]]
        total += float(np.hypot(a[0] - b[0], a[1] - b[1]))
    return total



def total_distance(routes: List[List[int]], depot: Tuple[float, float], customer_map: Dict[int, Tuple[float, float, int]]) -> float:
    coords: Dict[int, Tuple[float, float]] = {0: depot}
    for cid, (x, y, _) in customer_map.items():
        coords[cid] = (x, y)

    return sum(route_distance(route, coords) for route in routes)



def validate_routes(routes: List[List[int]], instance: Dict[str, Any]) -> Tuple[bool, str]:
    capacity = int(instance["C"])
    max_vehicles = int(instance["Nv"])
    demand_map = {int(c["customer_id"]): int(c.get("demand", 1)) for c in instance["customers"]}
    expected_customers = sorted(int(c["customer_id"]) for c in instance["customers"] if int(c["customer_id"]) != 0)

    if len(routes) > max_vehicles:
        return False, f"uses {len(routes)} vehicles, limit is {max_vehicles}"

    visited: List[int] = []
    for idx, route in enumerate(routes, start=1):
        if len(route) < 2 or route[0] != 0 or route[-1] != 0:
            return False, f"route {idx} must start and end at depot 0"

        load = 0
        for node in route[1:-1]:
            if node == 0:
                return False, f"route {idx} contains depot 0 in the middle"
            if node not in demand_map:
                return False, f"route {idx} contains unknown customer {node}"
            load += demand_map[node]
            visited.append(node)

        if load > capacity:
            return False, f"route {idx} exceeds capacity ({load} > {capacity})"

    if sorted(visited) != expected_customers:
        return False, "routes do not visit each customer exactly once"

    return True, "ok"



def plot_instance_routes(instance: Dict[str, Any], routes: List[List[int]], output_path: Path, title_suffix: str = "TXT Route Visualization") -> None:
    depot, customer_map = build_maps(instance)
    dist_val = total_distance(routes, depot, customer_map)
    valid, reason = validate_routes(routes, instance)

    fig, ax = plt.subplots(figsize=(12, 8))

    depot_x, depot_y = depot
    ax.scatter(depot_x, depot_y, c="black", marker="X", s=220, label="Depot", zorder=5)
    ax.text(depot_x + 0.25, depot_y + 0.25, "0", fontsize=11, weight="bold")

    xs = [customer_map[cid][0] for cid in sorted(customer_map)]
    ys = [customer_map[cid][1] for cid in sorted(customer_map)]
    ax.scatter(xs, ys, c="black", s=40, zorder=4)

    for cid in sorted(customer_map):
        x, y, _ = customer_map[cid]
        ax.text(x + 0.25, y + 0.25, str(cid), fontsize=9)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for route_idx, route in enumerate(routes, start=1):
        points = []
        for node in route:
            if node == 0:
                points.append((depot_x, depot_y))
            else:
                x, y, _ = customer_map[node]
                points.append((x, y))

        points = np.array(points, dtype=float)
        ax.plot(
            points[:, 0],
            points[:, 1],
            linewidth=2.2,
            marker="o",
            markersize=4,
            color=colors[(route_idx - 1) % len(colors)],
            label=f"Route {route_idx}: {route}",
            zorder=2,
        )

    status = "valid" if valid else f"invalid ({reason})"
    ax.set_title(
        f"{instance['instance_id']} - {title_suffix}\n"
        f"dist={dist_val:.3f}, {status}",
        fontsize=16,
    )
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.grid(True, alpha=0.5)
    ax.legend(fontsize=8, loc="best")
    ax.set_aspect("equal", adjustable="box")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def infer_instance_id_from_txt(txt_path: Path) -> str:
    stem = txt_path.stem.lower()
    stem = stem.replace(" ", "_")
    if stem.startswith("instance_"):
        return stem
    if stem.startswith("instance") and stem[8:].isdigit():
        return f"instance_{stem[8:]}"
    raise ValueError(f"Could not infer instance_id from TXT filename: {txt_path.name}")



def process_all(json_path: Path, txt_dir: Path, output_dir: Path) -> None:
    instances = load_instances(json_path)
    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {txt_dir}")

    for txt_path in txt_files:
        instance_id = infer_instance_id_from_txt(txt_path)
        if instance_id not in instances:
            raise KeyError(f"{instance_id} from {txt_path.name} not found in JSON.")

        routes = parse_routes_txt(txt_path)
        out_path = output_dir / f"{instance_id}_routes.png"
        plot_instance_routes(instances[instance_id], routes, out_path)
        print(f"Saved {out_path}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize CVRP routes from official_instances.json and TXT route files.")
    parser.add_argument("--json", type=Path, default=Path("official_instances.json"), help="Path to official_instances.json")
    parser.add_argument("--txt-dir", type=Path, default=Path("."), help="Directory containing route TXT files")
    parser.add_argument("--output-dir", type=Path, default=Path("official_outputs/visualized_from_txt"), help="Directory to save route PNG files")
    args = parser.parse_args()

    process_all(args.json, args.txt_dir, args.output_dir)


if __name__ == "__main__":
    main()
    # Example usage:
    # python route_visualizer_from_json_txt.py --json official_instances.json --txt-dir txt --output-dir dir
