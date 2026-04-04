from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_PATH = ROOT_DIR / "data" / "schemas" / "cvrp_instance.schema.json"
DEFAULT_INSTANCE_DIR = ROOT_DIR / "data" / "instances"


def ensure_output_dirs() -> None:
    for path in (
        ROOT_DIR / "outputs" / "routes",
        ROOT_DIR / "outputs" / "metrics",
        ROOT_DIR / "outputs" / "plots",
        ROOT_DIR / "outputs" / "logs",
        ROOT_DIR / "submission",
    ):
        path.mkdir(parents=True, exist_ok=True)


def discover_instance_paths(instance_dir: Path | None = None) -> list[Path]:
    target_dir = instance_dir or DEFAULT_INSTANCE_DIR
    return sorted(target_dir.glob("Instance*.json"))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def output_paths_for_instance(instance_name: str) -> dict[str, Path]:
    return {
        "internal_route": ROOT_DIR / "outputs" / "routes" / f"{instance_name}.txt",
        "submission_route": ROOT_DIR / "submission" / f"{instance_name}.txt",
        "metrics": ROOT_DIR / "outputs" / "metrics" / f"{instance_name}_metrics.json",
        "plot": ROOT_DIR / "outputs" / "plots" / f"{instance_name}_routes.png",
    }


def serialize_routes(routes: Iterable[Iterable[int]]) -> str:
    lines = []
    for index, route in enumerate(routes, start=1):
        rendered = ", ".join(str(node_id) for node_id in route)
        lines.append(f"r{index}: {rendered}")
    return "\n".join(lines) + "\n"


def write_route_file(path: Path, routes: Iterable[Iterable[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_routes(routes), encoding="utf-8")


def sync_submission_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def to_repo_relative(path: Path) -> str:
    return str(path.relative_to(ROOT_DIR))
