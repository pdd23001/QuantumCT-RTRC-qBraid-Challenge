from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from jsonschema import Draft202012Validator

from src.models import CVRPInstance, Node


class InstanceValidationError(ValueError):
    """Raised when an input instance or output route set is invalid."""


def _load_schema(schema_path: Path) -> dict[str, Any]:
    import json

    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_instance(raw: dict[str, Any], schema_path: Path) -> CVRPInstance:
    schema = _load_schema(schema_path)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(raw), key=lambda error: list(error.path))
    if errors:
        rendered = "; ".join(error.message for error in errors)
        raise InstanceValidationError(f"Schema validation failed: {rendered}")

    depot = Node(
        id=int(raw["depot"]["id"]),
        x=float(raw["depot"]["x"]),
        y=float(raw["depot"]["y"]),
    )
    customers = tuple(
        sorted(
            (
                Node(id=int(customer["id"]), x=float(customer["x"]), y=float(customer["y"]))
                for customer in raw["customers"]
            ),
            key=lambda node: node.id,
        )
    )

    customer_ids = [customer.id for customer in customers]
    if len(customer_ids) != len(set(customer_ids)):
        raise InstanceValidationError("Customer ids must be unique.")
    if depot.id != 0:
        raise InstanceValidationError("Depot id must be 0.")
    if any(customer.id <= 0 for customer in customers):
        raise InstanceValidationError("Customer ids must be positive integers.")

    vehicles = int(raw["vehicles"])
    capacity = int(raw["capacity"])
    if vehicles <= 0:
        raise InstanceValidationError("vehicles must be positive.")
    if capacity <= 0:
        raise InstanceValidationError("capacity must be positive.")
    if len(customers) > vehicles * capacity:
        raise InstanceValidationError(
            "Instance is infeasible: customer count exceeds vehicles * capacity."
        )

    return CVRPInstance(
        instance_name=str(raw["instance_name"]),
        depot=depot,
        vehicles=vehicles,
        capacity=capacity,
        customers=customers,
        qaoa_overrides=dict(raw.get("qaoa", {})),
    )


def validate_routes(
    instance: CVRPInstance, routes: Sequence[Sequence[int]]
) -> tuple[bool, tuple[str, ...]]:
    errors: list[str] = []

    if len(routes) != instance.vehicles:
        errors.append(
            f"Expected exactly {instance.vehicles} routes, received {len(routes)}."
        )

    expected_customers = set(instance.customer_ids)
    seen_customers: list[int] = []
    known_nodes = {0, *expected_customers}

    for route_index, route in enumerate(routes, start=1):
        if not route:
            errors.append(f"Route r{route_index} is empty.")
            continue
        if route[0] != 0 or route[-1] != 0:
            errors.append(f"Route r{route_index} must start and end at depot 0.")
        payload = [node_id for node_id in route[1:-1] if node_id != 0]
        if len(payload) > instance.capacity:
            errors.append(
                f"Route r{route_index} exceeds capacity {instance.capacity} with {len(payload)} customers."
            )
        unknown = [node_id for node_id in route if node_id not in known_nodes]
        if unknown:
            errors.append(f"Route r{route_index} contains unknown node ids: {unknown}.")
        duplicates_within_route = {
            node_id for node_id in payload if payload.count(node_id) > 1
        }
        if duplicates_within_route:
            errors.append(
                f"Route r{route_index} visits duplicate customers: {sorted(duplicates_within_route)}."
            )
        seen_customers.extend(payload)

    seen_set = set(seen_customers)
    duplicate_across_routes = {node_id for node_id in seen_set if seen_customers.count(node_id) > 1}
    if duplicate_across_routes:
        errors.append(
            f"Customers visited more than once overall: {sorted(duplicate_across_routes)}."
        )

    missing_customers = expected_customers - seen_set
    if missing_customers:
        errors.append(f"Customers not visited: {sorted(missing_customers)}.")

    extra_customers = seen_set - expected_customers
    if extra_customers:
        errors.append(f"Unexpected customers visited: {sorted(extra_customers)}.")

    return (not errors, tuple(errors))
