import json
import random
from pathlib import Path


def generate_instance(instance_id: str, num_customers: int = 25):
    customers = []

    # Depot
    customers.append({
        "customer_id": 0,
        "x": 0,
        "y": 0,
        "demand": 0
    })

    # Customers
    for i in range(1, num_customers + 1):
        customers.append({
            "customer_id": i,
            "x": random.randint(-10, 10),
            "y": random.randint(-10, 10),
            "demand": random.randint(1, 2)
        })

    return {
        "instance_id": instance_id,
        "Nv": 7,   # you can tweak this if needed
        "C": 5,
        "customers": customers
    }


def main():
    random.seed(42)  # reproducible

    num_instances = 5
    instances = []

    for i in range(1, num_instances + 1):
        inst = generate_instance(f"instance_{i}")
        instances.append(inst)

    output_path = Path("demand2_instances.json")
    with open(output_path, "w") as f:
        json.dump(instances, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()