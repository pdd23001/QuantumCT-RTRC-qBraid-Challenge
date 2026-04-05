import json
import math
import random

random.seed(42)

instance_names = [
    "A_n32_k5",
    "A_n33_k5",
    "A_n33_k6",
    "A_n34_k5",
    "A_n36_k5",
    "A_n37_k5",
    "A_n37_k6",
    "A_n38_k5",
    "A_n39_k5",
    "A_n39_k6",
    "A_n44_k6",
    "A_n45_k6",
    "A_n45_k7",
    "A_n46_k7",
    "A_n48_k7",
    "A_n53_k7",
    "A_n54_k7",
    "A_n55_k9",
    "A_n60_k9",
    "A_n61_k9",
    "A_n62_k8",
    "A_n63_k9",
    "A_n63_k10",
    "A_n64_k9",
    "A_n65_k9",
    "A_n69_k9",
    "A_n80_k10",
]

def extract_nv(instance_id):
    return int(instance_id.split("_k")[1])

def generate_instance(instance_id, min_customers=50, max_customers=100):
    Nv = extract_nv(instance_id)
    num_customers = random.randint(min_customers, max_customers)
    C = math.ceil(num_customers / Nv)

    customers = [{
        "customer_id": 0,
        "x": random.randint(0, 100),
        "y": random.randint(0, 100),
        "demand": 0
    }]

    for customer_id in range(1, num_customers + 1):
        customers.append({
            "customer_id": customer_id,
            "x": random.randint(0, 100),
            "y": random.randint(0, 100),
            "demand": 1
        })

    return {
        "instance_id": instance_id,
        "Nv": Nv,
        "C": C,
        "customers": customers
    }

instances_json = [generate_instance(inst) for inst in instance_names]

with open("setA_random_instances_grouped.json", "w") as f:
    json.dump(instances_json, f, indent=2)

print("Created setA_random_instances_grouped.json")