# Quantum Courier Challenge

**Sponsors:** RTX Technology Research Center (RTRC), QuantumCT, and qBraid  
**Event:** Yale Hackathon 2026

## Overview

Welcome to the Quantum Courier Challenge! Your mission: develop and execute your own implementation(s) of quantum optimization algorithms for solving capacitated vehicle routing problems (CVRPs) that arise in various logistics applications.

The CVRP is NP-hard — the solution space grows exponentially with problem size. Quantum algorithms offer alternative heuristics that may produce better-quality solutions faster than known classical methods. Even marginal improvements (e.g., 0.5%) can translate to millions of dollars in annual logistics savings.

The full challenge description, problem instances, and submission format are provided in `qCourier-YaleHackathon-2026.ipynb`.

---

## Getting Started

### Access qBraid

All circuits must run on qBraid-provided simulators.

1. Register at [account.qbraid.com](https://account.qbraid.com/signin) and confirm your email.
2. Go to **Account > Wallet**, select "Custom", enter $10, and apply promo code **`YQ26`** for $10 in free credits.
3. Go to **Account > Wallet** again, select the **Standard** monthly subscription ($20/mo), and apply promo code **`YQHACK26`** for one free month of standard subscription.

For more detailed instructions, see the notebook. 

---

## Requirements

- Formulate CVRP as a mathematical optimization program.
- Implement a quantum or quantum-classical hybrid algorithm — no classical-only solutions.
- Solution must be generalized (not hardcoded to specific instances).
- All circuits must run on qBraid simulators.
- Solve the CVRP instances provided in the notebook, starting from simple and scaling up.

---

## Submission

Submit the following by forking this GitHub repository. Make sure to keep it public. 

1. **Code** — quantum implementation with a package specification file to recreate the environment.
2. **Results** — solution files for each CVRP instance in the required format (see notebook), along with qubit count, gate operation count, and execution time per instance.
3. **Documentation** — explanation of your algorithm, scalability, and any novel insights.

---

## Judging Criteria

- Approximation ratio of the solution
- Scale of problems solved
- Novelty and resource efficiency
- Quality of presentation

---

## Prizes

- 1st Place: $1,000 Amazon gift card
- 2nd Place: $600 Amazon gift card
- 3rd Place: $400 Amazon gift card

All participants receive one month of free qBraid platform access (non-commercial).

---

## Resources

- [qBraid SDK Documentation](https://docs.qbraid.com/v2/)
- [Wikipedia: Vehicle Routing Problem](https://en.wikipedia.org/wiki/Vehicle_routing_problem)
- [Qiskit Optimization Tutorial on VRP](https://qiskit-community.github.io/qiskit-optimization/tutorials/07_examples_vehicle_routing.html)
- Online office hours: **4–5 PM EST, Saturday April 4th** — RTRC and QuantumCT staff available for questions. Use this [link](https://teams.microsoft.com/meet/28228728269636?p=XNlY8wgpb2SGm9iKdy)
- On-site support available during the hackathon with RTX and QuantumCT staff present. 
