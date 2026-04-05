# Exact-Subset Route Encoding + BF-DCQO + Selective BBB-DCQO Refinement

This notebook is the non-QAOA companion to the current routing notebook.

It is designed to sit side-by-side with the existing QAOA notebook while following a very similar structure:
1. imports + utility functions
2. phase-based helper code
3. configuration cell
4. main single-instance execution cell
5. feasibility check cell
6. optional run-all-instances cell

The key difference is the optimization model.

Instead of:
- decomposing customers into clusters,
- building a separate TSP QUBO per cluster,
- and solving each cluster route with QAOA,

this notebook:
- precomputes **exact feasible depot-to-depot mini-routes** for all customer subsets up to vehicle capacity,
- introduces one binary variable per feasible subset-route,
- solves a **global route-selection QUBO**,
- uses **BF-DCQO** as the main quantum optimizer,
- and uses **BBB-DCQO only as a selective refinement stage** when the BF result is close but still ambiguous.

---

## High-Level Idea

The original CVRP formulation uses edge variables like \(x_{ijk}\), where:
- \(i\) = source node
- \(j\) = destination node
- \(k\) = vehicle

That representation is expressive, but it becomes expensive because it must encode:
- depot departure and return,
- exactly-one-visit constraints,
- flow conservation,
- capacity constraints,
- and sometimes subtour elimination.

This notebook avoids that large edge-based decision space.

### New decision variable

For every feasible customer subset \(S\) with \(|S| \le C\), define:

\[
y_S \in \{0,1\}
\]

where:
- \(y_S = 1\) means one vehicle is assigned to serve exactly the customer subset \(S\),
- and that route already includes the best depot-to-depot order for the customers in \(S\).

For each subset \(S\), the notebook computes:

\[
c_S = \min_{\pi \in \text{Perm}(S)} \left( d_{0,\pi_1} + \sum_t d_{\pi_t,\pi_{t+1}} + d_{\pi_{|S|},0} \right)
\]

So the order of visiting customers inside a chosen route is solved **before** the quantum step.

Then the quantum optimization only decides:

> Which exact-feasible routes should be selected so that every customer is covered exactly once and exactly \(N_v\) routes are chosen?

---

## Why this is a good fit for this hackathon

This works particularly well here because the challenge capacity is defined as:

> maximum number of customers a vehicle may visit

That means the feasible route library is easy to enumerate for small benchmark instances:
- choose all customer subsets up to size \(C\),
- compute the exact best order for each subset,
- assign each such subset-route one binary variable.

This dramatically reduces modeling complexity compared with the full edge-variable CVRP formulation.

---

# Cell-by-Cell Explanation

---

## Cell 0 — Imports + Utilities

### Purpose

This cell provides:
- numeric tools,
- timing,
- route geometry,
- bitstring helpers,
- and general QUBO-energy evaluation.

It plays the same role as the import/utilities cell in the QAOA notebook.

### Main responsibilities

#### 1. Imports
This cell imports:
- `numpy`, `math`, `time`, `json`, `warnings`
- combinatorics helpers like `combinations`, `permutations`
- Qiskit circuit tools and Aer simulator

These are used for:
- computing Euclidean distances,
- enumerating customer subsets,
- building circuits,
- sampling measured bitstrings,
- and evaluating QUBO energies.

#### 2. Distance helpers
Functions like:
- `euclidean_distance(...)`
- `build_distance_matrix(...)`
- `route_distance(...)`

are used throughout the notebook.

They ensure a consistent geometric interpretation of the problem:
- depot is node `0`
- travel cost is Euclidean
- route length is the sum of consecutive segment distances

#### 3. Bitstring helpers
Functions like:
- `infer_num_vars_from_qubo(Q)`
- `bitstring_to_array(...)`

are needed because the quantum solver returns measured bitstrings, but the rest of the notebook wants binary arrays indexed by variable number.

#### 4. Energy evaluation
`energy_of_bitstring(...)` computes the QUBO objective value for a bitstring.

This is crucial because:
- the solver samples many candidate states,
- not every sampled state is feasible,
- and BBB refinement later needs to know which low-energy samples are worth preserving.

### Why this cell exists
Without this cell, every later phase would need to duplicate:
- distance logic,
- QUBO indexing logic,
- and bitstring parsing logic.

This cell centralizes those operations.

---

## Cell 1 — Phase 1: Exact Subset-Route Library

### Purpose

This cell replaces the classical clustering step in the QAOA notebook.

Instead of dividing customers into clusters first, it generates a **library of feasible mini-routes**.

Each mini-route corresponds to:
- a customer subset \(S\),
- its exact best order,
- and its exact depot-to-depot cost.

### Main functions

#### `exact_subset_route_cost(subset, D)`
For a given customer subset:
- enumerates all permutations,
- computes the depot-to-first + internal transitions + last-to-depot cost,
- returns the best order and exact cost.

This is only practical because the capacity \(C\) is small in the benchmark instances.

For example:
- if \(C = 3\), then the largest subset requires only \(3! = 6\) permutations,
- which is cheap.

#### `generate_route_pool(nodes, Nv, C, top_k=None)`
This is the most important function in the cell.

It:
1. builds the distance matrix,
2. enumerates all customer subsets of sizes `1` through `C`,
3. computes the exact best route for each subset,
4. computes a heuristic route score,
5. optionally truncates the pool with `top_k`,
6. assigns a variable index to every route.

Each route record includes:
- `subset`
- `order`
- `cost`
- `size`
- `score`
- `var_idx`

### Why the route score exists
The score is used for optional route-pool compression later.

The idea is:
- some feasible subset-routes are clearly better building blocks than others,
- so for larger instances it may help to keep only the strongest candidates.

This is not required for correctness, but it is useful for scalability experiments.

### Output of this cell
The final route pool is a list of dictionaries such as:

```python
{
    "subset": (2, 5, 7),
    "order": [5, 2, 7],
    "cost": 14.31,
    "size": 3,
    "score": 2.74,
    "var_idx": 81
}
```

That means:
- this route serves customers `{2, 5, 7}`,
- the best internal visiting order is `5 → 2 → 7`,
- the depot-to-depot route cost is `14.31`,
- and the binary variable controlling this route is `y_81`.

### Why this cell matters
This cell is where the notebook compresses the CVRP structure.

After this step:
- capacity is automatically handled, because only subsets of size at most `C` are generated,
- route continuity is automatically handled, because each route is already a full depot-to-depot tour,
- and the quantum step becomes a route-selection problem instead of a route-construction problem.

---

## Cell 2 — Phase 2: Build the Subset-Selection QUBO

### Purpose

This cell turns the route library into a single global optimization problem.

The binary decision variables are:
- one variable per feasible subset-route in the route pool.

### Optimization model

The notebook builds the following QUBO:

\[
E(y)=\sum_S c_S y_S
+ A\sum_{i=1}^n \left(1-\sum_{S\ni i} y_S\right)^2
+ B\left(N_v-\sum_S y_S\right)^2
\]

### Interpretation of each term

#### 1. Route-cost term
\[
\sum_S c_S y_S
\]

This is the true routing cost.

If route \(S\) is selected, its exact cost \(c_S\) is paid.

#### 2. Customer-coverage term
\[
A\sum_{i=1}^n \left(1-\sum_{S\ni i} y_S\right)^2
\]

This enforces:
- every customer must be covered exactly once.

If a customer is:
- missed entirely, penalty is added
- covered twice, penalty is added
- covered exactly once, penalty is zero

#### 3. Route-count term
\[
B\left(N_v-\sum_S y_S\right)^2
\]

This enforces:
- exactly `Nv` selected routes

So the chosen subset-routes correspond to exactly the number of available vehicles.

### Main functions

#### `build_subset_qubo(route_pool, n_customers, Nv, A=None, B=None)`
This constructs:
- the QUBO coefficient dictionary `Q`,
- the constant offset,
- and the actual penalty values `A`, `B`.

The function:
1. adds route costs on the diagonal,
2. adds coverage penalties for every customer,
3. adds route-count penalties across all variables.

#### `qubo_dict_to_ising(Q, n)`
Converts the QUBO to Ising form using the usual substitution:

\[
x = \frac{1 - Z}{2}
\]

This gives:
- local field coefficients `h`
- pairwise couplings `J`
- a constant offset

These are what the BF-DCQO solver uses to construct the circuit.

#### `decode_route_selection(bitstring_or_bits, route_pool, Nv, n_customers)`
This turns a measured bitstring back into actual routes.

It:
1. finds which subset variables were set to `1`,
2. checks whether exactly `Nv` routes were chosen,
3. checks whether all customers are covered exactly once,
4. returns depot-to-depot routes if feasible,
5. otherwise returns infeasible.

### Why this cell matters
This cell is the bridge between:
- classical route precomputation,
- and quantum binary optimization.

It takes a routing problem and recasts it as a route-selection Hamiltonian.

---

## Cell 3 — Phase 3: BF-DCQO Solver Scaffold

### Purpose

This is the main quantum optimization cell.

It implements a **BF-DCQO-style iterative bias-field solver** for the route-selection QUBO.

### Important design note
This notebook intentionally uses a **runnable BF-DCQO-style scaffold**, not a final paper-perfect implementation of every counterdiabatic detail.

That means:
- the outer logic matches the intended BF workflow,
- but one circuit-construction function is the natural place to upgrade later if you want a more paper-faithful counterdiabatic block.

This makes the notebook easy to run now and easy to improve later.

### How BF-DCQO is represented here

The core idea is:

1. Start with an Ising Hamiltonian derived from the route-selection QUBO.
2. Build a shallow digitized circuit with:
   - an X mixer,
   - local Z fields,
   - ZZ couplings,
   - and learned bias fields \(h_b\).
3. Sample bitstrings.
4. Keep only elite low-energy samples.
5. Update the bias field from those elite samples.
6. Repeat for several iterations.
7. Use a stronger signed final update to sharpen toward a better low-energy region.

### Main functions

#### `schedule_lambda(step, total_steps)`
Provides a smooth annealing-like schedule across the digitized layers.

Conceptually:
- early layers lean more toward exploration,
- later layers lean more toward the problem Hamiltonian.

#### `build_bf_dcqo_surrogate_circuit(h, J, hb, p=3, ...)`
Builds the actual circuit used in one BF iteration.

Current structure:
- initialize all qubits in `|+>`
- apply RX mixer rotations
- apply local RZ rotations from:
  - the problem fields `h`
  - the learned bias fields `hb`
- apply RZZ interactions from `J`
- measure all qubits

This is the **main upgrade hook** if you later want a closer implementation of the paper’s counterdiabatic block.

#### `sample_counts(circuit, shots=1024, optimization_level=1)`
Executes the circuit on `AerSimulator`.

Returns:
- measurement counts,
- transpiled gate count,
- qubit count

This is useful because the notebook also tracks:
- execution effort,
- depth/gate load,
- and the largest quantum footprint.

#### `elite_shots(counts, Q, constant, alpha=0.02)`
Selects only the lowest-energy fraction of measured samples.

This is the bias-field version of saying:
> only trust the better sampled states when deciding where the search should move next.

#### `compute_bias_update(...)`
Computes the next bias field.

There are two modes:

##### unsigned mode
Used in the early iterations.

The notebook:
- estimates the magnitude of the bias from elite-sample Z statistics,
- and uses the best elite bitstring to choose direction.

This helps stabilize the search.

##### signed mode
Used in the last iteration.

The notebook:
- uses the signed elite expectation directly,
- scaled by `kappa`,
- to sharpen the search toward the best observed region.

#### `run_bf_dcqo_generic_qubo(...)`
This is the generic iterative BF loop.

It:
1. converts the QUBO to Ising form,
2. initializes the bias field to zero,
3. builds and runs a circuit each iteration,
4. records counts and gate information,
5. updates the bias field,
6. tracks the best low-energy sample found.

#### `run_bf_dcqo_subset_solver(...)`
This is the route-specific wrapper around the generic BF solver.

It additionally:
- decodes each sampled bitstring as a route selection,
- tracks the best **feasible** route set,
- computes the feasible sample rate,
- and returns route-level outputs instead of just raw bitstrings.

### Why this cell matters
This cell is the notebook’s primary quantum engine.

It takes the route-selection QUBO and tries to find a low-energy, feasible combination of subset-routes without using a classical variational optimizer loop like QAOA.

---

## Cell 4 — Phase 4: BBB-DCQO Selective Refinement

### Purpose

This cell is **not** the main solver.

It is only used after BF-DCQO if:
- BF found a decent solution,
- but the final measured distribution still contains ambiguity.

This is the notebook’s selective branch-and-bound refinement stage.

### Main idea

Instead of running a heavy branch-and-bound process from the beginning, this notebook only refines after BF-DCQO.

It does this by:
1. computing variable marginals from the final BF sample distribution,
2. fixing very confident variables,
3. branching only on the most uncertain variables,
4. solving each reduced branch with a shallower BF solve.

That keeps BBB as a targeted rescue tool rather than the main runtime burden.

### Main functions

#### `compute_marginals(counts, n)`
Computes the empirical probability that each route variable was measured as `1`.

If:
- `p_i ≈ 1`, the solver strongly believes route \(i\) should be selected
- `p_i ≈ 0`, it strongly believes route \(i\) should not be selected
- `p_i ≈ 0.5`, the solver is uncertain

#### `reduce_qubo_with_fixed(Q, constant, n, fixed)`
Substitutes fixed binary decisions into the QUBO and produces a smaller reduced problem.

This is how branching becomes computationally manageable.

#### `expand_reduced_bits(...)`
Takes a solution of the reduced QUBO and reinserts the fixed variables to recover the full decision vector.

#### `conflicts_if_set_one(var_idx, fixed, route_pool)`
Checks whether forcing a route variable to `1` would conflict with already-fixed selected routes.

This is a problem-specific pruning rule:
- if two chosen subset-routes overlap on customers,
- they cannot both be selected.

#### `run_bbb_dcqo_refinement(...)`
This is the main refinement wrapper.

It:
1. reads the final BF counts,
2. computes marginals,
3. fixes obvious 0s and obvious 1s,
4. identifies the most uncertain variables,
5. branches on them,
6. solves each branch with a shallower BF run,
7. returns the best feasible refined solution found.

### Why this cell matters
This cell adds a second stage of intelligence without making the whole notebook branch-and-bound heavy from the start.

It is only used when helpful.

That matches the intended role:
- BF-DCQO = main lightweight solver
- BBB-DCQO = targeted accuracy refinement

---

## Cell 5 — Instance Library + Configuration

### Purpose

This cell mirrors the configuration cell in the current QAOA notebook.

It gives a single place to choose:
- which instance to run,
- whether to read from file or from an in-notebook dictionary,
- route-pool size,
- BF parameters,
- BBB refinement settings.

### Main sections

#### 1. Instance dictionary
Provides the benchmark instances directly in notebook form.

Each instance contains:
- `Nv`
- `C`
- `nodes`

where node `0` is the depot.

#### 2. File-based input option
If `read_from_file = True`, the notebook loads the chosen instance from JSON.

This mirrors the pattern used by the current QAOA notebook.

#### 3. Route-pool settings
`route_pool_limit` controls whether the route library is:
- full,
- or compressed to the top `k` candidates.

For the first pass, use:
- `route_pool_limit = None`

#### 4. BF solver settings
Typical initial values:
- `p = 3`
- `bias_iters = 6`
- `shots = 1024`
- `alpha = 0.02`

#### 5. BBB refinement settings
Typical initial values:
- `p_small = 2`
- `bias_iters_small = 3`
- `shots_small = 512`

### Why this cell matters
This cell separates:
- algorithm definition,
- from experimental configuration.

That makes the notebook much easier to tune and rerun.

---

## Cell 6 — Main Run: Route Pool → QUBO → BF → Optional BBB

### Purpose

This is the main execution cell for one instance.

It takes the selected instance from configuration and runs the full pipeline.

### Detailed flow

#### Step 1. Generate the exact route pool
```python
route_pool, D = generate_route_pool(...)
```

This builds all feasible subset-routes and their exact costs.

This is where the classical preprocessing happens.

#### Step 2. Build the subset-selection QUBO
```python
Q, constant, A, B = build_subset_qubo(...)
```

This encodes:
- route cost minimization,
- exact customer coverage,
- exact number of selected routes.

#### Step 3. Run BF-DCQO
```python
bf_result = run_bf_dcqo_subset_solver(...)
```

This performs the main quantum optimization.

Outputs include:
- best feasible decoded routes
- best feasible route cost
- feasible sample rate
- iteration history
- total gates
- total time
- max qubits

#### Step 4. Optional BBB refinement
```python
bbb_result = run_bbb_dcqo_refinement(...)
```

Only used if enabled.

This is the selective second-stage improvement.

#### Step 5. Choose the final solution
The notebook compares:
- the BF result
- and the BBB-refined result

and keeps whichever is better.

### What gets printed
This cell prints:
- route-pool summary
- penalty values
- iteration-by-iteration BF information
- best feasible BF result
- BBB branch count and refined result
- final selected depot-to-depot routes

### Why this cell matters
This is the notebook’s main “run one instance” driver.

It corresponds directly to the single-instance execution cell in the QAOA notebook.

---

## Cell 7 — Feasibility Check

### Purpose

This cell mirrors the feasibility-check cell in the QAOA notebook.

It verifies that the final selected route set satisfies the CVRP requirements.

### What it checks

#### 1. Every route starts and ends at depot
Each route must begin with `0` and end with `0`.

#### 2. Capacity constraint
No route may contain more than `C` customers.

#### 3. No repeated customers inside a route
Each selected route should visit each of its assigned customers once.

#### 4. Every customer visited exactly once globally
Across all routes:
- no customer may be missing,
- no customer may be duplicated.

#### 5. Vehicle count
The number of used routes must not exceed or violate the intended fleet size.

#### 6. Flow conservation
The cell tracks:
- in-degree
- out-degree

For the depot:
- number of departures should match number of returns

For each customer:
- in-degree should be `1`
- out-degree should be `1`

#### 7. Distance verification
The notebook recomputes route distances directly from geometry and verifies they match the solver’s reported total.

### Why this cell matters
This is the final correctness gate.

A low-energy bitstring is not enough by itself:
- it must decode into a valid CVRP solution.

This cell confirms that.

---

## Cell 8 — Optional Run-All-Instances

### Purpose

This is the batch-run cell.

It mirrors the final cell in the QAOA notebook that iterates through all instances.

### What it does

For each benchmark instance:
1. builds the exact route pool,
2. builds the global route-selection QUBO,
3. runs BF-DCQO,
4. optionally runs BBB refinement,
5. prints the selected routes,
6. stores a summary row.

### Final summary table
The summary table reports:
- instance id
- number of customers
- vehicles
- capacity
- final distance
- qubits
- gates
- total time
- source of final answer (`BF-DCQO` or `BBB-DCQO`)
- feasibility status

### Why this cell matters
This cell is the easiest way to:
- verify the notebook on all benchmark instances,
- gather resource usage,
- and confirm that the new solver behaves consistently across sizes.

---

# End-to-End Pipeline Summary

The complete notebook workflow is:

1. **Read one CVRP instance**
2. **Enumerate every feasible customer subset up to capacity**
3. **Compute the exact best order for each subset**
4. **Create one binary variable per exact feasible route**
5. **Build a global route-selection QUBO**
6. **Convert the QUBO to Ising form**
7. **Run BF-DCQO as the main quantum optimization loop**
8. **Decode feasible measured route sets**
9. **Optionally run BBB-DCQO selective refinement**
10. **Choose the best feasible solution**
11. **Validate feasibility and recompute distances**
12. **Optionally repeat for all benchmark instances**

---

# Why this notebook is different from the current QAOA notebook

## Current QAOA notebook
The current notebook:
- decomposes customers into clusters,
- solves one TSP per cluster,
- uses QAOA for each cluster separately,
- and stitches the route set together afterward.

## This notebook
This notebook:
- does **not** cluster first,
- instead constructs a global library of exact feasible mini-routes,
- uses one binary variable per route,
- solves the route-selection problem globally,
- and only uses BBB refinement if BF needs help.

So the optimization emphasis changes from:

> “Build routes inside clusters”

to:

> “Choose the best combination of exact feasible routes”

---

# Main Advantages of This Approach

1. **Capacity is enforced structurally**
   - only subsets of size at most `C` are included

2. **Route continuity is enforced structurally**
   - every variable already corresponds to a full depot-to-depot route

3. **The quantum problem is global**
   - the optimizer decides which subset-routes best cover the full instance

4. **No classical variational loop like QAOA**
   - BF-DCQO uses iterative bias-field updates instead

5. **BBB is only used when needed**
   - so the refinement logic remains lightweight and selective

---

# Practical Notes

## Best initial settings
For a first working version:
- `route_pool_limit = None`
- `p = 3`
- `bias_iters = 6`
- `shots = 1024`
- `alpha = 0.02`
- `use_bbb_refinement = True`

## If Instance 4 becomes heavy
Try:
- `route_pool_limit = 120`
- then `route_pool_limit = 60`

That gives a smaller variable set and a lighter QUBO.

## Best place for future improvement
If you want a more faithful BF-DCQO implementation later, the main upgrade point is:

```python
build_bf_dcqo_surrogate_circuit(...)
```

Everything else in the notebook can mostly stay the same.

---

# One-Sentence Summary

This notebook replaces cluster-by-cluster QAOA routing with a global exact-subset route-selection formulation, then uses BF-DCQO as the primary quantum search method and BBB-DCQO only as a targeted refinement stage when the BF solution is promising but still uncertain.