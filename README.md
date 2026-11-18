# Pipeline-Parallel Training: Optimal Stage Partitioning with Gurobi

Course project for *Statistics and Architectures for Big Data Processing – Module 2* (University of Bologna, Lecture 20: Parallelism and Pipeline).

**Group 9:** Mina Shahbazi, Arefe Rasouli, Arash Rezaei

## Problem

A neural network is a chain of `L` layers. We cut the chain into consecutive **stages** and decide how many processing elements (PEs, e.g. GPUs) **replicate** each stage. In a pipeline the slowest stage sets the pace, so the goal is to minimize `τ`, the time of the slowest stage.

Inputs per layer `l`: compute time `D_l`, activation transfer time `c_l` (after the layer), weight-sync time `w_l^q` (on `q` replicas) and memory `m_l`. Global inputs: number of PEs `|P|` and memory per PE `Γ`.

## Model

Binary variable `x[i,j,q] = 1` if layers `i..j` form one stage replicated on `q` PEs.

```
min τ
(1) unit flow from node 0 to node L (stage i..j = arc i-1 -> j)   # every layer in exactly one stage, stages consecutive
(2) τ >= T(i,j,q) * x[i,j,q],  T(i,j,q) = (1/q) * max(sum D_l, sum w_l^q)
(3) τ >= 2*c_j * (a stage ends at layer j)                         # communication at each cut
(4) sum q * x[i,j,q] <= |P|                                        # number of PEs
    stage memory <= Γ                                              # infeasible stages are never created
```

`T(i,j,q)` is a constant, precomputed for every candidate stage, so the model stays linear.

### Assumptions (not given in the lecture slides)

- `w_l^q = g_l * (q-1)/q` (all-reduce style weight synchronization)
- stage memory = sum of the memories of its layers

Replace them with the exact formulas from the PipeDream / memory-efficient pipeline papers if you need a more faithful model.

## Repository content

| File | Description |
|---|---|
| `pipeline_parallel_gurobi.py` | Instance generator, Gurobi model, exact DP and brute-force checks, naive baseline |
| `run_experiments.py` | Optimality tests and scalability study; writes the CSV files and the plot |
| `optimality_tests.csv` | Gurobi vs DP vs brute force on 40 random instances |
| `scalability.csv` | Solve time, model size and gap for `L = 6..20`, `|P| = 4, 8` |
| `scalability.png` | Solve time and model size plots |
| `Pipeline_Parallel_Training.pptx` | 7-minute presentation |

## Setup and usage

```bash
pip install gurobipy matplotlib pandas
python pipeline_parallel_gurobi.py   # small example: Gurobi, DP and brute force
python run_experiments.py            # all experiments
```

Expected output of the first command:

```
Gurobi : 18.0 [(1, 2, 1), (3, 5, 1), (6, 7, 1), (8, 8, 1)]
DP     : 18.0
Brute  : 18.0
```

Each tuple is `(first layer, last layer, number of PEs)`.

> The `pip` version of `gurobipy` ships with a size-limited license (about 2000 variables/constraints). The experiments here stay below it. For larger instances use a free academic license.

## Results

- **Optimality:** on 40 random instances (5–8 layers, 3–5 PEs) Gurobi, DP and brute force agree in 40/40 cases. In 3 instances all three methods report infeasibility (memory budget too tight).
- **Scalability:** `L = 6..20`, `|P| = 4, 8`, 5 seeds each. All instances solved to optimality (MIP gap 0). The largest model has 1129 binary variables (`L=20`, `|P|=8`) and solves in about 0.12 s at most.
- **Baseline:** an equal split with one PE per stage is 1.03x–1.96x slower than the optimum (average over seeds, per configuration).

## Theoretical scalability

- Binary variables: `O(L² · |P|)` (about `L²/2` intervals times `|P|` replica counts); constraints: same order.
- An exact dynamic program with `O(L² · |P|²)` exists (PipeDream style) and is used here as an independent check. The ILP is useful because extra linear constraints (memory, energy, deadlines) are easy to add.

## Applications

Large neural network training on many GPUs, and any chain of tasks that can be pipelined (video processing, data pipelines, assembly lines).

## Limitations

Random input data, simplified synchronization and memory model, and the size cap of the free Gurobi license.

## References

- A. Harlap et al., *PipeDream: Fast and Efficient Pipeline Parallel DNN Training*, arXiv:1806.03377, 2018.
- D. Narayanan et al., *Memory-Efficient Pipeline-Parallel DNN Training*, ICML 2021.
- J. Hennessy, D. Patterson, *Computer Architecture: A Quantitative Approach*, chapters 3–5.
