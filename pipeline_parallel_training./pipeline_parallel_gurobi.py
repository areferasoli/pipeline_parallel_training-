"""
Pipeline-Parallel Training: partition a chain of layers into stages and
choose how many PEs replicate each stage, minimizing the slowest stage (tau).

Model (course notation, Harlap et al. 2018 Sec 3.2, bottleneck form):
  layers l = 1..L : compute time D_l, activation transfer time c_l (after l),
                    weight-sync time w_l^q (on q replicas), memory m_l
  |P| PEs, memory Gamma per PE

  min tau
  (1) layers form consecutive stages, each layer exactly once
  (2) T(i->j,q) = (1/q) * max( sum D_l , sum w_l^q )  <= tau
  (3) 2*c_j <= tau  for every cut after layer j
  (4) sum_s q_s <= |P|  and  memory of a stage <= Gamma

Linearization: T(i->j,q) is a constant -> precomputed for every candidate
stage (i, j, q). Binary x[i,j,q] = 1 if layers i..j form a stage on q PEs.
Consecutiveness = unit flow from node 0 to node L over the "layer graph"
(stage i..j is an arc i-1 -> j).

ASSUMPTIONS (the slides do not give them, state them in the presentation):
  w_l^q = g_l * (q-1)/q   (ring all-reduce style, g_l = weight size / bandwidth)
  stage memory = sum of m_l of its layers (per replica)
"""
import itertools
import random
import time

import gurobipy as gp
from gurobipy import GRB


# ----------------------------------------------------------------- instance
def make_instance(L, P, seed=0, mem_tight=True):
    rng = random.Random(seed)
    D = [rng.randint(2, 12) for _ in range(L)]        # compute time
    c = [rng.randint(0, 3) for _ in range(L)]         # activation transfer
    g = [rng.randint(1, 10) for _ in range(L)]        # weight sync base
    m = [rng.randint(1, 6) for _ in range(L)]         # memory per layer
    # memory budget: roughly 40% of the model fits in one PE
    Gamma = max(max(m), int(0.4 * sum(m))) if mem_tight else sum(m)
    return dict(L=L, P=P, D=D, c=c, g=g, m=m, Gamma=Gamma)


def stage_time(inst, i, j, q):
    """T(i->j, q), layers 1-indexed inclusive."""
    D, g = inst["D"], inst["g"]
    comp = sum(D[i - 1:j])
    sync = sum(g[l - 1] for l in range(i, j + 1)) * (q - 1) / q
    return max(comp, sync) / q


def stage_mem(inst, i, j):
    return sum(inst["m"][i - 1:j])


def candidates(inst):
    """All feasible stages (i, j, q) with memory check."""
    L, P = inst["L"], inst["P"]
    out = {}
    for i in range(1, L + 1):
        for j in range(i, L + 1):
            if stage_mem(inst, i, j) > inst["Gamma"]:
                break  # longer stages only need more memory
            for q in range(1, P + 1):
                out[(i, j, q)] = stage_time(inst, i, j, q)
    return out


# ------------------------------------------------------------ Gurobi model
def solve_gurobi(inst, time_limit=60, verbose=False):
    L, P, c = inst["L"], inst["P"], inst["c"]
    cand = candidates(inst)
    t0 = time.time()
    mdl = gp.Model("pipeline_parallel")
    mdl.Params.OutputFlag = 1 if verbose else 0
    mdl.Params.TimeLimit = time_limit

    x = mdl.addVars(cand.keys(), vtype=GRB.BINARY, name="x")
    tau = mdl.addVar(lb=0.0, name="tau")
    mdl.setObjective(tau, GRB.MINIMIZE)

    # (1) flow: one path 0 -> L, arc (i-1 -> j) = stage i..j
    for node in range(0, L + 1):
        out_f = gp.quicksum(x[i, j, q] for (i, j, q) in cand if i - 1 == node)
        in_f = gp.quicksum(x[i, j, q] for (i, j, q) in cand if j == node)
        if node == 0:
            mdl.addConstr(out_f == 1, name="source")
        elif node == L:
            mdl.addConstr(in_f == 1, name="sink")
        else:
            mdl.addConstr(out_f == in_f, name=f"flow{node}")

    # (2) slowest stage
    for (i, j, q), T in cand.items():
        mdl.addConstr(tau >= T * x[i, j, q], name=f"T_{i}_{j}_{q}")

    # (3) communication at every cut after layer j < L
    for j in range(1, L):
        cut = gp.quicksum(x[i, jj, q] for (i, jj, q) in cand if jj == j)
        mdl.addConstr(tau >= 2 * c[j - 1] * cut, name=f"cut{j}")

    # (4) PEs
    mdl.addConstr(gp.quicksum(q * x[i, j, q] for (i, j, q) in cand) <= P,
                  name="pes")
    # (memory is enforced by not creating infeasible (i,j) stages)

    mdl.optimize()
    build_solve = time.time() - t0
    if mdl.SolCount == 0:
        return dict(status=mdl.Status, tau=None, stages=None,
                    time=build_solve, gap=None, nvars=mdl.NumVars,
                    ncons=mdl.NumConstrs)
    stages = sorted((i, j, q) for (i, j, q) in cand if x[i, j, q].X > 0.5)
    return dict(status=mdl.Status, tau=tau.X, stages=stages, time=build_solve,
                gap=mdl.MIPGap, nvars=mdl.NumVars, ncons=mdl.NumConstrs)


# --------------------------------------------- independent optimality checks
def solve_dp(inst):
    """Exact dynamic programming (PipeDream style), O(L^2 * P^2)."""
    L, P, c = inst["L"], inst["P"], inst["c"]
    INF = float("inf")
    # best[j][p] = min bottleneck for layers 1..j using exactly p PEs
    best = [[INF] * (P + 1) for _ in range(L + 1)]
    best[0][0] = 0.0
    for j in range(1, L + 1):
        for i in range(1, j + 1):
            if stage_mem(inst, i, j) > inst["Gamma"]:
                continue
            cut = 2 * c[i - 2] if i > 1 else 0.0
            for q in range(1, P + 1):
                T = max(stage_time(inst, i, j, q), cut)
                for p in range(q, P + 1):
                    prev = best[i - 1][p - q]
                    if prev < INF:
                        val = max(prev, T)
                        if val < best[j][p]:
                            best[j][p] = val
    return min(best[L][1:]) if min(best[L][1:]) < INF else None


def solve_bruteforce(inst):
    """Enumerate every cut set and every replica assignment (tiny L only)."""
    L, P, c = inst["L"], inst["P"], inst["c"]
    best = float("inf")
    for ncuts in range(0, L):
        for cuts in itertools.combinations(range(1, L), ncuts):
            bounds = [0, *cuts, L]
            segs = [(bounds[k] + 1, bounds[k + 1]) for k in range(len(bounds) - 1)]
            if any(stage_mem(inst, i, j) > inst["Gamma"] for i, j in segs):
                continue
            if len(segs) > P:
                continue
            for qs in itertools.product(range(1, P + 1), repeat=len(segs)):
                if sum(qs) > P:
                    continue
                tau = max(stage_time(inst, i, j, q) for (i, j), q in zip(segs, qs))
                for j in cuts:
                    tau = max(tau, 2 * c[j - 1])
                best = min(best, tau)
    return None if best == float("inf") else best


# ----------------------------------------------------------------- baselines
def one_pe_per_stage_equal(inst):
    """Naive baseline: one stage per PE, equal number of layers, q = 1."""
    L, P, c = inst["L"], inst["P"], inst["c"]
    k = min(P, L)
    size = -(-L // k)
    segs, s = [], 1
    while s <= L:
        segs.append((s, min(s + size - 1, L)))
        s += size
    if any(stage_mem(inst, i, j) > inst["Gamma"] for i, j in segs):
        return None
    tau = max(stage_time(inst, i, j, 1) for i, j in segs)
    for _, j in segs[:-1]:
        tau = max(tau, 2 * c[j - 1])
    return tau


if __name__ == "__main__":
    inst = make_instance(L=8, P=4, seed=1)
    r = solve_gurobi(inst)
    print("Gurobi :", r["tau"], r["stages"])
    print("DP     :", solve_dp(inst))
    print("Brute  :", solve_bruteforce(inst))
