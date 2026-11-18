import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gurobipy as gp

from pipeline_parallel_gurobi import (make_instance, solve_gurobi, solve_dp,
                                      solve_bruteforce, one_pe_per_stage_equal)

OUT = "/mnt/user-data/outputs/"

# ---------------------------------------------------------------- 1) optimality
rows = []
for seed in range(40):
    L = 5 + seed % 4          # 5..8 layers (brute force stays tractable)
    P = 3 + seed % 3          # 3..5 PEs
    inst = make_instance(L, P, seed=seed)
    g = solve_gurobi(inst)["tau"]
    d = solve_dp(inst)
    b = solve_bruteforce(inst)
    rows.append(dict(seed=seed, L=L, P=P, gurobi=g, dp=d, brute=b,
                     match=((g is None and d is None and b is None) or
                            (g is not None and d is not None and b is not None
                             and abs(g - d) < 1e-6 and abs(g - b) < 1e-6)),
                     feasible=g is not None))
opt = pd.DataFrame(rows)
opt.to_csv(OUT + "optimality_tests.csv", index=False)
print(f"Optimality: {opt.match.sum()}/{len(opt)} instances agree (Gurobi == DP == brute force); "
      f"{(~opt.feasible).sum()} of them are infeasible for all three methods")

# ---------------------------------------------------------------- 2) scalability
rows = []
for L in [6, 8, 10, 12, 14, 16, 18, 20]:
    for P in [4, 8]:
        times_g, times_d, gaps, taus, nv, ratio = [], [], [], [], None, []
        ok = True
        for seed in range(5):
            inst = make_instance(L, P, seed=100 + seed)
            try:
                r = solve_gurobi(inst, time_limit=60)
            except gp.GurobiError as e:   # size-limited license
                ok = False
                break
            t0 = time.time()
            d = solve_dp(inst)
            times_d.append(time.time() - t0)
            times_g.append(r["time"])
            gaps.append(r["gap"] if r["gap"] is not None else float("nan"))
            nv = r["nvars"]
            assert r["tau"] is not None and abs(r["tau"] - d) < 1e-6, (L, P, seed)
            base = one_pe_per_stage_equal(inst)
            if base:
                ratio.append(base / r["tau"])
        if not ok:
            print(f"L={L} P={P}: stopped (license size limit)")
            continue
        rows.append(dict(L=L, P=P, n_vars=nv,
                         gurobi_s=sum(times_g) / len(times_g),
                         gurobi_max_s=max(times_g),
                         dp_s=sum(times_d) / len(times_d),
                         mip_gap=max(gaps),
                         naive_over_opt=(sum(ratio) / len(ratio)) if ratio else float("nan")))
        print(rows[-1])
sc = pd.DataFrame(rows)
sc.to_csv(OUT + "scalability.csv", index=False)

# ---------------------------------------------------------------- plots
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for P, grp in sc.groupby("P"):
    ax[0].plot(grp.L, grp.gurobi_s, "o-", label=f"Gurobi, |P|={P}")
    ax[0].plot(grp.L, grp.dp_s, "x--", label=f"DP, |P|={P}")
    ax[1].plot(grp.L, grp.n_vars, "o-", label=f"|P|={P}")
ax[0].set_xlabel("layers L"); ax[0].set_ylabel("solve time (s)")
ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title("Solve time")
ax[1].set_xlabel("layers L"); ax[1].set_ylabel("binary variables")
ax[1].legend(); ax[1].set_title("Model size (~ L^2 * |P|)")
plt.tight_layout(); plt.savefig(OUT + "scalability.png", dpi=150)
print(sc.to_string(index=False))
