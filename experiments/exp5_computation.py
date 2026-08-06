"""
Experiment 5 -- Computation time and complexity (R1.1, R5.7).

Three scalings are measured, each with the others held fixed:

    (a) data length N          10 ... 500
    (b) state dimension n      2 ... 10
    (c) generator count s_A    n ... n^2   (template richness)

against least squares as the reference.  Empirical exponents are obtained by a
log-log fit, which is the number to quote next to the analytical count:

  Least squares          O(N (n+m)^2)          one pseudo-inverse
  DCM (12)               decision variables  n(n+m) + n + 1 + s_w N
  ICM (10)               decision variables  n(n+m) + n + 1 + (s_A+s_B+s_w) N
                                             + (r_A s_A + r_B s_B) containment
                         equality constraints n N
  SMI [13], [14]         2 n (n+m) linear programs over N inequalities
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from icm_control.baselines import least_squares, set_membership, tube_baseline
from icm_control.identification import solve_dcm, solve_icm
from icm_control.reporting import (COL_H, COL_W, label, save_fig, save_table,
                                   style, use_paper_style)
from icm_control.sets import identity_blocks, row_blocks, structured_prior
from icm_control.systems import (LinearSystem, collect_data,
                                 operating_region_from_data)

import matplotlib.pyplot as plt


def random_plant(n, m, rng, spectral_radius=0.9):
    A = rng.normal(size=(n, n))
    A *= spectral_radius / max(np.abs(np.linalg.eigvals(A)).max(), 1e-9)
    B = rng.normal(size=(n, m)) / np.sqrt(n)
    return LinearSystem(A, B, name=f"rand{n}")


def timed_case(n, m, N, gen_mode, seed, alpha=2.0, template=0.01,
               rel_err=0.15, run_smi=True):
    rng = np.random.default_rng(seed)
    plant = random_plant(n, m, rng)
    Gw_true = alpha * template * np.eye(n)
    Gw_template = template * np.eye(n)
    Phi, U, Xp, Xs = collect_data(plant, N, 0.1 * np.ones(n), rng,
                                  u_range=0.5, w_gen=Gw_true,
                                  reset_radius=10.0)
    X_region = operating_region_from_data(Xs)

    if gen_mode == "row":
        GA = row_blocks(n, n, rel_err)
        GB = row_blocks(n, m, rel_err)
    else:
        GA = identity_blocks(n, n, rel_err)
        GB = identity_blocks(n, m, rel_err)
    A_side, _ = structured_prior(plant.A, rel_err * (np.abs(plant.A) + 0.01))
    B_side, _ = structured_prior(plant.B, rel_err * (np.abs(plant.B) + 0.01))

    rows = []
    ls = least_squares(Phi, U, Xp)
    rows.append({"method": "LS", "time": ls.solve_time, "n_var": n * (n + m),
                 "feasible": True})

    tb = tube_baseline(Phi, U, Xp, Gw_template, X_region)
    rows.append({"method": "Tube-MPC", "time": tb.solve_time,
                 "n_var": n * (n + m) + n + 1 + n * N, "feasible": tb.feasible})

    for name, fn in (("DCM", solve_dcm), ("ICM", solve_icm)):
        res = fn(Phi=Phi, U=U, Xp=Xp, Gw=Gw_template, X_region=X_region,
                 GA_blocks=GA, GB_blocks=GB, A_side=A_side, B_side=B_side)
        rows.append({"method": name, "time": res.solve_time,
                     "n_var": res.n_variables, "n_con": res.n_constraints,
                     "feasible": res.feasible})

    if run_smi:
        smi = set_membership(Phi, U, Xp, 2.0 * alpha * template * np.ones(n))
        rows.append({"method": "SMI", "time": smi.solve_time,
                     "n_var": 2 * n * (n + m), "feasible": smi.feasible})

    for r in rows:
        r.update({"n": n, "m": m, "N": N, "s_A": len(GA),
                  "gen_mode": gen_mode, "seed": seed})
    return rows


def loglog_slope(x, y):
    """Log-log slope, plus the number of points it was fit on.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan, int(ok.sum())
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0]), int(ok.sum())


def main(args):
    use_paper_style()
    rows = []

    # (a) scaling in N
    for N in args.Ns:
        for t in range(args.repeats):
            rows += timed_case(n=3, m=3, N=N, gen_mode="entry", seed=11 + t,
                               run_smi=(N <= 200))
        print(f"  N={N} done", flush=True)
    # (b) scaling in n
    for n in args.dims:
        for t in range(args.repeats):
            rows += timed_case(n=n, m=max(1, n // 2), N=args.N_fixed,
                               gen_mode="entry", seed=101 + t,
                               run_smi=(n <= 6))
        print(f"  n={n} done", flush=True)
    # (c) scaling in generator count
    for mode in ("row", "entry"):
        for t in range(args.repeats):
            rows += timed_case(n=4, m=2, N=args.N_fixed, gen_mode=mode,
                               seed=303 + t, run_smi=False)

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "results" / "tables"
    df.to_csv(out / "exp5_raw.csv", index=False)

    dN = df[(df.n == 3) & (df.gen_mode == "entry")]
    dn = df[(df.N == args.N_fixed) & (df.gen_mode == "entry") & (df.n != 3)]

    # ---- Fig: time vs N and time vs n -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2.06, COL_H))
    for m in ("LS", "Tube-MPC", "SMI", "DCM", "ICM"):
        s = dN[dN.method == m]
        if not s.empty:
            g = s.groupby("N")["time"].mean()
            axes[0].plot(g.index, g.values, label=label(m), **style(m))
        s = dn[dn.method == m]
        if not s.empty:
            g = s.groupby("n")["time"].mean()
            axes[1].plot(g.index, g.values, label=label(m), **style(m))
    axes[0].set_xlabel("Data length $N$")
    axes[1].set_xlabel("State dimension $n$")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel("Solve time (s)")
    axes[0].set_xscale("log")
    axes[0].legend(loc="upper left", fontsize=5.5)
    save_fig(fig, "exp5_computation_time")

    # ---- Fig: problem size -------------------------------------------------
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    for m in ("DCM", "ICM"):
        s = dN[dN.method == m]
        g = s.groupby("N")["n_var"].mean()
        ax.plot(g.index, g.values, label=label(m), **style(m))
    ax.set_xlabel("Data length $N$")
    ax.set_ylabel("Decision variables")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    save_fig(fig, "exp5_problem_size")

    # ---- Table: empirical exponents ---------------------------------------
    recs = []
    for m in ("LS", "Tube-MPC", "SMI", "DCM", "ICM"):
        sN = dN[dN.method == m].groupby("N")["time"].mean()
        sn = dn[dn.method == m].groupby("n")["time"].mean()
        ref = dN[(dN.method == m) & (dN.N == args.N_fixed)]["time"]
        recs.append({"Method": label(m),
                     "exp_N": loglog_slope(sN.index, sN.values)[0],
                     "exp_n": loglog_slope(sn.index, sn.values)[0],
                     "pts_N": loglog_slope(sN.index, sN.values)[1],
                     "pts_n": loglog_slope(sn.index, sn.values)[1],
                     "t_ref": ref.mean() if not ref.empty else np.nan})
    tab = pd.DataFrame(recs)
    ls_ref = tab.loc[tab.Method == label("LS"), "t_ref"].values
    tab["overhead"] = tab["t_ref"] / (ls_ref[0] if len(ls_ref) and ls_ref[0] > 0 else np.nan)
    tab = tab.rename(columns={"exp_N": "slope in $N$", "exp_n": "slope in $n$",
                              "pts_N": "pts ($N$)", "pts_n": "pts ($n$)",
                              "t_ref": "$t$ (s) at ref.", "overhead": "$\\times$ LS"})
    save_table(tab, "exp5_complexity")

    # ---- Table: generator template cost -----------------------------------
    dg = df[(df.n == 4) & df.method.isin(["DCM", "ICM"])]
    if not dg.empty:
        gt = (dg.groupby(["method", "gen_mode"])
              .agg(s_A=("s_A", "mean"), n_var=("n_var", "mean"),
                   time=("time", "mean")).reset_index())
        gt = gt.rename(columns={"method": "Method", "gen_mode": "Template",
                                "s_A": "$s_A$", "n_var": "vars",
                                "time": "$t$ (s)"})
        save_table(gt, "exp5_generator_cost",
                   caption="Cost of the generator template ($n=4$, $m=2$): one "
                           "generator per row versus one per matrix entry.")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+",
                    default=[10, 20, 50, 100, 200, 500])
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    ap.add_argument("--N-fixed", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    main(ap.parse_args())
