"""
Experiment 8 -- certified control of the TRUE linear plant (Section 3.3).

For each dataset of N independent one-step experiments:
  * certified set Theta_c = SPS boxes (m=300, q=1 per row, delta_A = n q/m)
    intersected with the prior box;
  * disturbance set W_c = 0 + tau_hat [-1,1]^n (fixed center c0 = 0);
  * LP (31) on (Theta_c, W_c) with u_max = 1: lambda* and the controller at
    lambda = 0.9;
  * the same for the ICM (gamma = 8) and for the prior box alone, on the same data;
  * validation on the TRUE plant: one 60-step rollout per dataset (procedure-
    level estimate) and 250 rollouts per certified controller (controller-level
    estimate), x0 uniform on S_s, disturbances from the true distribution.

Theorem (finite-horizon safety):  P(certified and exit by T) <= delta_A + beta + T eps_N
with eps_N = 1 - beta^(1/N).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import warnings
warnings.filterwarnings("ignore")

from icm_control.certification import (certified_set, coverage_level,
                                       disturbance_radius)
from icm_control.control_linear import (UncertainModel, lambda_contractive_lp,
                                        min_feasible_lambda, to_uncertain_model)
from icm_control.identification import solve_icm
from icm_control.protocol import REG, U_MAX_LIN, X_RADIUS_LIN
from icm_control.reporting import save_table
from icm_control.systems import (box_region, collect_independent,
                                 linear_side_info, safe_set_box,
                                 third_order_system)

SIGMA = 0.01


def box_model(C, Rd, n, m, tau, name):
    """Entrywise box [C - Rd, C + Rd] for [A B] and W = tau [-1,1]^n."""
    GA, GB = [], []
    for i in range(n):
        for j in range(n):
            if Rd[i, j] > 1e-12:
                E = np.zeros((n, n)); E[i, j] = Rd[i, j]; GA.append(E)
        for j in range(m):
            if Rd[i, n + j] > 1e-12:
                E = np.zeros((n, m)); E[i, j] = Rd[i, n + j]; GB.append(E)
    return UncertainModel(C[:, :n], C[:, n:], np.zeros(n), GA, GB,
                          tau * np.eye(n), name)


def rollouts(plant, K, safe, rng, n_traj, T):
    """Fraction of TRUE-plant trajectories that leave S_s within T steps."""
    n = plant.n
    bad = 0
    for _ in range(n_traj):
        x = rng.uniform(-1, 1, n)
        for _ in range(T):
            w = SIGMA * rng.uniform(-1, 1, n)
            x = plant.A @ x + plant.B @ (K @ x) + w
            if np.any(safe.H @ x > safe.h + 1e-9):
                bad += 1
                break
    return bad


def one_trial(seed, N, args):
    rng = np.random.default_rng(seed)
    plant = third_order_system(); n, m = plant.n, plant.B.shape[1]
    safe = safe_set_box(n, 1.0); Mx = safe.max_norm_bound(p=np.inf)
    A_side, GA, B_side, GB = linear_side_info(plant.A, plant.B, rel_err=args.rel_err)
    Alo, Ahi = A_side.interval(); Blo, Bhi = B_side.interval()
    Cp = np.hstack([0.5 * (Alo + Ahi), 0.5 * (Blo + Bhi)])
    Rp = np.hstack([0.5 * (Ahi - Alo), 0.5 * (Bhi - Blo)])
    Phi, U, Xp, _ = collect_independent(plant, N, rng, x_radius=X_RADIUS_LIN,
                                        u_range=0.5, w_gen=SIGMA * np.eye(n))
    theta_tr = np.hstack([plant.A, plant.B])
    rows = []

    def evaluate(name, model, contains, width, tau):
        row = {"seed": seed, "N": N, "method": name, "contains_true": contains,
               "mean_halfwidth": width, "w_radius": tau}
        row["lambda_min"] = min_feasible_lambda(model, safe, Mx, u_max=U_MAX_LIN)
        ctrl = lambda_contractive_lp(model, safe, lam=args.lam, Mx=Mx, u_max=U_MAX_LIN)
        row["certified"] = bool(ctrl.feasible)
        if ctrl.feasible:
            vr = np.random.default_rng(seed + 7_777)
            row["exit_one"] = rollouts(plant, ctrl.K, safe, vr, 1, args.T)
            row["exit_many"] = rollouts(plant, ctrl.K, safe, vr, args.n_val, args.T)
        rows.append(row)

    # certified set
    out = certified_set(Phi, U, Xp, Cp, Rp, args.m_sps, args.q, rng)
    if out is None:
        rows.append({"seed": seed, "N": N, "method": "certified", "empty": True})
    else:
        C, Rd = out
        tau = disturbance_radius(Phi, U, Xp, C, Rd)
        evaluate("certified", box_model(C, Rd, n, m, tau, "certified"),
                 bool(np.all(np.abs(theta_tr - C) <= Rd + 1e-12)),
                 float(Rd[:, :n].mean()), tau)
    # prior box alone (same disturbance construction)
    tau_p = disturbance_radius(Phi, U, Xp, Cp, Rp)
    evaluate("prior only", box_model(Cp, Rp, n, m, tau_p, "prior"), True,
             float(Rp[:, :n].mean()), tau_p)
    # ICM on the same data
    res = solve_icm(Phi=Phi, U=U, Xp=Xp, Gw=args.gamma * SIGMA * np.eye(n),
                    X_region=box_region(n, X_RADIUS_LIN), GA_blocks=GA,
                    GB_blocks=GB, A_side=A_side, B_side=B_side,
                    mu_floor=args.mu_floor, reg=REG)
    if res.feasible:
        Alo_i, Ahi_i = res.A_set.interval(); Blo_i, Bhi_i = res.B_set.interval()
        cont = bool(np.all((plant.A >= Alo_i - 1e-12) & (plant.A <= Ahi_i + 1e-12))
                    and np.all((plant.B >= Blo_i - 1e-12) & (plant.B <= Bhi_i + 1e-12)))
        hw = np.abs(res.Gw) @ res.muw
        evaluate("ICM", to_uncertain_model(res), cont,
                 float(np.mean(0.5 * (Ahi_i - Alo_i))), float(np.max(np.abs(res.cw) + hw)))
    return rows


def main(args):
    rows = []
    for N in args.Ns:
        for t in range(args.trials):
            rows += one_trial(9000 + 53 * t + N, N, args)
            print(f"  N={N} trial {t} done", flush=True)
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "results" / "tables"
    df.to_csv(out / "exp8_raw.csv", index=False)

    delta_A = args.n_rows * args.q / args.m_sps
    tab = []
    for (N, meth), g in df.groupby(["N", "method"], sort=False):
        eps = coverage_level(N, args.beta)
        cert = g["certified"].fillna(False).astype(bool)
        tab.append({
            "N": N, "Method": meth,
            "contains true (%)": 100 * g["contains_true"].fillna(False).astype(bool).mean(),
            "half-width A": g["mean_halfwidth"].median(),
            "w radius": g["w_radius"].median(),
            "lambda*": g["lambda_min"].median(),
            "certified (%)": 100 * cert.mean(),
            "exit, 1 per dataset": int(g.loc[cert, "exit_one"].sum()),
            "datasets": int(cert.sum()),
            "exit, per controller (max %)": 100 * (g.loc[cert, "exit_many"] / args.n_val).max(),
            "bound": (1 - delta_A - args.beta - args.T * eps) if meth == "certified" else np.nan,
        })
    tab = pd.DataFrame(tab)
    save_table(tab, "exp8_certified")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+", default=[400, 1000, 3500])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--m-sps", type=int, default=300)
    ap.add_argument("--q", type=int, default=1)
    ap.add_argument("--n-rows", type=int, default=3)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--T", type=int, default=60)
    ap.add_argument("--lam", type=float, default=0.90)
    ap.add_argument("--n-val", type=int, default=250)
    ap.add_argument("--rel-err", type=float, default=0.15)
    ap.add_argument("--gamma", type=float, default=8.0)
    ap.add_argument("--mu-floor", type=float, default=0.20)
    main(ap.parse_args())
