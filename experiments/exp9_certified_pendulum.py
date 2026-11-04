"""
Experiment 9 -- certified stability of the TRUE pendulum (Section 4, corollary).

For each dataset of N independent one-step experiments:
  * certified intervals for [A B] of (6): SPS per row (m=200, q=1, so
    delta_NL = n q / m = 0.01) intersected with the prior box;
  * SDP (61) at kappa = 0.9 on those intervals, verified kappa and rho,
    region of attraction c*_dyn and input-feasible level c*(u_max), u_max = 5;
  * the same for the ICM (Lemma 7) on the same data;
  * TRUE-plant rollouts (w = 0) from the boundary of Omega_{c*(u_max)}:
    invariance and the decay bound V(x_k) <= c(delta)^k V(x_0).
On the event that the true matrices lie in the certified intervals (probability
>= 1 - delta_NL), Theorems 2-3 apply to the true pendulum.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings
warnings.filterwarnings("ignore")

from exp2_roa import Q_fn, c_at_input, ellipse, input_for_level, input_profile
from icm_control.certification import certified_set, disturbance_radius
from icm_control.control_nonlinear import (best_delta_certificate,
                                           intervals_from_icm,
                                           nonlinearity_cancellation_sdp)
from icm_control.identification import solve_icm
from icm_control.protocol import REG, U_MAX_PEN
from icm_control.reporting import save_table
from icm_control.sets import Polytope
from icm_control.systems import Pendulum, collect_independent, pendulum_side_info

W_MAG = 0.01


def certify_design(A0, dA, Ah0, dAh, B0, dB, sysd, seed, u_max):
    des = nonlinearity_cancellation_sdp(A0, dA, Ah0, dAh, B0, dB, kappa_target=0.9)
    if not des.feasible:
        return None
    roa = best_delta_certificate(des, Q_fn, 2, A0, dA, B0, dB, x_max=np.pi,
                                 rng=np.random.default_rng(seed), c_max=0.99)
    if roa is None:
        return None
    cs, um = input_profile(des, sysd, c_hi=1.05 * roa["c_star"])
    c_um = c_at_input(cs, um, u_max, roa["c_star"])
    return des, roa, c_um


def true_rollouts(des, roa, c, sysd, n_ic=12, horizon=60):
    """Invariance of Omega_c and the decay bound on the TRUE pendulum, w = 0."""
    D, K = des.D, des.gain()
    if c <= 0:
        return np.nan, np.nan
    starts = ellipse(D, c, n=n_ic + 1)[:, :n_ic]
    inside = decay = 0
    for j in range(n_ic):
        x = starts[:, j]; V0 = x @ D @ x; ok_in = ok_dec = True
        for k in range(1, horizon + 1):
            x = sysd.step(x, K @ sysd.basis_full(x), np.zeros(2))
            V = x @ D @ x
            ok_in &= V <= c * (1 + 1e-6)
            ok_dec &= V <= V0 * roa["c_delta"] ** k * (1 + 1e-6) + 1e-12
        inside += ok_in; decay += ok_dec
    return inside / n_ic, decay / n_ic


def one_trial(seed, N, args):
    rng = np.random.default_rng(seed)
    sysd = Pendulum()
    A_tr, B_tr = sysd.true_matrices("full")
    A_side, GA, B_side, GB = pendulum_side_info(sysd, "full", rel_err=args.rel_err)
    Alo, Ahi = A_side.interval(); Blo, Bhi = B_side.interval()
    Cp = np.hstack([0.5 * (Alo + Ahi), 0.5 * (Blo + Bhi)])
    Rp = np.hstack([0.5 * (Ahi - Alo), 0.5 * (Bhi - Blo)])
    Z, U, Xp, _ = collect_independent(
        sysd, N, rng, x_radius=args.x_radius, u_range=0.5,
        w_sampler=lambda r: r.uniform(-W_MAG, W_MAG, size=2), basis=sysd.basis_full)
    theta_tr = np.hstack([A_tr, B_tr])
    rows = []

    def record(name, A0, dA, Ah0, dAh, B0, dB, contains, tau):
        row = {"seed": seed, "N": N, "method": name, "contains_true": contains,
               "w_radius": tau,
               "halfwidth_nonlinear": float(dAh.max())}
        d = certify_design(A0, dA, Ah0, dAh, B0, dB, sysd, seed, args.u_max)
        row["certified"] = d is not None
        if d is not None:
            des, roa, c_um = d
            row.update({"kappa": roa["kappa"], "rho_hat": des.rho,
                        "c_delta": roa["c_delta"], "c_star_dyn": roa["c_star"],
                        "c_star_umax": c_um})
            inv, dec = true_rollouts(des, roa, c_um, sysd)
            row["true_invariant"] = inv; row["true_decay"] = dec
            lam_max = float(np.linalg.eigvalsh(des.D)[-1])
            gap = 1 - np.sqrt(roa["c_delta"])
            row["c_req"] = lam_max * (np.sqrt(2) * tau) ** 2 / gap ** 2
        rows.append(row)

    out = certified_set(Z, U, Xp, Cp, Rp, args.m_sps, args.q, rng)
    if out is None:
        rows.append({"seed": seed, "N": N, "method": "certified", "empty": True})
    else:
        C, Rd = out
        tau = disturbance_radius(Z, U, Xp, C, Rd)
        record("certified", C[:, :2], Rd[:, :2], C[:, 2:4], Rd[:, 2:4],
               C[:, 4:], Rd[:, 4:],
               bool(np.all(np.abs(theta_tr - C) <= Rd + 1e-12)), tau)

    res = solve_icm(Phi=Z, U=U, Xp=Xp, Gw=args.gamma * W_MAG * np.eye(2),
                    X_region=Polytope(np.vstack([np.eye(2), -np.eye(2)]), 5 * np.ones(4)),
                    GA_blocks=GA, GB_blocks=GB, A_side=A_side, B_side=B_side,
                    mu_floor=args.mu_floor, reg=REG)
    if res.feasible:
        A0, dA, Ah0, dAh, B0, dB = intervals_from_icm(res, 2)
        Ai = np.hstack([A0, Ah0]); Ari = np.hstack([dA, dAh])
        cont = bool(np.all(np.abs(A_tr - Ai) <= Ari + 1e-12)
                    and np.all(np.abs(B_tr - B0) <= dB + 1e-12))
        tau_i = float(np.max(np.abs(res.cw) + np.abs(res.Gw) @ res.muw))
        record("ICM", A0, dA, Ah0, dAh, B0, dB, cont, tau_i)
    return rows


def main(args):
    rows = []
    for N in args.Ns:
        for t in range(args.trials):
            rows += one_trial(12000 + 41 * t + N, N, args)
            print(f"  N={N} trial {t} done", flush=True)
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "results" / "tables"
    df.to_csv(out / "exp9_raw.csv", index=False)
    tab = []
    for (N, meth), g in df.groupby(["N", "method"], sort=False):
        ok = g[g["certified"].fillna(False).astype(bool)]
        tab.append({"N": N, "Method": meth,
                    "contains true (%)": 100 * g["contains_true"].fillna(False).astype(bool).mean(),
                    "SDP + certificate (%)": 100 * g["certified"].fillna(False).astype(bool).mean(),
                    "rho_hat": ok["rho_hat"].median(),
                    "c(delta)": ok["c_delta"].median(),
                    "c*_dyn": ok["c_star_dyn"].median(),
                    "c*(u_max)": ok["c_star_umax"].median(),
                    "true invariant (%)": 100 * ok["true_invariant"].mean(),
                    "true decay (%)": 100 * ok["true_decay"].mean(),
                    "practical ok (%)": 100 * (ok["c_req"] <= ok["c_star_umax"]).mean()})
    tab = pd.DataFrame(tab)
    save_table(tab, "exp9_certified")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+", default=[400, 1000, 3500])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--m-sps", type=int, default=200)
    ap.add_argument("--q", type=int, default=1)
    ap.add_argument("--x-radius", type=float, default=1.0)
    ap.add_argument("--u-max", type=float, default=U_MAX_PEN)
    ap.add_argument("--rel-err", type=float, default=0.20)
    ap.add_argument("--gamma", type=float, default=27.26)
    ap.add_argument("--mu-floor", type=float, default=0.05)
    main(ap.parse_args())
