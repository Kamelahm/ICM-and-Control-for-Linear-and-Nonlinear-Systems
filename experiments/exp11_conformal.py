"""
Experiment 11 -- split-conformal least-squares baseline for Table 2 (Section 5.4).

Same training and validation data as exp4 (alpha_1 = 1, N in {10, 30}, 15
trials). Least squares is fitted on the N training samples; the residual box
half-width is the split-conformal quantile of the infinity-norm residuals on
n_cal = N additional independent calibration samples, at level 0.9, so the
total data budget is 2N.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import warnings
warnings.filterwarnings("ignore")

from icm_control.control_linear import UncertainModel, closed_loop_violation, lambda_contractive_lp
from icm_control.sets import Zonotope
from icm_control.evaluation import sample_validation
from icm_control.protocol import U_MAX_LIN, X_RADIUS_LIN
from icm_control.reporting import save_table
from icm_control.systems import collect_independent, safe_set_box, third_order_system

TEMPLATE = 0.01


def one_trial(seed, N, args):
    rng = np.random.default_rng(seed)
    plant = third_order_system(); n = plant.n
    Gw_true = args.alpha * TEMPLATE * np.eye(n)
    Phi, U, Xp, Xs = collect_independent(plant, N, rng, x_radius=X_RADIUS_LIN,
                                         u_range=0.5, w_gen=Gw_true)
    r_data = float(np.max(np.abs(Xs)))
    val = {f: sample_validation(plant, args.n_val, np.random.default_rng(seed + 31),
                                radius=f * r_data, w_gen=Gw_true)
           for f in (1.0, 2.0)}
    # least squares on the training data
    Z = np.vstack([Phi, U])
    M = Xp @ np.linalg.pinv(Z)
    # split-conformal calibration on fresh independent samples
    n_cal = N if args.n_cal is None else args.n_cal
    Pc, Uc, Xpc, _ = collect_independent(plant, n_cal, np.random.default_rng(seed + 5_000),
                                         x_radius=X_RADIUS_LIN, u_range=0.5, w_gen=Gw_true)
    scores = np.sort(np.max(np.abs(Xpc - M @ np.vstack([Pc, Uc])), axis=0))
    k = math.ceil((n_cal + 1) * args.level)
    q = float(scores[k - 1]) if k <= n_cal else np.inf
    row = {"seed": seed, "N": N, "n_cal": n_cal, "q": q}
    for f, (Pv, Uv, Xpv, _) in val.items():
        res = np.max(np.abs(Xpv - M @ np.vstack([Pv, Uv])), axis=0)
        row[f"coverage_x{f:g}"] = 100 * float(np.mean(res <= q + 1e-12))
    row["encl"] = n * q                       # generator mass of the box q*I
    safe = safe_set_box(n, 1.0)
    model = UncertainModel(M[:, :n], M[:, n:], np.zeros(n), [], [], q * np.eye(n), "LS-conformal")
    ctrl = lambda_contractive_lp(model, safe, lam=0.9, Mx=safe.max_norm_bound(p=np.inf),
                                 u_max=U_MAX_LIN)
    row["lp_feasible"] = bool(ctrl.feasible)
    if ctrl.feasible:   # same closed-loop test as exp4
        v, _ = closed_loop_violation(plant, ctrl.K, safe, Zonotope(np.zeros(n), Gw_true),
                                     np.random.default_rng(seed + 202), n_traj=25, horizon=30)
        row["cl_violation"] = v
    return row


def main(args):
    rows = [one_trial(606 + 19 * t + N, N, args) for N in args.Ns for t in range(args.trials)]
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "results" / "tables"
    df.to_csv(out / "exp11_raw.csv", index=False)
    tab = pd.DataFrame([{
        "Method": "LS + split conformal", "n_cal": "N",
        "Cov. 1x": df["coverage_x1"].mean(), "Cov. 2x": df["coverage_x2"].mean(),
        "Encl. size": df["encl"].mean(), "Synth. feas.": 100 * df["lp_feasible"].mean(),
        "CL viol.": 100 * df["cl_violation"].mean()}])
    save_table(tab, "exp11_conformal")
    print(tab.round(3).to_string(index=False))
    print(df.groupby("N")[["coverage_x1", "coverage_x2", "encl"]].mean().round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+", default=[10, 30])
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--level", type=float, default=0.9)
    ap.add_argument("--n-cal", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=40)
    main(ap.parse_args())
