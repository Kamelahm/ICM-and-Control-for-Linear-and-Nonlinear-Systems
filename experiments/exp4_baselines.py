"""
Experiment 4 -- Stronger baselines.

Beyond DCM, three further methods are run on identical data:

  LS         least squares point estimate, no uncertainty quantification
  SMI        set-membership identification [13], [14] -- returns the feasible
             parameter set, but *requires a known disturbance bound*
  Tube-ZPC   data-driven tube / zonotopic predictive control [24] -- point
             nominal model with all uncertainty in one additive zonotope

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from icm_control.baselines import least_squares, set_membership, tube_baseline
from icm_control.control_linear import (closed_loop_violation,
                                        lambda_contractive_lp,
                                        to_uncertain_model)
from icm_control.evaluation import coverage, enclosure_size, sample_validation
from icm_control.identification import solve_dcm, solve_icm
from icm_control.reporting import (COL_H, COL_W, label, save_fig, save_table,
                                   style, use_paper_style)
from icm_control.sets import Zonotope
from icm_control.systems import (collect_data, linear_side_info,
                                 operating_region_from_data, safe_set_box,
                                 third_order_system)

import matplotlib.pyplot as plt


def one_trial(seed, N, alpha=2.0, template=0.01, rel_err=0.15,
              smi_betas=(0.5, 1.0, 2.0), n_val=40, lam=0.90,
              gamma=8.0, mu_floor=0.20):
    rng = np.random.default_rng(seed)
    plant = third_order_system()
    n = plant.n
    Gw_true = alpha * template * np.eye(n)
    Gw_template = gamma * template * np.eye(n)
    W_true = Zonotope(np.zeros(n), Gw_true)
    w_true_bound = alpha * template * np.ones(n)

    Phi, U, Xp, Xs = collect_data(plant, N, 0.5 * np.ones(n), rng,
                                  u_range=0.5, w_gen=Gw_true)
    X_region = operating_region_from_data(Xs)
    safe = safe_set_box(n, 1.0)
    Mx = safe.max_norm_bound(p=np.inf)
    r_data = float(np.max(np.abs(Xs)))
    val = {f: sample_validation(plant, n_val, np.random.default_rng(seed + 31),
                                radius=f * r_data, w_gen=Gw_true)
           for f in (1.0, 2.0)}
    A_side, GA, B_side, GB = linear_side_info(plant.A, plant.B, rel_err=rel_err)

    models = {}
    models["LS"] = least_squares(Phi, U, Xp)
    models["LS"].W_set = Zonotope(np.zeros(n), np.zeros((n, 1)))
    models["Tube-MPC"] = tube_baseline(Phi, U, Xp, Gw_template, X_region)
    for b in smi_betas:
        models[f"SMI(beta={b:g})"] = set_membership(Phi, U, Xp, b * w_true_bound)
    models["DCM"] = solve_dcm(Phi=Phi, U=U, Xp=Xp, Gw=Gw_template,
                              X_region=X_region, GA_blocks=GA, GB_blocks=GB,
                              A_side=A_side, B_side=B_side, mu_floor=mu_floor)
    models["ICM"] = solve_icm(Phi=Phi, U=U, Xp=Xp, Gw=Gw_template,
                              X_region=X_region, GA_blocks=GA, GB_blocks=GB,
                              A_side=A_side, B_side=B_side, mu_floor=mu_floor)

    rows = []
    for name, mdl in models.items():
        row = {"seed": seed, "N": N, "alpha": alpha, "method": name,
               "needs_w_bound": name.startswith("SMI"),
               "id_feasible": bool(getattr(mdl, "feasible", False)),
               "id_time": getattr(mdl, "solve_time", np.nan)}
        if row["id_feasible"]:
            for f, (Pv, Uv, Xpv, _) in val.items():
                row[f"coverage_x{f:g}"] = coverage(mdl, Pv, Uv, Xpv)
                row[f"encl_x{f:g}"] = enclosure_size(mdl, Pv, Uv)
            row["coverage_train"] = coverage(mdl, Phi, U, Xp)
            ctrl = lambda_contractive_lp(to_uncertain_model(mdl), safe, lam, Mx)
            row["lp_feasible"] = ctrl.feasible
            row["lp_time"] = ctrl.solve_time
            if ctrl.feasible:
                v, worst = closed_loop_violation(
                    plant, ctrl.K, safe, W_true,
                    np.random.default_rng(seed + 202), n_traj=25, horizon=30)
                row["cl_violation"] = v
                row["cl_worst"] = worst
        rows.append(row)
    return rows


def main(args):
    use_paper_style()
    rows = []
    for N in args.data_lengths:
        for t in range(args.trials):
            rows += one_trial(seed=606 + 19 * t + N, N=N, alpha=args.alpha, gamma=args.gamma, mu_floor=args.mu_floor)
        print(f"  N={N} done", flush=True)
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "results" / "tables"
    df.to_csv(out / "exp4_raw.csv", index=False)

    order = [m for m in ["LS", "Tube-MPC", "SMI(beta=0.5)", "SMI(beta=1)",
                         "SMI(beta=2)", "DCM", "ICM"] if m in set(df.method)]

    # ---- Fig: coverage vs enclosure size (the real trade-off) -------------
    tradeoff_order = [m for m in order if m != "LS"]
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    for m in tradeoff_order:
        sub = df[(df.method == m) & df.id_feasible]
        if sub.empty:
            continue
        base = m.split("(")[0]
        st = dict(style(base))
        if bool(sub["needs_w_bound"].max()):
            st["markerfacecolor"] = "none"
        ax.errorbar(sub["encl_x1"].mean(), sub["coverage_x1"].mean() * 100,
                    xerr=sub["encl_x1"].std(), yerr=sub["coverage_x1"].std() * 100,
                    capsize=2, label=m, **st)
    ax.set_xlabel(r"Certified enclosure size $\sum_j \|G_{:,j}\|_1$")
    ax.set_ylabel("Validation coverage (\\%)")
    ax.set_xscale("log")
    ax.set_ylim(-3, 103)
    ax.legend(loc="lower right", fontsize=5.5,
              title=r"open $=$ needs $\bar w$", title_fontsize=5.5)
    save_fig(fig, "exp4_baselines_tradeoff")

    # ---- Table -------------------------------------------------------------
    agg = df.groupby("method").agg(
        needs_w=("needs_w_bound", "max"),
        id_feas=("id_feasible", "mean"),
        cov_tr=("coverage_train", "mean"),
        cov1=("coverage_x1", "mean"),
        cov2=("coverage_x2", "mean"),
        encl=("encl_x1", "mean"),
        lp=("lp_feasible", "mean"),
        viol=("cl_violation", "mean"),
        t_id=("id_time", "mean")).reindex(order).reset_index()
    for c in ("id_feas", "cov_tr", "cov1", "cov2", "lp", "viol"):
        agg[c] = 100 * agg[c]
    agg["needs_w"] = ["yes" if v else "no" for v in agg["needs_w"]]
    agg = agg.rename(columns={
        "method": "Method", "needs_w": "needs $\\bar w$",
        "id_feas": "ID feas.", "cov_tr": "Cov. train", "cov1": "Cov. $1\\times$",
        "cov2": "Cov. $2\\times$", "encl": "Encl. size", "lp": "LP feas.",
        "viol": "CL viol.", "t_id": "$t_{\\rm id}$ (s)"})
    save_table(agg, "exp4_baselines")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--data-lengths", type=int, nargs="+", default=[10, 30])
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="matches the regime of exp1 Fig. 2 so the two "
                         "comparisons are read on the same footing")
    ap.add_argument("--gamma", type=float, default=16.0,
                    help="disturbance template scale for DCM/ICM; the "
                         "SMI baselines are given the TRUE bound instead")
    ap.add_argument("--mu-floor", type=float, default=0.20,
                    help="lower bound on mu_w; keeps What full dimensional")
    main(ap.parse_args())
