"""
Experiment 1 -- Three-way ablation: DCM / ICM-without-side-information / ICM.

The three arms differ in exactly one feature each:

    DCM        point dynamics + additive disturbance set        (Eq. 12)
    ICM-noSI   zonotopic dynamics, no prior                     (Eq. 10 w/o 10e-10f)
    ICM        zonotopic dynamics + prior containment           (Eq. 10)

so DCM -> ICM-noSI isolates the *structural* contribution (matrix zonotopes)
and ICM-noSI -> ICM isolates the *side-information* contribution.  Objective,
disturbance template, operating region, solver and data are identical across
arms, so every difference in the tables is attributable to the single feature
being switched.

Outputs (two figures, per the paper's figure budget)
----------------------------------------------------
  exp1_feasibility_vs_noise.pdf      Fig 1: zonotopic sets keep synthesis feasible
  exp1_coverage_vs_extrapolation.pdf Fig 2: the prior is what makes the sets contain
                                     the truth, and the gap widens on extrapolation
  exp1_ablation.tex, exp1_error_sources.tex, exp1_raw_*.csv

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from icm_control.control_linear import (closed_loop_violation,
                                        lambda_contractive_lp,
                                        min_feasible_lambda,
                                        to_uncertain_model)
from icm_control.evaluation import (coverage, enclosure_by_source,
                                    enclosure_size, representability_gap,
                                    sample_validation)
from icm_control.identification import (solve_dcm, solve_icm,
                                        solve_icm_no_side_info)
from icm_control.reporting import (COL_H, COL_W, label, save_fig, save_table,
                                   style, use_paper_style, wilson_ci)
from icm_control.sets import Zonotope
from icm_control.systems import (collect_data, linear_side_info,
                                 operating_region_from_data, safe_set_box,
                                 third_order_system)

import matplotlib.pyplot as plt

ARMS = [("DCM", solve_dcm), ("ICM-noSI", solve_icm_no_side_info),
        ("ICM", solve_icm)]
ORDER = [a for a, _ in ARMS]


def one_trial(seed, alpha, N, lam_grid, rel_err=0.15, n_val=40,
              extrapolation=(1.0, 2.0, 3.0), lam_ref=0.90, template=0.01,
              mu_floor=0.20, gamma=8.0):
    """
    The disturbance *template* G^w is the modeller's prior guess and is held
    fixed at `template * I`; the *true* disturbance is alpha_1 * template * I.
    """
    rng = np.random.default_rng(seed)
    plant = third_order_system()
    n = plant.n
    Gw_true = alpha * template * np.eye(n)
    Gw_template = gamma * template * np.eye(n)
    W_true = Zonotope(np.zeros(n), Gw_true)

    Phi, U, Xp, Xs = collect_data(plant, N, 0.5 * np.ones(n), rng,
                                  u_range=0.5, w_gen=Gw_true)
    X_region = operating_region_from_data(Xs)
    safe = safe_set_box(n, 1.0)
    Mx = safe.max_norm_bound(p=np.inf)
    A_side, GA, B_side, GB = linear_side_info(plant.A, plant.B, rel_err=rel_err)

    r_data = float(np.max(np.abs(Xs)))
    val = {f: sample_validation(plant, n_val,
                                np.random.default_rng(seed + 10_000),
                                radius=f * r_data, w_gen=Gw_true)
           for f in extrapolation}

    id_rows, lp_rows = [], []
    for name, solver in ARMS:
        res = solver(Phi=Phi, U=U, Xp=Xp, Gw=Gw_template, X_region=X_region,
                     GA_blocks=GA, GB_blocks=GB, A_side=A_side, B_side=B_side,
                     mu_floor=mu_floor)
        row = {"seed": seed, "alpha": alpha, "N": N, "method": name,
               "id_feasible": res.feasible, "sX": res.sX,
               "id_time": res.solve_time, "n_var": res.n_variables}
        if res.feasible:
            row["coverage_train"] = coverage(res, Phi, U, Xp)
            for f, (Pv, Uv, Xpv, _) in val.items():
                row[f"coverage_x{f:g}"] = coverage(res, Pv, Uv, Xpv)
                row[f"encl_x{f:g}"] = enclosure_size(res, Pv, Uv)
            Pv, Uv, _, _ = val[1.0]
            row.update({f"src_{k}": v
                        for k, v in enclosure_by_source(res, Pv, Uv).items()})
            row["repr_gap"] = representability_gap(res, plant.A, plant.B, Pv, Uv)
            row["center_err"] = float(np.max(np.abs(res.CA - plant.A)))

            model = to_uncertain_model(res)
            row["lambda_min"] = min_feasible_lambda(model, safe, Mx)
            for lam in lam_grid:
                ctrl = lambda_contractive_lp(model, safe, lam=lam, Mx=Mx)
                lrow = {"seed": seed, "alpha": alpha, "N": N, "method": name,
                        "lam": lam, "lp_feasible": ctrl.feasible,
                        "rho": ctrl.rho, "lp_time": ctrl.solve_time}
                if ctrl.feasible and abs(lam - lam_ref) < 1e-9:
                    v, worst = closed_loop_violation(
                        plant, ctrl.K, safe, W_true,
                        np.random.default_rng(seed + 555), n_traj=20, horizon=30)
                    lrow["cl_violation"] = v
                    lrow["cl_worst_residual"] = worst
                lp_rows.append(lrow)
        else:
            for lam in lam_grid:
                lp_rows.append({"seed": seed, "alpha": alpha, "N": N,
                                "method": name, "lam": lam,
                                "lp_feasible": False, "rho": np.nan,
                                "lp_time": np.nan})
        id_rows.append(row)
    return id_rows, lp_rows


def _feas_curve(ax, sub, xcol):
    for name in ORDER:
        s = sub[sub.method == name]
        if s.empty:
            continue
        g = s.groupby(xcol)["lp_feasible"]
        rate = g.mean() * 100
        lo = [wilson_ci(int(k), int(nn))[0] * 100 for k, nn in zip(g.sum(), g.count())]
        hi = [wilson_ci(int(k), int(nn))[1] * 100 for k, nn in zip(g.sum(), g.count())]
        ax.plot(rate.index, rate.values, label=label(name), **style(name))
        ax.fill_between(rate.index, lo, hi, alpha=0.15, color=style(name)["color"])
    ax.set_ylim(-3, 103)


def main(args):
    use_paper_style()
    id_rows, lp_rows = [], []
    lam_grid = list(np.round(np.arange(0.70, 1.001, 0.05), 3))
    gammas = sorted({float(args.gamma_feas), float(args.gamma_cov)})
    for gam in gammas:
        for N in args.data_lengths:
            for a in args.alphas:
                for t in range(args.trials):
                    i_r, l_r = one_trial(seed=int(1000 * a) + 17 * t + 7 * N,
                                         alpha=float(a), N=N, lam_grid=lam_grid,
                                         n_val=args.n_val, rel_err=args.rel_err,
                                         lam_ref=args.lam_fixed,
                                         mu_floor=args.mu_floor, gamma=gam)
                    for r in i_r + l_r:
                        r["gamma"] = gam
                    id_rows += i_r
                    lp_rows += l_r
                print(f"  gamma={gam:g}, N={N}, alpha={a} done", flush=True)

    df = pd.DataFrame(id_rows)
    lp = pd.DataFrame(lp_rows)
    out = Path(__file__).resolve().parent.parent / "results" / "tables"
    df.to_csv(out / "exp1_raw_identification.csv", index=False)
    lp.to_csv(out / "exp1_raw_synthesis.csv", index=False)

    # -------- Fig: feasibility vs alpha (lambda fixed) ---------------------
    fig, axes = plt.subplots(1, len(args.data_lengths),
                             figsize=(COL_W * 2.06, COL_H), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, N in zip(axes, args.data_lengths):
        _feas_curve(ax, lp[(lp.N == N)
                           & (np.abs(lp.lam - args.lam_fixed) < 1e-9)
                           & (np.abs(lp.gamma - args.gamma_feas) < 1e-9)],
                    "alpha")
        ax.set_xlabel(r"$\alpha_1$ (noise scaling factor)")
        ax.set_title(f"$N = {N}$")
    axes[0].set_ylabel("Synthesis feasibility (\\%)")
    axes[0].legend(loc="lower left")
    save_fig(fig, "exp1_feasibility_vs_noise")

    # -------- Fig: coverage vs extrapolation -------------------------------
    cov_cols = [c for c in df.columns if c.startswith("coverage_x")]
    factors = [float(c.split("x")[1]) for c in cov_cols]
    cov_df = df[(np.abs(df.alpha - args.alpha_cov) < 1e-9)
                & (np.abs(df.gamma - args.gamma_cov) < 1e-9)
                & df.id_feasible]
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    encl_note = []
    for name in ORDER:
        sub = cov_df[cov_df.method == name]
        if sub.empty:
            continue
        ax.errorbar(factors, [sub[c].mean() * 100 for c in cov_cols],
                    yerr=[sub[c].sem() * 100 for c in cov_cols],
                    capsize=2, label=label(name), **style(name))
        encl_note.append(f"{label(name)}: {sub['encl_x1'].mean():.4f}")

    ax.set_xlabel(r"$r\,/\,r_{\mathrm{id}}$")
    ax.set_ylabel(r"Fraction with $x_{k+1}\in\widehat{\mathcal{X}}^{+}(x_k,u_k)$ (\%)")
    ax.set_ylim(-3, 103)
    ax.legend(loc="lower left")
    save_fig(fig, "exp1_coverage_vs_extrapolation")
    # The enclosure sizes belong in the caption: near-identical values are what
    # rule out "coverage was bought by inflating the sets".
    print(f"  [caption] Fig 2 at gamma = {args.gamma_cov:g}, alpha_1 = "
          f"{args.alpha_cov:g}; mean certified enclosure size at 1x:  "
          + ";  ".join(encl_note), flush=True)
    tstar = cov_df.groupby("method")["repr_gap"].median().reindex(ORDER)
    print("  [caption] representability index t* (median): "
          + ";  ".join(f"{label(m)}: {tstar[m]:.3f}" for m in ORDER
                       if m in tstar.index), flush=True)

    # -------- Tables --------------------------------------------------------
    ok = df[df.id_feasible & (np.abs(df.gamma - args.gamma_cov) < 1e-9)]

    key = ["seed", "alpha", "N", "gamma"]
    n_arms = df.groupby(key)["id_feasible"].transform("sum")
    common = df[(n_arms == len(ORDER)) & df.id_feasible]
    lam_common = common.groupby("method")["lambda_min"].mean().reindex(ORDER)
    n_common = common.groupby("method")["lambda_min"].count().reindex(ORDER)

    lpref = lp[(np.abs(lp.lam - args.lam_fixed) < 1e-9)
               & (np.abs(lp.gamma - args.gamma_cov) < 1e-9)]
    idfeas = df.groupby("method")["id_feasible"].mean().reindex(ORDER)
    agg = ok.groupby("method").agg(
        sX=("sX", "mean"),
        cov_tr=("coverage_train", "mean"),
        cov1=("coverage_x1", "mean"),
        cov2=("coverage_x2", "mean"),
        cov3=("coverage_x3", "mean"),
        encl=("encl_x1", "mean"),
        gap=("repr_gap", "mean"),
        cerr=("center_err", "mean"),
        t_id=("id_time", "mean")).reindex(ORDER)
    lpg = lpref.groupby("method").agg(lp_feas=("lp_feasible", "mean"),
                                      viol=("cl_violation", "mean")).reindex(ORDER)
    agg = agg.join(lpg)
    agg.insert(0, "lam_min", lam_common)
    agg.insert(0, "n_common", n_common)
    agg.insert(0, "id_feas", idfeas)
    agg = agg.reset_index()
    for c in ("cov_tr", "cov1", "cov2", "cov3", "lp_feas", "viol", "id_feas"):
        agg[c] = 100 * agg[c]
    agg["method"] = [label(m) for m in agg["method"]]
    agg = agg.rename(columns={
        "method": "Method", "sX": "$s_\\mathcal{X}$",
        "id_feas": "ID feas.", "lam_min": "$\\lambda^\\star$",
        "n_common": "$n_{\\rm common}$",
        "cov_tr": "Cov. train", "cov1": "Cov. $1\\times$", "cov2": "Cov. $2\\times$",
        "cov3": "Cov. $3\\times$", "encl": "Encl. size", "gap": "Repr. gap",
        "cerr": "$\\|C^A - A^L_{\\rm tr}\\|_\\infty$", "t_id": "$t_{\\rm id}$ (s)",
        "lp_feas": "LP feas.", "viol": "CL viol."})
    save_table(agg, "exp1_ablation",
               caption="Three-way ablation on the third-order plant (averages "
                       "over $\\alpha_1$, $N$ and trials; coverage and rates in "
                       "\\%). DCM $\\to$ ICM w/o side info.\\ isolates the "
                       "zonotopic structure, ICM w/o side info.\\ $\\to$ ICM "
                       "isolates the prior. $\\lambda^\\star$ is the smallest "
                       "contraction factor certifiable by (19), averaged over "
                       "the $n_{\\rm common}$ trials on which all three arms "
                       "are feasible.")

    src_cols = [c for c in df.columns if c.startswith("src_")]
    if src_cols:
        src = ok.groupby("method")[src_cols].mean().reindex(ORDER).reset_index()
        src["method"] = [label(m) for m in src["method"]]
        save_table(src, "exp1_error_sources",
                   caption="Mean contribution of each uncertainty source to the "
                           "certified one-step enclosure (18); DCM certifies "
                           "only the additive term.")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[1.0, 2.0, 3.0, 4.0, 5.0])
    ap.add_argument("--data-lengths", type=int, nargs="+", default=[10, 30])
    ap.add_argument("--lam-fixed", type=float, default=0.90)
    ap.add_argument("--alpha-fixed", type=float, default=3.0)
    ap.add_argument("--rel-err", type=float, default=0.15)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--mu-floor", type=float, default=0.20,
                    help="lower bound on mu_w; keeps What full dimensional")
    ap.add_argument("--gamma-feas", type=float, default=1.0,
                    help="template scale for the feasibility figure "
                         "(tight template: alpha_1 then underestimates)")
    ap.add_argument("--gamma-cov", type=float, default=8.0,
                    help="template scale for the coverage figure "
                         "(generous template: needed for Assumption 2)")
    ap.add_argument("--alpha-cov", type=float, default=1.0,
                    help="alpha_1 at which the coverage figure is drawn; "
                         "alpha_1 is the template underestimation factor, so "
                         "alpha_1 = 1 is the nominal design condition")
    main(ap.parse_args())
