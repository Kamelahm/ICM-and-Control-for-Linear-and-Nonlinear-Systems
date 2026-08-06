"""
Experiment 7 -- Generator and initial-set sensitivity (R5.2, R1.7).

Every guarantee in the paper is conditional on generator directions that are
"fixed a priori", so the natural question is how much the results depend on
that choice.  Four axes are swept:

  (a) generator magnitude   eps in the template  G = eps * (direction)
  (b) template structure    one generator per entry / per row / random / PCA of
                            the least-squares residuals (Remark 1)
  (c) generator count       s_A from n to n^2
  (d) initial set           radius of the region the data is collected in
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from icm_control.control_linear import (lambda_contractive_lp,
                                        min_feasible_lambda,
                                        to_uncertain_model)
from icm_control.evaluation import coverage, enclosure_size, sample_validation
from icm_control.identification import solve_icm
from icm_control.reporting import (COL_H, COL_W, save_fig, save_table,
                                   use_paper_style)
from icm_control.sets import (MatrixZonotope, identity_blocks, random_blocks,
                              residual_pca_blocks, row_blocks,
                              structured_prior)
from icm_control.systems import (collect_data, operating_region_from_data,
                                 safe_set_box, third_order_system)

import matplotlib.pyplot as plt

TEMPLATES = ["entry", "row", "random", "pca"]
TEMPLATE_LABEL = {"entry": "per entry", "row": "per row",
                  "random": "random", "pca": "residual PCA"}


def build_template(kind, n, p, m, eps, rng, Phi, U, Xp, n_random=None):
    if kind == "entry":
        return identity_blocks(n, p, eps), identity_blocks(n, m, eps)
    if kind == "row":
        return row_blocks(n, p, eps), row_blocks(n, m, eps)
    if kind == "random":
        k = n_random or n
        return (random_blocks(n, p, eps, k, rng),
                random_blocks(n, m, eps, k, rng))
    if kind == "pca":
        return (residual_pca_blocks(Phi, U, Xp, eps, n),
                random_blocks(n, m, eps, n, rng))
    raise ValueError(kind)


def one_config(seed, eps, template, x0_radius, N=30, alpha=2.0,
               template_w=0.01, rel_err=0.15, n_val=40, lam=0.90,
               n_random=None):
    rng = np.random.default_rng(seed)
    plant = third_order_system()
    n, m = plant.n, plant.m
    Gw_true = alpha * template_w * np.eye(n)
    Gw_template = template_w * np.eye(n)

    Phi, U, Xp, Xs = collect_data(plant, N, x0_radius * np.ones(n), rng,
                                  u_range=0.5, w_gen=Gw_true,
                                  reset_radius=5.0 * x0_radius)
    X_region = operating_region_from_data(Xs)
    safe = safe_set_box(n, 1.0)
    Mx = np.sqrt(n)
    r_data = float(np.max(np.abs(Xs)))
    Pv, Uv, Xpv, _ = sample_validation(plant, n_val,
                                       np.random.default_rng(seed + 8),
                                       radius=r_data, w_gen=Gw_true)

    GA, GB = build_template(template, n, n, m, eps, rng, Phi, U, Xp, n_random)
    # the prior is kept fixed and generous so that the template, not the prior,
    # is what varies across this sweep
    A_side = MatrixZonotope(plant.A, identity_blocks(n, n, rel_err + eps))
    B_side = MatrixZonotope(plant.B, identity_blocks(n, m, rel_err + eps))

    res = solve_icm(Phi=Phi, U=U, Xp=Xp, Gw=Gw_template, X_region=X_region,
                    GA_blocks=GA, GB_blocks=GB, A_side=A_side, B_side=B_side)
    row = {"seed": seed, "eps": eps, "template": template,
           "x0_radius": x0_radius, "s_A": len(GA), "N": N,
           "id_feasible": res.feasible, "sX": res.sX,
           "id_time": res.solve_time}
    if res.feasible:
        row["coverage"] = coverage(res, Pv, Uv, Xpv)
        row["coverage_train"] = coverage(res, Phi, U, Xp)
        row["encl"] = enclosure_size(res, Pv, Uv)
        row["mu_sum"] = float(np.sum(res.muA))
        model = to_uncertain_model(res)
        ctrl = lambda_contractive_lp(model, safe, lam, Mx)
        row["lp_feasible"] = ctrl.feasible
        row["lambda_min"] = min_feasible_lambda(model, safe, Mx)
    return row


def main(args):
    use_paper_style()
    rows = []

    # (a)+(b) magnitude x template
    for eps in args.eps_grid:
        for tpl in TEMPLATES:
            for t in range(args.trials):
                rows.append(one_config(seed=4000 + 23 * t, eps=float(eps),
                                       template=tpl, x0_radius=0.5, N=args.N))
        print(f"  eps={eps} done", flush=True)
    df_a = pd.DataFrame(rows)

    # (c) generator count (random templates of increasing richness)
    rows_c = []
    for k in args.gen_counts:
        for t in range(args.trials):
            rows_c.append(one_config(seed=5000 + 29 * t, eps=args.eps_ref,
                                     template="random", x0_radius=0.5,
                                     N=args.N, n_random=int(k)))
    df_c = pd.DataFrame(rows_c)

    # (d) initial-set radius
    rows_d = []
    for r0 in args.x0_radii:
        for t in range(args.trials):
            rows_d.append(one_config(seed=6000 + 31 * t, eps=args.eps_ref,
                                     template="entry", x0_radius=float(r0),
                                     N=args.N))
    df_d = pd.DataFrame(rows_d)

    out = Path(__file__).resolve().parent.parent / "results" / "tables"
    for nm, d in (("a_template", df_a), ("c_gencount", df_c),
                  ("d_initialset", df_d)):
        d.to_csv(out / f"exp7_raw_{nm}.csv", index=False)

    # ---- Fig: coverage heatmap over (eps, template) -----------------------
    piv = (df_a[df_a.id_feasible].pivot_table(index="template", columns="eps",
                                              values="coverage", aggfunc="mean")
           .reindex(TEMPLATES) * 100)
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([TEMPLATE_LABEL[t] for t in piv.index])
    ax.set_xlabel(r"generator magnitude $\varepsilon$")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=6, color="w" if v < 60 else "k")
    fig.colorbar(im, ax=ax, label="coverage (\\%)")
    ax.grid(False)
    save_fig(fig, "exp7_template_heatmap")

    # ---- Fig: sX / coverage / lambda* vs eps ------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(COL_W * 2.06, COL_H))
    for tpl in TEMPLATES:
        s = df_a[(df_a.template == tpl) & df_a.id_feasible]
        if s.empty:
            continue
        for ax, col in zip(axes, ("sX", "coverage", "lambda_min")):
            g = s.groupby("eps")[col]
            y = g.mean() * (100 if col == "coverage" else 1)
            ax.plot(y.index, y.values, marker="o", ms=2.6,
                    label=TEMPLATE_LABEL[tpl])
    for ax, lab in zip(axes, ("$s_\\mathcal{X}$", "coverage (\\%)",
                              "$\\lambda^\\star$")):
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(lab)
        ax.set_xscale("log")
    axes[0].legend(loc="best", fontsize=5.5)
    save_fig(fig, "exp7_sensitivity_eps")

    # ---- Fig: initial-set radius ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2.06, COL_H))
    s = df_d[df_d.id_feasible]
    for ax, col, lab in zip(axes, ("coverage", "encl"),
                            ("coverage (\\%)", "enclosure size")):
        g = s.groupby("x0_radius")[col]
        y = g.mean() * (100 if col == "coverage" else 1)
        e = g.std() * (100 if col == "coverage" else 1)
        ax.errorbar(y.index, y.values, yerr=e.values, capsize=2,
                    marker="o", ms=2.8, color="#1f77b4")
        ax.set_xlabel("initial-set radius")
        ax.set_ylabel(lab)
    save_fig(fig, "exp7_initial_set")

    # ---- Tables ------------------------------------------------------------
    t_a = (df_a.groupby(["template", "eps"])
           .agg(id_feas=("id_feasible", "mean"), sX=("sX", "mean"),
                cov=("coverage", "mean"), encl=("encl", "mean"),
                lam=("lambda_min", "mean"), lp=("lp_feasible", "mean"))
           .reset_index())
    for c in ("id_feas", "cov", "lp"):
        t_a[c] = 100 * t_a[c]
    t_a["template"] = [TEMPLATE_LABEL[t] for t in t_a["template"]]
    t_a = t_a.rename(columns={"template": "Template", "eps": "$\\varepsilon$",
                              "id_feas": "ID feas.", "sX": "$s_\\mathcal{X}$",
                              "cov": "Coverage", "encl": "Encl.",
                              "lam": "$\\lambda^\\star$", "lp": "LP feas."})
    save_table(t_a, "exp7_template_sensitivity")

    t_c = (df_c.groupby("s_A").agg(id_feas=("id_feasible", "mean"),
                                   sX=("sX", "mean"), cov=("coverage", "mean"),
                                   encl=("encl", "mean"),
                                   t=("id_time", "mean")).reset_index())
    for c in ("id_feas", "cov"):
        t_c[c] = 100 * t_c[c]
    t_c = t_c.rename(columns={"s_A": "$s_A$", "id_feas": "ID feas.",
                              "sX": "$s_\\mathcal{X}$", "cov": "Coverage",
                              "encl": "Encl.", "t": "$t_{\\rm id}$ (s)"})
    save_table(t_c, "exp7_generator_count",
               caption="Effect of the number of generators (random templates).")

    t_d = (df_d.groupby("x0_radius")
           .agg(id_feas=("id_feasible", "mean"), sX=("sX", "mean"),
                cov=("coverage", "mean"), cov_tr=("coverage_train", "mean"),
                encl=("encl", "mean"), lam=("lambda_min", "mean")).reset_index())
    for c in ("id_feas", "cov", "cov_tr"):
        t_d[c] = 100 * t_d[c]
    t_d = t_d.rename(columns={"x0_radius": "init.\\ radius",
                              "id_feas": "ID feas.", "sX": "$s_\\mathcal{X}$",
                              "cov": "Cov. val.", "cov_tr": "Cov. train",
                              "encl": "Encl.", "lam": "$\\lambda^\\star$"})
    save_table(t_d, "exp7_initial_set")
    print(t_a.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--eps-grid", type=float, nargs="+",
                    default=[0.02, 0.05, 0.10, 0.20, 0.40])
    ap.add_argument("--eps-ref", type=float, default=0.10)
    ap.add_argument("--gen-counts", type=int, nargs="+", default=[1, 3, 6, 9])
    ap.add_argument("--x0-radii", type=float, nargs="+",
                    default=[0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--N", type=int, default=30)
    main(ap.parse_args())
