"""
Experiment 3 -- Validation of the input-affine nonlinear results.

Section 4 of the paper carries half the theory (Lemma 7, Lemma 9, Corollary 2,
Theorems 2-3) and currently rests on a single phase-plane figure.  Nothing in
the simulations measures the two quantities the section actually claims:

  * Lemma 9 is a *coverage* statement about the identified nonlinear ICM on a
    fresh transition, and no coverage number is reported anywhere for the
    pendulum;
  * Corollary 2 says the axis-aligned interval relaxation of Lemma 1 -- which
    is what the SDP of Theorem 2 consumes -- inherits that coverage.  The
    relaxation is therefore free in coverage and costly in tightness, and the
    size of that cost is unmeasured.

This script fills both gaps and adds the two comparisons a reviewer will ask
for: a nonlinear ablation (does side information help on the pendulum, as it
demonstrably does on the linear plant?), and a point-model arm standing in for
the approximate-nonlinearity-cancellation design of [43], whose pendulum
benchmark this is.

Outputs
-------
  exp3_nonlinear_coverage.pdf   coverage and enclosure vs extrapolation radius
  exp3_roa_by_arm.pdf           certified ROA per arm + true-plant behaviour
  exp3_nonlinear.tex, exp3_interval_cost.tex, exp3_raw.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from icm_control.assumptions import DEFAULT_MU_FLOOR, check_assumption
from icm_control.baselines import least_squares
from icm_control.control_nonlinear import (best_delta_certificate,
                                           intervals_from_icm,
                                           nonlinearity_cancellation_sdp,
                                           verify_contraction)
from icm_control.evaluation import (coverage, coverage_interval,
                                    enclosure_size, enclosure_size_interval,
                                    sample_validation)
from icm_control.identification import (solve_dcm, solve_icm,
                                        solve_icm_no_side_info)
from icm_control.reporting import (COL_H, COL_W, label, save_fig, save_table,
                                   style, use_paper_style)
from icm_control.sets import Polytope, random_blocks
from icm_control.systems import Pendulum, pendulum_side_info

import matplotlib.pyplot as plt

ARMS = [("DCM", solve_dcm), ("ICM-noSI", solve_icm_no_side_info),
        ("ICM", solve_icm)]
ORDER = ["LS", "DCM", "ICM-noSI", "ICM"]

# Extra arm used only for the Corollary 2 table.  With the entrywise template of
# `pendulum_side_info` every generator is a single-entry matrix, so the
# identified matrix zonotope IS an interval matrix and Lemma 1 is exact -- the
# inflation ratio is then 1.000 by construction and measures nothing.  The cost
# of the relaxation is only visible for generator directions that are not
# axis-aligned, which is what this arm supplies.
ROT = "ICM-rot"


def Q_fn(x):
    """Q(x) = [sin x1 - x1, 1 - cos x1]^T, the nonlinear block of Z(x)."""
    x1 = float(np.atleast_1d(x)[0])
    return np.array([np.sin(x1) - x1, 1.0 - np.cos(x1)])


def collect(sysd, N, x0, rng, w_mag, u_range=0.5):
    x = np.array(x0, float)
    Phi, U, Xp, Xs = [], [], [], []
    for _ in range(N):
        u = rng.uniform(-u_range, u_range, size=1)
        w = rng.uniform(-w_mag, w_mag, size=2)
        xn = sysd.step(x, u, w)
        Phi.append(sysd.basis_full(x)); U.append(u); Xp.append(xn); Xs.append(x)
        x = xn
    return (np.array(Phi).T, np.array(U).T, np.array(Xp).T, np.array(Xs).T)


def ls_intervals(res_ls, n):
    """Point model as a degenerate interval matrix: the [43]-style design."""
    A_full, B = res_ls.CA, res_ls.CB
    return (A_full[:, :n], np.zeros((n, n)),
            A_full[:, n:], np.zeros((n, A_full.shape[1] - n)),
            B, np.zeros_like(B))


def roa_area(D, c_star):
    """Area of {x : x' D x <= c*} in the plane."""
    if D is None or not np.isfinite(c_star):
        return np.nan
    return float(np.pi * c_star / np.sqrt(max(np.linalg.det(D), 1e-300)))


def true_plant_check(sysd, D, c_star, c_delta, K_full, n_ic, horizon, w_test):
    """
    Roll the TRUE plant from the boundary of the certified sublevel set.
    Returns (invariance rate, decay rate, worst decay ratio).
    """
    if D is None or not np.isfinite(c_star):
        return np.nan, np.nan, np.nan
    th = np.linspace(0, 2 * np.pi, n_ic, endpoint=False)
    L = np.linalg.cholesky(np.linalg.inv(D) * c_star)
    starts = L @ np.vstack([np.cos(th), np.sin(th)])
    inside, decay_ok, worst = 0, 0, 0.0
    for j in range(n_ic):
        rng = np.random.default_rng(4000 + j)
        x = starts[:, j].copy()
        traj = [x.copy()]
        for _ in range(horizon):
            u = K_full @ sysd.basis_full(x)
            w = rng.uniform(-w_test, w_test, size=2) if w_test > 0 else np.zeros(2)
            x = sysd.step(x, u, w)
            traj.append(x.copy())
            if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 50:
                break
        traj = np.array(traj)
        V = np.einsum("ij,jk,ik->i", traj, D, traj)
        inside += int(np.max(V) <= c_star * (1 + 1e-6))
        bound = V[0] * c_delta ** np.arange(len(V))
        ratio = float(np.max(V / np.maximum(bound, 1e-300)))
        worst = max(worst, ratio)
        decay_ok += int(ratio <= 1.0 + 1e-6)
    return inside / n_ic, decay_ok / n_ic, worst


def one_trial(seed, args):
    sysd = Pendulum()
    n = 2
    rng = np.random.default_rng(seed)
    Phi, U, Xp, Xs = collect(sysd, args.N, args.x0, rng, args.w_mag)
    A_tr, B_tr = sysd.true_matrices("full")

    A_side, GA, B_side, GB = pendulum_side_info(sysd, "full", rel_err=args.rel_err)
    X_region = Polytope(np.vstack([np.eye(2), -np.eye(2)]), 5 * np.ones(4))
    Gw = args.gamma * args.w_mag * np.eye(2)

    # Held-out transitions at increasing radius.  The regressor is Z(x), so the
    # validation sampler is given the basis map; states are drawn in the
    # physical space and lifted, matching Assumption 9.
    r_data = float(np.max(np.abs(Xs)))
    val = {f: sample_validation(sysd, args.n_val,
                                np.random.default_rng(seed + 90_000),
                                radius=f * r_data, w_gen=args.w_mag * np.eye(2),
                                basis=sysd.basis_full, state_dim=n)
           for f in args.extrapolation}

    rows = []
    models = {}
    for name, solver in ARMS:
        res = solver(Phi=Phi, U=U, Xp=Xp, Gw=Gw, X_region=X_region,
                     GA_blocks=GA, GB_blocks=GB, A_side=A_side, B_side=B_side,
                     mu_floor=args.mu_floor)
        row = {"seed": seed, "method": name, "id_feasible": bool(res.feasible)}
        if res.feasible:
            asm = check_assumption(res, A_tr, B_tr, Phi, U)
            row.update({"t_star": asm["t_star"], "assumption_holds": asm["holds"],
                        "sX": res.sX, "id_time": res.solve_time})
            for f, (Pv, Uv, Xpv, _) in val.items():
                row[f"cov_x{f:g}"] = coverage(res, Pv, Uv, Xpv)
                row[f"covI_x{f:g}"] = coverage_interval(res, Pv, Uv, Xpv)
                row[f"encl_x{f:g}"] = enclosure_size(res, Pv, Uv)
                row[f"enclI_x{f:g}"] = enclosure_size_interval(res, Pv, Uv)
            models[name] = intervals_from_icm(res, n)
        rows.append(row)

    # Rotated-template arm (Corollary 2 table only).  Directions are no longer
    # aligned with the coordinate axes, and the disturbance template is tight so
    # that the minimum-volume objective is forced to put mass on the dynamics.
    # Both are needed for the ratio to measure anything: with a generous
    # template the objective drives mu_A to zero, the enclosure reduces to the
    # (axis-aligned) additive term, and Lemma 1 is exactly free.  Side
    # information is dropped because the entrywise prior sets cannot contain
    # generators that put mass on the structural zeros.
    rot_rng = np.random.default_rng(seed + 31)
    GA_rot = random_blocks(n, Phi.shape[0], args.rot_eps, len(GA), rot_rng)
    GB_rot = random_blocks(n, 1, args.rot_eps, max(len(GB), 1), rot_rng)
    res_rot = solve_icm_no_side_info(
        Phi=Phi, U=U, Xp=Xp, Gw=args.rot_gamma * args.w_mag * np.eye(2),
        X_region=X_region, GA_blocks=GA_rot, GB_blocks=GB_rot,
        mu_floor=args.mu_floor)
    rrow = {"seed": seed, "method": ROT, "id_feasible": bool(res_rot.feasible)}
    if res_rot.feasible:
        rrow["muA_sum"] = float(np.sum(res_rot.muA))
        for f, (Pv, Uv, Xpv, _) in val.items():
            rrow[f"cov_x{f:g}"] = coverage(res_rot, Pv, Uv, Xpv)
            rrow[f"covI_x{f:g}"] = coverage_interval(res_rot, Pv, Uv, Xpv)
            rrow[f"encl_x{f:g}"] = enclosure_size(res_rot, Pv, Uv)
            rrow[f"enclI_x{f:g}"] = enclosure_size_interval(res_rot, Pv, Uv)
    rows.append(rrow)

    # Point-model arm: least squares in the same basis, zero parametric radii.
    # This is the uncertainty description that an approximate-nonlinearity-
    # cancellation design in the style of [43] hands to the same SDP.
    res_ls = least_squares(Phi, U, Xp)
    models["LS"] = ls_intervals(res_ls, n)
    rows.append({"seed": seed, "method": "LS", "id_feasible": True,
                 "id_time": res_ls.solve_time,
                 "center_err": float(np.max(np.abs(res_ls.CA - A_tr)))})

    # -- synthesis, certificate and true-plant behaviour, arm by arm ---------
    for row in rows:
        name = row["method"]
        if name not in models:
            continue
        A0, dA, Ah0, dAh, B0, dB = models[name]
        des = nonlinearity_cancellation_sdp(A0, dA, Ah0, dAh, B0, dB,
                                            kappa_target=args.kappa)
        row["sdp_feasible"] = bool(des.feasible)
        if not des.feasible:
            continue
        chk = verify_contraction(des, args.kappa, A0, dA, B0, dB,
                                 rng=np.random.default_rng(seed), n_samples=400)
        row.update({"rho": des.rho, "audit_ok": chk["ok"],
                    "audit_max_eig": chk["max_eig"]})
        roa = best_delta_certificate(des, Q_fn, n, A0, dA, B0, dB,
                                     x_max=args.x_max,
                                     rng=np.random.default_rng(seed),
                                     c_max=args.c_max)
        if roa is None:
            continue
        row.update({"r_star": roa["r_star"], "c_star": roa["c_star"],
                    "c_delta": roa["c_delta"],
                    "r_star_capped": bool(roa["r_star_capped"]),
                    "roa_area": roa_area(des.D, roa["c_star"])})
        inv, dec, worst = true_plant_check(sysd, des.D, roa["c_star"],
                                           roa["c_delta"], des.gain(),
                                           args.n_ic, args.horizon, args.w_test)
        row.update({"true_invariance": inv, "true_decay": dec,
                    "true_worst_ratio": worst})
    return rows


def main(args):
    use_paper_style()
    rows = []
    for t in range(args.trials):
        rows += one_trial(args.seed + t, args)
        print(f"  trial {t} done", flush=True)

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "results" / "tables"
    df.to_csv(out / "exp3_raw.csv", index=False)

    ok = df[df.id_feasible]
    cov_cols = [f"cov_x{f:g}" for f in args.extrapolation]
    covI_cols = [f"covI_x{f:g}" for f in args.extrapolation]
    encl_cols = [f"encl_x{f:g}" for f in args.extrapolation]
    enclI_cols = [f"enclI_x{f:g}" for f in args.extrapolation]

    # -------- Fig: coverage (left) and enclosure (right) vs extrapolation ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_W * 2.06, COL_H))
    for name in ORDER:
        sub = ok[ok.method == name]
        if sub.empty or cov_cols[0] not in sub or sub[cov_cols[0]].isna().all():
            continue
        ax1.errorbar(args.extrapolation,
                     [sub[c].mean() * 100 for c in cov_cols],
                     yerr=[sub[c].sem() * 100 for c in cov_cols],
                     capsize=2, label=label(name), **style(name))
        ax2.errorbar(args.extrapolation, [sub[c].mean() for c in encl_cols],
                     yerr=[sub[c].sem() for c in encl_cols],
                     capsize=2, label=label(name), **style(name))
        st = dict(style(name)); st["ls"] = ":"; st["marker"] = "None"
        ax2.plot(args.extrapolation, [sub[c].mean() for c in enclI_cols], **st)
    ax1.set_xlabel(r"$r\,/\,r_{\mathrm{id}}$")
    ax1.set_ylabel(r"Coverage of $\widehat{\mathcal{F}}_{\theta}(x,u)$ (\%)")
    ax1.set_ylim(-3, 103)
    ax1.legend(loc="lower left")
    ax2.set_xlabel(r"$r\,/\,r_{\mathrm{id}}$")
    ax2.set_ylabel(r"$\|\mathcal{E}\|$  (solid: zonotopic, dotted: interval)")
    save_fig(fig, "exp3_nonlinear_coverage")

    # -------- Table: nonlinear ablation ------------------------------------
    tab = []
    for name in ORDER:
        s = ok[ok.method == name]
        if s.empty:
            continue
        tab.append({
            "Method": label(name),
            "ID feas.": 100 * df[df.method == name]["id_feasible"].mean(),
            "Cov. $1\\times$": 100 * s[cov_cols[0]].mean(),
            "Cov. $2\\times$": 100 * s[cov_cols[1]].mean(),
            "Cov. $3\\times$": 100 * s[cov_cols[-1]].mean(),
            "Encl. size": s[encl_cols[0]].mean(),
            "SDP feas.": 100 * s.get("sdp_feasible", pd.Series(dtype=float)).mean(),
            "$\\hat\\rho^\\star$": s.get("rho", pd.Series(dtype=float)).mean(),
            "$r^\\star$": s.get("r_star", pd.Series(dtype=float)).mean(),
            "$r^\\star$ capped": 100 * s.get("r_star_capped", pd.Series(dtype=float)).mean(),
            "$c^\\star$": s.get("c_star", pd.Series(dtype=float)).mean(),
            "ROA area": s.get("roa_area", pd.Series(dtype=float)).mean(),
            "True inv.": 100 * s.get("true_invariance", pd.Series(dtype=float)).mean(),
            "True decay": 100 * s.get("true_decay", pd.Series(dtype=float)).mean(),
        })
    save_table(pd.DataFrame(tab), "exp3_nonlinear",
               caption="Nonlinear benchmark (inverted pendulum), ablation over "
                       "the same three identification arms as the linear case "
                       "plus a point-model arm standing in for the "
                       "nonlinearity-cancellation design of [43]. Coverage is "
                       "the held-out rate of Lemma~9 at increasing "
                       "extrapolation radius; `True inv.'\\ and `True decay'\\ "
                       "roll the true plant from the boundary of "
                       "$\\Omega_{c^\\star}$ and report the fraction of "
                       "trajectories satisfying the conclusions of Theorem~3. "
                       "The point model buys ROA area by asserting zero "
                       "parametric uncertainty, which the true-plant columns "
                       "then have to pay for.")

    # -------- Table: what the interval relaxation costs (Corollary 2) -------
    icost = []
    for name in ORDER + [ROT]:
        s = ok[ok.method == name]
        if s.empty or s[cov_cols[0]].isna().all():
            continue
        r = {"Method": label(name)}
        for f, cc, ci, ec, ei in zip(args.extrapolation, cov_cols, covI_cols,
                                     encl_cols, enclI_cols):
            r[f"Cov. ${f:g}\\times$ (zon./int.)"] = \
                f"{100 * s[cc].mean():.1f} / {100 * s[ci].mean():.1f}"
            r[f"$\\|\\mathcal{{E}}\\|$ ratio ${f:g}\\times$"] = \
                s[ei].mean() / max(s[ec].mean(), 1e-12)
        icost.append(r)
    save_table(pd.DataFrame(icost), "exp3_interval_cost",
               caption="Price of the axis-aligned interval over-approximation "
                       "of Lemma~1, which is the uncertainty description the "
                       "SDP of Theorem~2 consumes. Coverage is never reduced, "
                       "as Corollary~2 requires; the ratio column is the "
                       "resulting inflation of the certified enclosure. For the "
                       "entrywise template of Section~5.1 every generator is a "
                       "single-entry matrix, so the identified matrix zonotope "
                       "is already an interval matrix and the relaxation is "
                       "exact. The last row uses non-axis-aligned generator "
                       "directions and a tight disturbance template, the only "
                       "regime in which Lemma~1 loses anything.")

    # -------- Fig: certified ROA per arm -----------------------------------
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    th = np.linspace(0, 2 * np.pi, 300)
    circ = np.vstack([np.cos(th), np.sin(th)])
    drew = False
    for name in ORDER:
        s = ok[(ok.method == name) & ok.get("c_star", pd.Series(dtype=float)).notna()]
        if s.empty:
            continue
        area = s["roa_area"].mean()
        if not np.isfinite(area):
            continue
        # a circle of equal area is a fair one-number summary across arms
        rad = np.sqrt(area / np.pi)
        E = rad * circ
        ax.plot(E[0], E[1], label=f"{label(name)} ({area:.3f})", **style(name))
        drew = True
    if drew:
        ax.set_xlabel(r"$x^1$ (rad)")
        ax.set_ylabel(r"$x^2$ (rad/s)")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=6)
        save_fig(fig, "exp3_roa_by_arm")
    else:
        plt.close(fig)

    print(pd.DataFrame(tab).to_string(index=False))
    if icost:
        print(pd.DataFrame(icost).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--w-mag", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=27.26,
                    help="disturbance template scale, calibrated by "
                         "scripts/calibrate_assumptions.py")
    ap.add_argument("--mu-floor", type=float, default=DEFAULT_MU_FLOOR)
    ap.add_argument("--rel-err", type=float, default=0.20)
    ap.add_argument("--rot-eps", type=float, default=0.05,
                    help="generator magnitude of the non-axis-aligned template "
                         "used only to measure the cost of Lemma 1")
    ap.add_argument("--rot-gamma", type=float, default=1.0,
                    help="disturbance template scale for the rotated arm; must "
                         "be tight enough that mu_A > 0, otherwise the interval "
                         "relaxation is trivially exact")
    ap.add_argument("--kappa", type=float, default=0.90)
    ap.add_argument("--c-max", type=float, default=0.99)
    ap.add_argument("--x-max", type=float, default=np.pi)
    ap.add_argument("--x0", type=float, nargs=2, default=[0.05, 0.0])
    ap.add_argument("--extrapolation", type=float, nargs="+",
                    default=[1.0, 2.0, 3.0])
    ap.add_argument("--n-ic", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--w-test", type=float, default=0.0,
                    help="disturbance in the TEST rollout; Theorem 3 covers "
                         "the disturbance-free loop, so this defaults to 0")
    main(ap.parse_args())
