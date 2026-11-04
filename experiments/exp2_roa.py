"""
Experiment 2 -- Certified region of attraction for the input-affine nonlinear case.

Outputs
-------
  exp2_roa_phase.pdf        Fig: Omega_{c*}, controlled and uncontrolled orbits
  exp2_roa.tex, exp2_raw.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from icm_control.assumptions import DEFAULT_MU_FLOOR, check_assumption
from icm_control.control_nonlinear import (best_delta_certificate,
                                           intervals_from_icm,
                                           nonlinearity_cancellation_sdp,
                                           verify_contraction)
from icm_control.identification import solve_icm
from icm_control.reporting import COL_H, COL_W, save_fig, save_table, use_paper_style
from icm_control.sets import Polytope
from icm_control.protocol import REG, U_MAX_PEN, X_RADIUS_PEN
from icm_control.systems import Pendulum, collect_independent, pendulum_side_info

import matplotlib.pyplot as plt


def Q_fn(x):
    """Q(x) = [sin x1 - x1, 1 - cos x1]^T, the nonlinear block of Z(x)."""
    x1 = float(np.atleast_1d(x)[0])
    return np.array([np.sin(x1) - x1, 1.0 - np.cos(x1)])


def identify(seed, N, w_mag, gamma, mu_floor, x0, rel_err):
    """Independent one-step experiments (Assumption 9); x0 is unused and kept
    only for backward compatibility of the call signature."""
    rng = np.random.default_rng(seed)
    sysd = Pendulum()
    Phi, U, Xp, _ = collect_independent(
        sysd, N, rng, x_radius=X_RADIUS_PEN, u_range=0.5,
        w_sampler=lambda r: r.uniform(-w_mag, w_mag, size=2),
        basis=sysd.basis_full)
    A_side, GA, B_side, GB = pendulum_side_info(sysd, "full", rel_err=rel_err)
    res = solve_icm(Phi=Phi, U=U, Xp=Xp, Gw=gamma * w_mag * np.eye(2),
                    X_region=Polytope(np.vstack([np.eye(2), -np.eye(2)]),
                                      5 * np.ones(4)),
                    GA_blocks=GA, GB_blocks=GB, A_side=A_side, B_side=B_side,
                    mu_floor=mu_floor, reg=REG)
    A_tr, B_tr = sysd.true_matrices("full")
    return sysd, res, (Phi, U, A_tr, B_tr)


def rollout(sysd, x0, K_full, horizon, rng, w_mag):
    """Closed loop on the TRUE plant; K_full=None gives the uncontrolled orbit."""
    x = np.array(x0, float)
    traj = [x.copy()]
    for _ in range(horizon):
        u = np.zeros(1) if K_full is None else K_full @ sysd.basis_full(x)
        w = rng.uniform(-w_mag, w_mag, size=2)
        x = sysd.step(x, u, w)
        traj.append(x.copy())
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 50:
            break
    return np.array(traj)


def ellipse(D, c, n=400):
    """Boundary of { x : x' D x = c } as a 2 x n array."""
    th = np.linspace(0, 2 * np.pi, n)
    circ = np.vstack([np.cos(th), np.sin(th)])
    L = np.linalg.cholesky(np.linalg.inv(D) * c)     # D^{-1} c = L L'
    return L @ circ


def input_profile(des, sysd, c_hi, n_levels=400, n_dir=241):
    """
    c -> sup_{x in Omega_c} |K_Z Z(x)| on a log grid of levels, as a running
    maximum over boundaries (Omega_c is the union of the smaller boundaries).
    Used for c*(u_max) of (92b) and for the input needed by the
    practical-stability condition.
    """
    K = des.gain()
    E = ellipse(des.D, 1.0, n=n_dir)
    cs = np.geomspace(1e-7, c_hi, n_levels)
    um = [max(float(np.abs(K @ sysd.basis_full(np.sqrt(c) * E[:, j])).max())
              for j in range(n_dir)) for c in cs]
    return cs, np.maximum.accumulate(um)


def c_at_input(cs, um, u_max, c_cap):
    """Largest grid level <= c_cap whose boundary-sup input is <= u_max."""
    ok = cs[(um <= u_max) & (cs <= c_cap)]
    return float(ok.max()) if ok.size else 0.0


def input_for_level(cs, um, c):
    """Input bound needed on Omega_c (conservative: next grid level up)."""
    i = int(np.searchsorted(cs, c))
    return float(um[i]) if i < len(um) else np.inf


def main(args):
    use_paper_style()
    rows = []
    best = None

    for t in range(args.trials):
        sysd, res, (Phi, U, A_tr, B_tr) = identify(
            seed=args.seed + t, N=args.N, w_mag=args.w_mag, gamma=args.gamma,
            mu_floor=args.mu_floor, x0=args.x0, rel_err=args.rel_err)
        row = {"trial": t, "id_feasible": bool(res.feasible)}
        if not res.feasible:
            rows.append(row); continue

        asm = check_assumption(res, A_tr, B_tr, Phi, U)
        row.update({"t_star": asm["t_star"], "assumption_holds": asm["holds"]})

        A0, dA, Ah0, dAh, B0, dB = intervals_from_icm(res, 2)
        des = nonlinearity_cancellation_sdp(A0, dA, Ah0, dAh, B0, dB,
                                            kappa_target=args.kappa)
        row["sdp_feasible"] = bool(des.feasible)
        if not des.feasible:
            rows.append(row); continue

        chk = verify_contraction(des, args.kappa, A0, dA, B0, dB,
                                 rng=np.random.default_rng(t), n_samples=800)
        roa = best_delta_certificate(des, Q_fn, 2, A0, dA, B0, dB,
                                     x_max=args.x_max,
                                     rng=np.random.default_rng(t),
                                     c_max=args.c_max)
        row.update({"rho": des.rho, "rho_sdp": des.meta.get("rho_sdp"),
                    "audit_ok": chk["ok"], "audit_max_eig": chk["max_eig"]})
        if roa is not None:
            row.update({"kappa": roa["kappa"], "r_star": roa["r_star"],
                        "c_star": roa["c_star"], "c_delta": roa["c_delta"],
                        "r_star_capped": roa["r_star_capped"]})
            # input-feasible levels (92b) and the practical-stability condition
            cs, um = input_profile(des, sysd, c_hi=1.05 * roa["c_star"])
            lam_min, lam_max = np.linalg.eigvalsh(des.D)[[0, -1]]
            row["u_on_cstar"] = input_for_level(cs, um, roa["c_star"])
            for u in (0.5, 1.0, 2.0, 5.0, 10.0):
                c = c_at_input(cs, um, u, roa["c_star"])
                row[f"c_u{u:g}"] = c
                # largest angle |x^1| on Omega_c (Table: max |x^1| in degrees)
                row[f"xdeg_u{u:g}"] = float(np.degrees(np.sqrt(c * np.linalg.inv(des.D)[0, 0])))
            wbar = float(np.linalg.norm(res.cw) + sum(
                res.muw[i] * np.linalg.norm(res.Gw[:, i]) for i in range(len(res.muw))))
            gap = 1.0 - np.sqrt(roa["c_delta"])
            c_req = float(lam_max * wbar ** 2 / gap ** 2)
            row.update({"wbar": wbar, "lam_max": float(lam_max),
                        "lam_min": float(lam_min),
                        "rhs_on_cstar": float(gap * np.sqrt(roa["c_star"])),
                        "c_req": c_req,
                        "u_req": (input_for_level(cs, um, c_req)
                                  if c_req <= roa["c_star"] else np.inf)})
            c_um = c_at_input(cs, um, args.u_max, roa["c_star"])
            row["c_umax"] = c_um
            if best is None or roa["c_star"] > best[1]["c_star"]:
                best = (sysd, roa, des, c_um)
        rows.append(row)
        print(f"  trial {t} done", flush=True)

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "results" / "tables"
    df.to_csv(out / "exp2_raw.csv", index=False)

    if best is None:
        print("  no feasible certificate; nothing to plot")
        return
    sysd, roa, des, c_um = best
    D = des.D
    # The figure shows the INPUT-FEASIBLE certified set Omega_{c*(u_max)} of
    # (92b); Omega_{c*_dyn} itself needs |u| up to ~20 and is not drawn.
    c_star = c_um
    K_full = des.gain()

    # ---- Figure: Omega_{c*(u_max)} with controlled / uncontrolled orbits ----
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    E = ellipse(D, c_star)
    ax.fill(E[0], E[1], color="tab:red", alpha=0.12, zorder=0)
    ax.plot(E[0], E[1], color="tab:red", lw=1.2, zorder=1,
            label=r"$\Omega_{c^\star(u_{\max})}$ (Thm. 4)")


    n_ic = args.n_ic
    starts = ellipse(D, c_star, n=n_ic + 1)[:, :n_ic]
    left, decay_ok, decay_worst, u_seen = 0, 0, 0.0, 0.0
    c_delta = roa["c_delta"]
    for j in range(n_ic):
        x0 = starts[:, j]
        tr = rollout(sysd, x0, K_full, args.horizon,
                     np.random.default_rng(1000 + j), args.w_test)
        un = rollout(sysd, x0, None, args.horizon,
                     np.random.default_rng(1000 + j), args.w_test)
        # Theorem 3, Part 3:  V(x_k) <= c(delta)^k V(x_0).
        V = np.einsum("ij,jk,ik->i", tr, D, tr)
        bound = V[0] * c_delta ** np.arange(len(V))
        ratio = float(np.max(V / np.maximum(bound, 1e-300)))
        decay_worst = max(decay_worst, ratio)
        decay_ok += int(ratio <= 1.0 + 1e-6)
        ax.plot(un[:, 0], un[:, 1], color="0.55", lw=0.7, alpha=0.8, zorder=2,
                label="uncontrolled" if j == 0 else None)
        ax.plot(tr[:, 0], tr[:, 1], color="tab:blue", lw=0.9, zorder=3,
                label="closed loop (Thm. 3)" if j == 0 else None)
        ax.plot(x0[0], x0[1], "o", ms=2.5, color="tab:blue", zorder=4)
        if np.max(np.einsum("ij,jk,ik->i", tr, D, tr)) > c_star * (1 + 1e-6):
            left += 1
        u_seen = max(u_seen, float(np.max(np.abs(
            [K_full @ sysd.basis_full(x) for x in tr]))))

    ax.plot(0, 0, "k+", ms=6, zorder=5)
    lim = 1.6 * np.max(np.abs(E))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$x^1_k$ (rad)")
    ax.set_ylabel(r"$x^2_k$ (rad/s)")
    ax.legend(loc="upper right", fontsize=6)
    save_fig(fig, "exp2_roa_phase")
    print(f"  [check] Thm 4, invariance of Omega_c*: {n_ic - left}/{n_ic} "
          f"trajectories remain (predicts {n_ic}/{n_ic})", flush=True)
    print(f"  [check] Thm 4, decay V(x_k) <= c(delta)^k V(x_0): "
          f"{decay_ok}/{n_ic} satisfy it; worst ratio "
          f"max_k V(x_k)/(c^k V(x_0)) = {decay_worst:.3f} (predicts <= 1)",
          flush=True)
    print(f"  [check] input along the plotted closed-loop trajectories: "
          f"max |u| = {u_seen:.3f} (u_max = {args.u_max:g})", flush=True)
    print(f"  [note] test rollout disturbance w_test = {args.w_test:g}; "
          f"Theorem 4 covers the disturbance-free loop, so a nonzero value "
          f"here is an illustration, not a test.", flush=True)

    # ---- Table -------------------------------------------------------------
    ok = df[df.get("c_star").notna()] if "c_star" in df else df.iloc[0:0]
    tab = pd.DataFrame([{
        "ID feas.": 100 * df["id_feasible"].mean(),
        "SDP feas.": 100 * df.get("sdp_feasible", pd.Series(dtype=float)).mean(),
        "Assum. holds": 100 * df.get("assumption_holds", pd.Series(dtype=float)).mean(),
        "$t^\\star$ (med.)": df.get("t_star", pd.Series(dtype=float)).median(),
        "audit ok": 100 * df.get("audit_ok", pd.Series(dtype=float)).mean(),
        "$\\rho^\\star$": ok["rho"].mean() if not ok.empty else np.nan,
        "$\\kappa$": args.kappa,
        "$r^\\star$": ok["r_star"].mean() if not ok.empty else np.nan,
        "$c^\\star$": ok["c_star"].mean() if not ok.empty else np.nan,
        "$c(\\delta)$": ok["c_delta"].mean() if not ok.empty else np.nan,
        "$r^\\star$ capped": (100 * ok["r_star_capped"].mean()
                            if not ok.empty else np.nan),
        "$c^\\star(u_{\\max})$": ok["c_umax"].median() if not ok.empty else np.nan,
        "invariance": 100.0 * (n_ic - left) / n_ic,
        "decay": 100.0 * decay_ok / n_ic,
    }])
    save_table(tab, "exp2_roa")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--w-mag", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=27.26,
                    help="disturbance template scale; calibrated for the "
                         "pendulum by scripts/calibrate_assumptions.py")
    ap.add_argument("--mu-floor", type=float, default=DEFAULT_MU_FLOOR)
    ap.add_argument("--rel-err", type=float, default=0.20)
    ap.add_argument("--kappa", type=float, default=0.90)
    ap.add_argument("--u-max", type=float, default=U_MAX_PEN,
                    help="input bound for (92b) and Fig. 2")
    ap.add_argument("--c-max", type=float, default=0.99)
    ap.add_argument("--x-max", type=float, default=np.pi,
                    help="cap on r* from the operating region; a capped "
                         "r* reports the region boundary, not the certificate")
    ap.add_argument("--x0", type=float, nargs=2, default=[0.05, 0.0])
    ap.add_argument("--n-ic", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--w-test", type=float, default=0.0,
                    help="disturbance in the TEST rollout. Theorem 3 covers the "
                         "disturbance-free loop, so this defaults to 0; a "
                         "nonzero value is an illustration, not a test.")
    main(ap.parse_args())
