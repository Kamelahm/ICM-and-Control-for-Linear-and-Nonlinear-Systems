"""
Control synthesis and certificates for input-affine nonlinear systems.

  * `nonlinearity_cancellation_sdp`  -- Theorem 2, Eq. (32)
  * `contraction_factor`             -- smallest certified kappa
  * `verify_contraction`             -- Monte-Carlo audit of the certificate
  * `roa_certificate`                -- Theorem 3 (delta*, r*, c*)

CORRESPONDENCE WITH THE MANUSCRIPT
----------------------------------
  Pbar        <->  S      (Lyapunov variable; D = Pbar^{-1})
  Ybar        <->  Y      (= Kbar Pbar; Kbar = Ybar Pbar^{-1} recovered after)
  Theta_*                 SUBTRACTED from the (2,2) blocks
  V_1, V_2, V_3           POSITIVE on the diagonal (Schur complement of +U V^-1 U^T)
  kappa                   design parameter in the (1,1) block, not read off
  Pbar >= I               scale normalization (w.l.o.g. by homogeneity)

"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cvxpy as cp

from .solver_utils import safe_solve, solved_ok
import numpy as np


# --------------------------------------------------------------------------- #
@dataclass
class NonlinearDesign:
    feasible: bool
    status: str
    K: Optional[np.ndarray] = None          # Kbar, linear part,    m x n
    Khat: Optional[np.ndarray] = None       # Khat, nonlinear part, m x (S-n)
    S: Optional[np.ndarray] = None          # Pbar
    D: Optional[np.ndarray] = None          # D = Pbar^{-1}
    rho: float = np.nan                     # ||Ahat + B Khat||_2 <= rho
    rho_sq: float = np.nan                  # rhobar = rho^2
    solve_time: float = np.nan
    meta: dict = field(default_factory=dict)

    def gain(self):
        """Full gain K_full = [Kbar  Khat] so that u = K_full Z(x)."""
        return np.hstack([self.K, self.Khat])


def intervals_from_icm(res, n: int):
    """
    Split the identified nonlinear sets into linear / nonlinear parts and return
    nominal matrices with elementwise radii, i.e. Eq. (28)-(30).
    """
    Alo, Ahi = res.A_set.interval()
    Blo, Bhi = res.B_set.interval()
    A0_full = 0.5 * (Alo + Ahi)
    dA_full = 0.5 * (Ahi - Alo)
    B0 = 0.5 * (Blo + Bhi)
    dB = 0.5 * (Bhi - Blo)
    return (A0_full[:, :n], dA_full[:, :n],
            A0_full[:, n:], dA_full[:, n:], B0, dB)


# --------------------------------------------------------------------------- #
#  Shared LMI assembly
# --------------------------------------------------------------------------- #
def _theta(slack, radii, n, E):
    """Theta = sum_ij slack_ij * radii_ij^2 * e_i e_i^T  (squared radii)."""
    q = radii.shape[1]
    return sum(slack[i * q + j] * float(radii[i, j]) ** 2 * np.outer(E[:, i], E[:, i])
               for i in range(n) for j in range(q))


def _lyapunov_block(kappa_expr, P, Y, A0, dA, B0, dB, lam, eps2):
    """
    Robust Lyapunov LMI in congruence form (corrected Eq. 32b):

        [ kappa*P    (A0 P + B0 Y)^T    U3    U4 ]
        [   *      P - Th_A - Th_B      0     0  ]  > 0
        [   *            *              V3    0  ]
        [   *            *              *     V2 ]
    """
    n, m = A0.shape[0], B0.shape[1]
    E, Z = np.eye(n), np.zeros
    Th_A = _theta(lam, dA, n, E)
    Th_B = _theta(eps2, dB, n, E)
    U3 = cp.hstack([P for _ in range(n)])            # n x n^2
    U4 = cp.hstack([Y.T for _ in range(n)])          # n x nm
    M = A0 @ P + B0 @ Y
    return cp.vstack([
        cp.hstack([kappa_expr * P, M.T, U3, U4]),
        cp.hstack([M, P - Th_A - Th_B, Z((n, n * n)), Z((n, n * m))]),
        cp.hstack([U3.T, Z((n * n, n)), cp.diag(lam), Z((n * n, n * m))]),
        cp.hstack([U4.T, Z((n * m, n)), Z((n * m, n * n)), cp.diag(eps2)]),
    ])


# --------------------------------------------------------------------------- #
def nonlinearity_cancellation_sdp(A0, dA, Ahat0, dAhat, B0, dB,
                                  solver="CLARABEL", eps_slack=1e-6,
                                  margin=1e-7, k_max=50.0, gain_weight=1e-3,
                                  kappa_target=0.90, cond_weight=1e-2,
                                  verbose=False) -> NonlinearDesign:
    """
    Theorem 2 / Eq. (32): joint design of the cancellation gain Khat and the
    stabilizing gain Kbar, robust to the identified interval uncertainty.

    A feasible point certifies  A_cl^T D A_cl <= kappa_target * D  for every
    admissible realization, with D = Pbar^{-1}.  kappa_target is a design
    parameter: with kappa = 1 the LMI certifies only non-strict decrease and
    delta*, r*, c* of Theorem 3 degenerate.

    Pbar >= I is imposed without loss of generality -- the LMI is homogeneous of
    degree one in (Pbar, Ybar, lam, eps) and Kbar = Ybar Pbar^{-1} is invariant
    under that scaling.
    """
    if not (0.0 < kappa_target < 1.0):
        raise ValueError(f"kappa_target must lie in (0,1), got {kappa_target}")

    n, m = A0.shape[0], B0.shape[1]
    Snl = Ahat0.shape[1]                     # S - n
    E, Z = np.eye(n), np.zeros
    t0 = time.perf_counter()

    rho = cp.Variable(nonneg=True)                     # rhobar = rho^2
    Khat = cp.Variable((m, Snl))
    P = cp.Variable((n, n), symmetric=True)            # Pbar
    Y = cp.Variable((m, n))                            # Ybar = Kbar Pbar
    beta = cp.Variable(n * Snl, nonneg=True)           # LMI-2 slacks (Ahat)
    eps1 = cp.Variable(n * m, nonneg=True)             # LMI-2 slacks (B), split
    lam = cp.Variable(n * n, nonneg=True)              # LMI-1 slacks (Abar)
    eps2 = cp.Variable(n * m, nonneg=True)             # LMI-1 slacks (B), split
    smax = cp.Variable(nonneg=True)                    # conditioning of Pbar

    cons = [beta >= eps_slack, eps1 >= eps_slack,
            lam >= eps_slack, eps2 >= eps_slack,
            P >> np.eye(n),                       # Pbar >= I, w.l.o.g.
            smax * np.eye(n) - P >> 0,            # Pbar <= smax I
            cp.norm(Y, 2) <= k_max,
            cp.norm(Khat, 2) <= k_max]

    LMI_lyap = _lyapunov_block(kappa_target, P, Y, A0, dA, B0, dB, lam, eps2)
    cons.append(LMI_lyap >> margin * np.eye(LMI_lyap.shape[0]))

    Th_Ahat = _theta(beta, dAhat, n, E)
    Th_B1 = _theta(eps1, dB, n, E)
    U1 = np.tile(np.eye(Snl), (1, n))                       # (S-n) x n(S-n)
    U2 = cp.hstack([Khat.T for _ in range(n)])              # (S-n) x nm
    M1 = Ahat0 + B0 @ Khat                                  # n x (S-n)
    LMI_canc = cp.vstack([
        cp.hstack([rho * np.eye(Snl), M1.T, U1, U2]),
        cp.hstack([M1, np.eye(n) - Th_Ahat - Th_B1,
                   Z((n, n * Snl)), Z((n, n * m))]),
        cp.hstack([U1.T, Z((n * Snl, n)), cp.diag(beta), Z((n * Snl, n * m))]),
        cp.hstack([U2.T, Z((n * m, n)), Z((n * m, n * Snl)), cp.diag(eps1)]),
    ])
    cons.append(LMI_canc >> margin * np.eye(LMI_canc.shape[0]))

    prob = cp.Problem(cp.Minimize(rho + gain_weight * cp.norm(Y, 'fro')
                                  + gain_weight * cp.norm(Khat, 'fro')
                                  + cond_weight * smax), cons)
    status, _ = safe_solve(prob, solver=solver, verbose=verbose)
    dt = time.perf_counter() - t0

    if status not in ("optimal", "optimal_inaccurate"):
        return NonlinearDesign(False, status, solve_time=dt)

    Pv = np.array(P.value)
    Pv = 0.5 * (Pv + Pv.T)
    Kv = np.array(Y.value) @ np.linalg.inv(Pv)          # Kbar = Ybar Pbar^{-1}
    Khv = np.array(Khat.value)
    # rho is NOT read off the solver: with the SCS fallback an
    # "optimal_inaccurate" point can report rho ~ 0 while the actual bound is
    # ~0.1.  certified_rho evaluates sup ||Ahat + B Khat||_2 over the intervals
    # for the returned Khat directly.
    rho_c = certified_rho(Khv, Ahat0, dAhat, B0, dB)
    des = NonlinearDesign(True, status, K=Kv, Khat=Khv,
                          S=Pv, D=np.linalg.inv(Pv),
                          rho_sq=float(rho_c ** 2), rho=float(rho_c),
                          solve_time=dt)
    des.meta["rho_sdp"] = float(np.sqrt(max(rho.value, 0.0)))
    des.meta["min_eig_lyap"] = float(np.linalg.eigvalsh(np.array(LMI_lyap.value))[0])
    des.meta["kappa_target"] = kappa_target
    des.meta["cond_S"] = float(np.linalg.cond(Pv))
    return des


def certified_rho(Khat, Ahat0, dAhat, B0, dB, max_vertices=2 ** 16):
    """
    Upper bound on sup ||Ahat + B Khat||_2 over the interval family, for a
    GIVEN Khat.  The spectral norm is convex in (Ahat, B), so the supremum over
    the box is attained at a vertex; it is computed exactly when the number of
    vertices is at most `max_vertices`, and otherwise bounded by
    ||Ahat0 + B0 Khat|| + || dAhat ||_2 + || dB |Khat| ||_2  (valid because
    the spectral norm is monotone on entrywise-nonnegative matrices).
    """
    Khat = np.asarray(Khat, float)
    nom = Ahat0 + B0 @ Khat
    tri = float(np.linalg.norm(nom, 2) + np.linalg.norm(dAhat, 2)
                + np.linalg.norm(dB @ np.abs(Khat), 2))
    ia = np.argwhere(dAhat > 0); ib = np.argwhere(dB > 0)
    nv = len(ia) + len(ib)
    if 2 ** nv > max_vertices:
        return tri
    import itertools
    best = 0.0
    for s in itertools.product((-1.0, 1.0), repeat=nv):
        Ah = Ahat0.copy(); B = B0.copy()
        for (i, j), si in zip(ia, s[:len(ia)]):
            Ah[i, j] += si * dAhat[i, j]
        for (i, j), si in zip(ib, s[len(ia):]):
            B[i, j] += si * dB[i, j]
        best = max(best, float(np.linalg.norm(Ah + B @ Khat, 2)))
    return min(best, tri)


# --------------------------------------------------------------------------- #
def _lyap_feasible(kap, design, A0, dA, B0, dB, solver="CLARABEL"):
    """Is the frozen-(Pbar, Ybar) Lyapunov LMI feasible at this kappa?"""
    n, m = A0.shape[0], B0.shape[1]
    P, Y = design.S, design.K @ design.S
    lam = cp.Variable(n * n, nonneg=True)
    eps2 = cp.Variable(n * m, nonneg=True)
    blk = _lyapunov_block(float(kap), P, Y, A0, dA, B0, dB, lam, eps2)
    prob = cp.Problem(cp.Minimize(0),
                      [blk >> 0, lam >= 1e-7, eps2 >= 1e-7])
    status, _ = safe_solve(prob, solver=solver)
    if not solved_ok(status) or lam.value is None or eps2.value is None:
        return False
    # A posteriori check.  When CLARABEL fails, the SCS fallback can report
    # "optimal"/"optimal_inaccurate" for an infeasible kappa, and the bisection
    # then accepted kappa values that verify_contraction rejects.  The check is
    # done on the Schur complement of the diagonal multiplier blocks, which is
    # equivalent to blk >= 0 for positive multipliers and, unlike blk itself,
    # does not inherit the scale 1/radius^2 of the multipliers (multipliers of
    # order 1e11 appear when the ICM radii collapse to ~1e-11).
    lv = np.asarray(lam.value, float); ev = np.asarray(eps2.value, float)
    if lv.min() <= 0 or ev.min() <= 0:
        return False
    E = np.eye(n)
    Th = _theta(lv, dA, n, E) + _theta(ev, dB, n, E)
    U3 = np.hstack([P] * n); U4 = np.hstack([Y.T] * n)
    TL = kap * P - U3 @ np.diag(1.0 / lv) @ U3.T - U4 @ np.diag(1.0 / ev) @ U4.T
    Mm = A0 @ P + B0 @ Y
    R = np.block([[TL, Mm.T], [Mm, P - Th]])
    R = 0.5 * (R + R.T)
    return bool(np.linalg.eigvalsh(R)[0] >= -1e-9 * max(1.0, np.abs(P).max()))


def contraction_factor(design: NonlinearDesign, A0, dA, B0, dB,
                       solver="CLARABEL", tol=1e-3, hi=1.0):
    """
    Smallest certified kappa with A_cl^T D A_cl <= kappa D for all admissible
    (A, B), by BISECTION on the fixed-kappa feasibility LMI with (Pbar, Ybar)
    frozen.  Minimizing kappa as a free variable stalls -- see module docstring.
    """
    if not design.feasible:
        return np.nan
    if not _lyap_feasible(hi, design, A0, dA, B0, dB, solver):
        return np.nan
    lo = 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _lyap_feasible(mid, design, A0, dA, B0, dB, solver):
            hi = mid
        else:
            lo = mid
    return float(hi)


def certified_kappa(design, A0, dA, B0, dB, solver="CLARABEL"):
    """
    Verified contraction factor for the reported design.  The bisection value
    is used, and the design target replaces it only when the target itself
    passes the verified LMI check.  (Previously the target was substituted
    whenever the bisection returned more than the target, which certified
    kappa = 0.3 for a design whose actual worst-case contraction was 0.87.)
    """
    kappa = contraction_factor(design, A0, dA, B0, dB, solver=solver)
    kt = design.meta.get("kappa_target")
    if (kt is not None and np.isfinite(kappa) and kappa > kt
            and _lyap_feasible(kt, design, A0, dA, B0, dB, solver)):
        kappa = float(kt)
    return kappa


def verify_contraction(design: NonlinearDesign, kappa, A0, dA, B0, dB,
                       rng=None, n_samples=2000, tol=1e-8):
    """
    Monte-Carlo audit of A_cl^T D A_cl <= kappa D over the identified intervals.
    Samples vertices as well as interior points, since the worst case of a
    multilinear expression over a box sits at a vertex.  Necessary check, not a
    proof -- the LMI is the proof.
    """
    rng = rng or np.random.default_rng(0)
    D, K = design.D, design.K
    worst = -np.inf
    for t in range(n_samples):
        if t % 2:                                    # vertex sample
            sA = rng.choice([-1.0, 1.0], size=A0.shape)
            sB = rng.choice([-1.0, 1.0], size=B0.shape)
        else:                                        # interior sample
            sA = rng.uniform(-1, 1, size=A0.shape)
            sB = rng.uniform(-1, 1, size=B0.shape)
        Acl = (A0 + dA * sA) + (B0 + dB * sB) @ K
        worst = max(worst, float(np.linalg.eigvalsh(Acl.T @ D @ Acl - kappa * D)[-1]))
    return {"max_eig": worst, "ok": bool(worst <= tol), "kappa": float(kappa)}


# --------------------------------------------------------------------------- #
def _g_profile(Q: Callable, r_max: float, n: int, rng, n_samples=6000,
               n_grid=200):
    """
    Precompute  r -> sup_{0<||x||<=r} ||Q(x)||/||x||  once.

    Sampling the ball afresh inside a bisection is slow and inconsistent (the
    sup estimate jitters between iterations, so the bisection can converge to
    the noise).  One sample set plus a running maximum over shells gives a
    monotone profile that can be inverted directly.
    """
    # LOG-spaced radii and grid.  r* is typically orders of magnitude below
    # r_cap, so a linear grid over [0, r_cap] resolves it with a handful of
    # points; r* then quantizes coarsely, c* goes flat across the delta sweep,
    # and the selected delta is decided by tie-breaking rather than by the
    # trade-off.  Log spacing puts the resolution where r* actually lives.
    lo_exp, hi_exp = np.log10(r_max) - 4.0, np.log10(r_max)
    X = rng.normal(size=(n_samples, n))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    radii = 10.0 ** rng.uniform(lo_exp, hi_exp, size=(n_samples, 1))
    X = X * radii
    nrm = np.linalg.norm(X, axis=1)
    g = np.array([np.linalg.norm(Q(x)) for x in X]) / np.maximum(nrm, 1e-12)

    grid = np.logspace(lo_exp, hi_exp, n_grid)
    idx = np.searchsorted(grid, nrm, side="left")
    sup = np.zeros(n_grid)
    for i, gi in zip(idx, g):
        if i < n_grid and gi > sup[i]:
            sup[i] = gi
    sup = np.maximum.accumulate(sup)          # monotone in r
    return grid, sup


def _invert_profile(grid, sup, rho, delta, r_cap):
    """Largest r <= r_cap with rho * sup_g(r) <= delta."""
    if rho <= 1e-12:
        return float(r_cap)
    ok = (rho * sup <= delta) & (grid <= r_cap + 1e-12)
    if not np.any(ok):
        return 0.0
    return float(grid[np.max(np.nonzero(ok)[0])])


def roa_certificate(design: NonlinearDesign, Q: Callable, n: int,
                    A0, dA, B0, dB, x_max=None, rng=None,
                    delta_frac=0.9, solver="CLARABEL", profile=None,
                    kappa=None):
    """
    Theorem 3: kappa, a, delta*, r*, c* and the decay rate c(delta).
    `x_max` optionally caps r* by the operating region X; `r_star_capped`
    flags when that cap is active, in which case r* reports the region
    boundary rather than the certificate.
    """
    rng = rng or np.random.default_rng(0)
    D = design.D
    evals = np.linalg.eigvalsh(D)
    lam_lo, lam_hi = float(evals[0]), float(evals[-1])
    if kappa is None:
        kappa = certified_kappa(design, A0, dA, B0, dB, solver=solver)
    if not np.isfinite(kappa) or kappa >= 1.0:
        return {"kappa": kappa, "feasible": False}

    # certified bound on a = sup ||D A_cl||_2 over the identified intervals.
    # |D (A0 + dA_pert)| <= |D| dA elementwise, and the spectral norm is
    # monotone on entrywise-nonnegative matrices -- much tighter than the
    # Frobenius/triangle bound.
    Acl0 = A0 + B0 @ design.K
    a = float(np.linalg.norm(D @ Acl0, 2)
              + np.linalg.norm(np.abs(D) @ dA, 2)
              + np.linalg.norm(np.abs(D) @ dB @ np.abs(design.K), 2))
    delta_star = (-a + np.sqrt(a ** 2 + (1 - kappa) * lam_lo * lam_hi)) / lam_hi
    delta = delta_frac * delta_star
    c_delta = kappa + (2 * a * delta + lam_hi * delta ** 2) / lam_lo

    rho = design.rho
    r_cap = float(x_max) if x_max is not None else float(np.pi)
    grid, sup = (profile if profile is not None
                 else _g_profile(Q, r_cap, n, rng))
    r_star = _invert_profile(grid, sup, rho, delta, r_cap)

    c_star = lam_lo * r_star ** 2
    return {"feasible": c_delta < 1 and r_star > 0,
            "kappa": kappa, "a": a, "lam_min": lam_lo, "lam_max": lam_hi,
            "delta_star": float(delta_star), "delta": float(delta),
            "delta_frac": float(delta_frac),
            "c_delta": float(c_delta), "rho": rho,
            "r_star": float(r_star), "c_star": float(c_star),
            "r_star_capped": bool(abs(r_star - r_cap) < 1e-9)}


def best_delta_certificate(design, Q, n, A0, dA, B0, dB, x_max=None, rng=None,
                           fracs=np.linspace(0.02, 0.98, 25), solver="CLARABEL",
                           c_max=0.99):
    """
    Choose delta in (0, delta*): the largest certified ROA that still achieves
    the decay budget c(delta) <= c_max.

    r* increases with delta while c(delta) worsens toward 1, so maximizing c*
    without a budget drives delta -> delta* and the certified convergence rate
    becomes arbitrarily slow.  Theorem 4's wbar_max used to supply the balance
    between the two; with the ISS results removed the trade-off has to be made
    explicit, and `c_max` is a reporting choice that belongs in the caption
    next to r* and c*.
    """
    rng = rng or np.random.default_rng(0)
    r_cap = float(x_max) if x_max is not None else float(np.pi)
    profile = _g_profile(Q, r_cap, n, rng)
    kappa = certified_kappa(design, A0, dA, B0, dB, solver=solver)
    best, best_c = None, -1.0
    for f in fracs:
        roa = roa_certificate(design, Q, n, A0, dA, B0, dB, x_max=x_max,
                              rng=rng, delta_frac=float(f), solver=solver,
                              profile=profile, kappa=kappa)
        if not roa.get("feasible", False) or roa["c_delta"] > c_max:
            continue
        if roa["c_star"] > best_c:
            best, best_c = roa, roa["c_star"]
    if best is not None:
        best["c_max"] = float(c_max)
    return best


def sweep_contraction_targets(A0, dA, Ahat0, dAhat, B0, dB, Q, n,
                              kappas=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
                              x_max=1.0, rng=None, audit=True, c_max=0.99,
                              **sdp_kwargs):
    """
    Sweep the certified contraction rate kappa and keep the design with the
    largest certified region of attraction c* subject to the decay budget.
    Returns (best_design, best_certificate, table_of_all_rows).
    """
    rng = rng or np.random.default_rng(0)
    rows, best, best_roa, best_c = [], None, None, -1.0
    for kap in kappas:
        des = nonlinearity_cancellation_sdp(A0, dA, Ahat0, dAhat, B0, dB,
                                            kappa_target=kap, **sdp_kwargs)
        row = {"kappa_target": kap, "sdp_feasible": des.feasible}
        if des.feasible:
            if audit:
                chk = verify_contraction(des, kap, A0, dA, B0, dB, rng=rng,
                                         n_samples=400)
                row["audit_max_eig"] = chk["max_eig"]
                row["audit_ok"] = chk["ok"]
            roa = best_delta_certificate(des, Q, n, A0, dA, B0, dB,
                                         x_max=x_max, rng=rng, c_max=c_max)
            if roa is not None:
                row.update({"rho": des.rho, "kappa": roa["kappa"],
                            "delta_star": roa["delta_star"],
                            "r_star": roa["r_star"], "c_star": roa["c_star"],
                            "c_delta": roa["c_delta"],
                            "r_star_capped": roa["r_star_capped"],
                            "K_norm": float(np.linalg.norm(des.K, 2))})
                if roa["c_star"] > best_c:
                    best, best_roa, best_c = des, roa, roa["c_star"]
        rows.append(row)
    return best, best_roa, rows
