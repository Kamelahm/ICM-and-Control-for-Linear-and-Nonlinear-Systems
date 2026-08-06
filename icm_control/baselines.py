"""
Stronger baselines requested by the reviewers (R4 minor 2, R5.7).

1. `least_squares`      -- point estimate, no uncertainty quantification
                           (reference for the complexity discussion, R1.1)
2. `set_membership`     -- classical SMI [13], [14]: assumes the disturbance
                           bound is KNOWN and returns the feasible parameter
                           set, outer-bounded by an interval matrix
3. `tube_baseline`      -- data-driven tube / zonotopic-predictive-control style
                           [24]: a point nominal model plus a single additive
                           zonotopic disturbance set covering the residuals

All three return objects exposing the same `next_state_set(phi, u)` /
`covers(...)` interface as `IdentificationResult`, so the evaluation code in the
experiments is method-agnostic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cvxpy as cp

from .solver_utils import safe_solve, solved_ok
import numpy as np

from .sets import MatrixZonotope, Zonotope


# --------------------------------------------------------------------------- #
@dataclass
class BaselineResult:
    method: str
    feasible: bool
    status: str = "optimal"
    CA: Optional[np.ndarray] = None
    CB: Optional[np.ndarray] = None
    cw: Optional[np.ndarray] = None
    W_set: Optional[Zonotope] = None
    A_interval: Optional[tuple] = None
    B_interval: Optional[tuple] = None
    solve_time: float = np.nan
    sX: float = np.nan
    meta: dict = field(default_factory=dict)

    def predict(self, phi, u):
        return self.CA @ np.asarray(phi, float).reshape(-1) \
            + self.CB @ np.asarray(u, float).reshape(-1) + self.cw

    def next_state_set(self, phi, u):
        """One-step set. For SMI the parametric interval also contributes."""
        c = self.predict(phi, u)
        G = [self.W_set.G] if self.W_set is not None else []
        if self.A_interval is not None:
            phi = np.asarray(phi, float).reshape(-1)
            u = np.asarray(u, float).reshape(-1)
            radA = 0.5 * (self.A_interval[1] - self.A_interval[0])
            radB = 0.5 * (self.B_interval[1] - self.B_interval[0])
            r = radA @ np.abs(phi) + radB @ np.abs(u)
            G.append(np.diag(r))
        G = np.hstack(G) if G else np.zeros((c.size, 1))
        return Zonotope(c, G)

    def covers(self, phi, u, x_next, tol=1e-6):
        Z = self.next_state_set(phi, u)
        lo, hi = Z.interval()
        x_next = np.asarray(x_next, float).reshape(-1)
        if np.any(x_next < lo - 1e-6) or np.any(x_next > hi + 1e-6):
            return False
        return Z.contains(x_next, tol=tol)

    def enclosure(self, phi, u):
        Z = self.next_state_set(phi, u)
        return Zonotope(np.zeros(Z.dim), Z.G), {"additive": float(np.sum(np.abs(Z.G)))}


# --------------------------------------------------------------------------- #
def least_squares(Phi, U, Xp):
    """Ordinary least squares  [A B] = Xp Theta^+  (complexity reference)."""
    t0 = time.perf_counter()
    Theta = np.vstack([np.atleast_2d(Phi), np.atleast_2d(U)])
    M = Xp @ np.linalg.pinv(Theta)
    dt = time.perf_counter() - t0
    p = np.atleast_2d(Phi).shape[0]
    res = BaselineResult(method="LS", feasible=True, CA=M[:, :p], CB=M[:, p:],
                         cw=np.zeros(Xp.shape[0]), solve_time=dt)
    res.meta["residuals"] = Xp - M @ Theta
    return res


# --------------------------------------------------------------------------- #
def set_membership(Phi, U, Xp, w_bound, solver="CLARABEL"):
    """
    Set-membership identification [13], [14].

    Feasible parameter set
        FPS = {(A,B) : |x_{k+1} - A phi_k - B u_k| <= w_bound  elementwise, all k},
    a polytope in the parameters, outer-bounded here by the tightest interval
    matrix (2 LPs per entry).  Assumes `w_bound` is KNOWN -- exactly the
    assumption ICM removes -- so this baseline is also the natural vehicle for
    showing what happens when that bound is wrong.
    """
    Phi = np.atleast_2d(Phi)
    U = np.atleast_2d(U)
    p, N = Phi.shape
    n = Xp.shape[0]
    m = U.shape[0]
    w_bound = np.asarray(w_bound, float).reshape(-1)

    t0 = time.perf_counter()
    Alo = np.zeros((n, p)); Ahi = np.zeros((n, p))
    Blo = np.zeros((n, m)); Bhi = np.zeros((n, m))
    feasible = True
    Theta = np.vstack([Phi, U])                         # (p+m, N)

    for i in range(n):                                  # rows decouple
        theta = cp.Variable(p + m)
        base = [cp.abs(Xp[i, :] - theta @ Theta) <= w_bound[i]]
        for j in range(p + m):
            for sgn, store in ((1.0, "hi"), (-1.0, "lo")):
                prob = cp.Problem(cp.Maximize(sgn * theta[j]), base)
                st, _ = safe_solve(prob, solver=solver)
                if not solved_ok(st):
                    feasible = False
                    break
                val = sgn * prob.value
                tgt = (Ahi if j < p else Bhi) if store == "hi" else (Alo if j < p else Blo)
                tgt[i, j if j < p else j - p] = val
            if not feasible:
                break
        if not feasible:
            break
    dt = time.perf_counter() - t0

    if not feasible:
        return BaselineResult(method="SMI", feasible=False, status="infeasible",
                              solve_time=dt)
    res = BaselineResult(method="SMI", feasible=True,
                         CA=0.5 * (Alo + Ahi), CB=0.5 * (Blo + Bhi),
                         cw=np.zeros(n),
                         W_set=Zonotope(np.zeros(n), np.diag(w_bound)),
                         A_interval=(Alo, Ahi), B_interval=(Blo, Bhi),
                         solve_time=dt)
    res.meta["A_width"] = float(np.sum(Ahi - Alo))
    res.meta["B_width"] = float(np.sum(Bhi - Blo))
    return res


# --------------------------------------------------------------------------- #
def tube_baseline(Phi, U, Xp, Gw, X_region, mu_max=None, solver="CLARABEL"):
    """
    Data-driven tube / zonotopic predictive control baseline in the spirit
    of [24]: a *point* nominal model (least squares) with all remaining
    uncertainty lumped into one additive zonotopic disturbance set, obtained as
    the smallest envelope-scaled <0, Gw diag(mu)> containing every residual.

    Note: unlike ICM/DCM the envelope is NOT capped at mu <= 1.  The cap in
    Eq. (10d) normalizes the template against a prior magnitude, which this
    baseline does not use; leaving mu free is the favourable reading for the
    baseline and avoids trivially declaring it infeasible.
    """
    t0 = time.perf_counter()
    ls = least_squares(Phi, U, Xp)
    R = ls.meta["residuals"]                            # n x N
    n, N = R.shape
    sw = Gw.shape[1]

    mu = cp.Variable(sw, nonneg=True)
    lam = cp.Variable((sw, N))
    c = cp.Variable(n)
    sX = cp.Variable(nonneg=True)
    cons = [R == cp.reshape(c, (n, 1), order="C") @ np.ones((1, N)) + Gw @ lam,
            cp.abs(lam) <= cp.reshape(mu, (sw, 1), order="C") @ np.ones((1, N)),
            X_region.H @ c + np.abs(X_region.H @ Gw) @ mu <= sX * X_region.h]
    if mu_max is not None:
        cons.append(mu <= mu_max)
    prob = cp.Problem(cp.Minimize(sX + 0.1 * cp.sum(mu)), cons)
    status = "infeasible"
    status, _ = safe_solve(prob, solver=solver)
    dt = time.perf_counter() - t0

    if status not in ("optimal", "optimal_inaccurate"):
        return BaselineResult(method="Tube-MPC", feasible=False,
                              status=status, solve_time=dt)
    muv = np.maximum(np.array(mu.value).reshape(-1), 0.0)
    return BaselineResult(method="Tube-MPC", feasible=True,
                          CA=ls.CA, CB=ls.CB, cw=np.array(c.value).reshape(-1),
                          W_set=Zonotope(np.zeros(n), Gw * muv[None, :]),
                          sX=float(sX.value), solve_time=dt)
