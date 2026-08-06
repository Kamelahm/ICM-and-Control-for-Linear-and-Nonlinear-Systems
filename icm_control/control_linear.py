"""
Robust control synthesis for linear systems: the lambda-contractiveness linear
program of Theorem 1, Eq. (19), plus closed-loop evaluation utilities.

The synthesis is written against a light-weight adapter so that ICM, DCM, SMI
and the tube baseline can all be fed to the *same* controller and compared on
equal terms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import cvxpy as cp

from .solver_utils import safe_solve, solved_ok
import numpy as np

from .sets import Polytope, Zonotope


# --------------------------------------------------------------------------- #
@dataclass
class UncertainModel:
    """(CA, CB, cw) with generator lists for the state-, input- and additive part."""
    CA: np.ndarray
    CB: np.ndarray
    cw: np.ndarray
    GA: List[np.ndarray]          # scaled blocks mu_i G^A_i  (n x n)
    GB: List[np.ndarray]          # scaled blocks mu_i G^B_i  (n x m)
    Gw: np.ndarray                # scaled additive generators (n x s_w)
    name: str = ""


def to_uncertain_model(res) -> Optional[UncertainModel]:
    """Adapter for IdentificationResult / BaselineResult."""
    if not getattr(res, "feasible", False):
        return None
    name = getattr(res, "method", "")
    CA, CB, cw = res.CA, res.CB, res.cw
    GA, GB = [], []
    if hasattr(res, "A_set_template"):                      # ICM family
        GA = [m * G for m, G in zip(res.muA, res.A_set_template)]
        GB = [m * G for m, G in zip(res.muB, res.B_set_template)]
        Gw = res.Gw * res.muw[None, :]
    else:                                                   # baselines
        Gw = res.W_set.G if res.W_set is not None else np.zeros((CA.shape[0], 1))
        if getattr(res, "A_interval", None) is not None:    # SMI interval matrix
            radA = 0.5 * (res.A_interval[1] - res.A_interval[0])
            radB = 0.5 * (res.B_interval[1] - res.B_interval[0])
            n, p = radA.shape
            for i in range(n):
                for j in range(p):
                    if radA[i, j] > 1e-12:
                        E = np.zeros((n, p)); E[i, j] = radA[i, j]; GA.append(E)
            for i in range(radB.shape[0]):
                for j in range(radB.shape[1]):
                    if radB[i, j] > 1e-12:
                        E = np.zeros(radB.shape); E[i, j] = radB[i, j]; GB.append(E)
    return UncertainModel(CA, CB, cw, GA, GB, Gw, name)


# --------------------------------------------------------------------------- #
@dataclass
class ControlResult:
    feasible: bool
    status: str
    K: Optional[np.ndarray] = None
    P: Optional[np.ndarray] = None
    rho: float = np.nan
    solve_time: float = np.nan


def lambda_contractive_lp(model: UncertainModel,
                          safe_set: Polytope,
                          lam: float = 0.9,
                          Mx: Optional[float] = None,
                          solver: str = "CLARABEL", norm_type: str = "inf") -> ControlResult:
    """
    Theorem 1 / Eq. (19).  Renders the polyhedral safe set S_s = <H,h>_P
    robustly lambda-contractive for every realization of the identified sets.

        min rho
        s.t.  P h <= lam h - H c^w - Mx sum_i ||H_j,: G^A_i||
                     - rho Mx sum_i ||H_j,: G^B_i|| - sum_i |H_j,: G^w_:,i|
              P H = H (C^A + C^B K),   ||K|| <= rho,   P >= 0
    """
    if model is None:
        return ControlResult(False, "no_model")
    H, h = safe_set.H, safe_set.h
    q, n = H.shape
    m = model.CB.shape[1]
    if Mx is None:
        Mx = safe_set.max_norm_bound()

    P = cp.Variable((q, q), nonneg=True)
    K = cp.Variable((m, n))
    rho = cp.Variable(nonneg=True)

    # constant per-row aggregates
    aA = np.array([sum(np.linalg.norm(H[j, :] @ G) for G in model.GA) if model.GA else 0.0
                   for j in range(q)])
    aB = np.array([sum(np.linalg.norm(H[j, :] @ G) for G in model.GB) if model.GB else 0.0
                   for j in range(q)])
    aW = np.sum(np.abs(H @ model.Gw), axis=1)

    K_bound = cp.norm(K, "inf") if norm_type == "inf" else cp.norm(K, 2)
    cons = [P @ h <= lam * h - H @ model.cw - Mx * aA - rho * Mx * aB - aW,
            P @ H == H @ (model.CA + model.CB @ K),
            K_bound <= rho]

    prob = cp.Problem(cp.Minimize(rho), cons)
    t0 = time.perf_counter()
    status, _ = safe_solve(prob, solver=solver)
    dt = time.perf_counter() - t0

    ok = solved_ok(status)
    return ControlResult(ok, status,
                         K=np.array(K.value).reshape(m, n) if ok else None,
                         P=np.array(P.value) if ok else None,
                         rho=float(rho.value) if ok else np.nan,
                         solve_time=dt)


# --------------------------------------------------------------------------- #
def min_feasible_lambda(model: UncertainModel, safe_set: Polytope,
                        Mx: Optional[float] = None, tol=5e-3, solver="CLARABEL", norm_type: str = "inf"):
    """
    Smallest lambda for which (19) is feasible: a *continuous* measure of how
    much contraction the identified model can certify.  Lower is better; a
    binary feasibility rate at one fixed lambda throws this information away.
    Returns np.nan when even lambda = 1 is infeasible.
    """
    if model is None:
        return np.nan
    if Mx is None:
        Mx = safe_set.max_norm_bound()
    if not lambda_contractive_lp(model, safe_set, 1.0, Mx, solver,
                                 norm_type=norm_type).feasible:
        return np.nan
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if lambda_contractive_lp(model, safe_set, mid, Mx, solver,
                                 norm_type=norm_type).feasible:
            hi = mid
        else:
            lo = mid
    return hi


def closed_loop_violation(system, K, safe_set: Polytope, W_true: Zonotope,
                          rng, n_traj=50, horizon=40, x0_scale=0.9):
    """
    Monte-Carlo check of the *true* closed loop: fraction of trajectories that
    leave the safe set, and the worst constraint residual observed.
    A synthesis certificate that is not valid for the true plant shows up here.
    """
    n = system.n
    viol, worst = 0, -np.inf
    for _ in range(n_traj):
        lo, hi = -x0_scale * np.ones(n), x0_scale * np.ones(n)
        x = rng.uniform(lo, hi)
        # keep the initial condition inside the safe set
        while not safe_set.contains(x):
            x = rng.uniform(lo, hi)
        left = False
        for _ in range(horizon):
            u = K @ x
            w = W_true.c + W_true.G @ rng.uniform(-1, 1, size=W_true.n_gen)
            x = system.step(x, u, w)
            resid = float(np.max(safe_set.H @ x - safe_set.h))
            worst = max(worst, resid)
            if resid > 1e-9:
                left = True
                break
        viol += int(left)
    return viol / n_traj, worst
