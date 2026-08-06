"""
Evaluation metrics shared by the experiments.

The headline metric is *validation coverage*: the fraction of held-out
transitions of the true plant that the identified model actually explains,
measured both inside the identification region and on progressively larger
extrapolation regions.  Coverage is what a safety guarantee rests on; the
scaling factor s_X alone does not distinguish a tight model from an
under-approximating one.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .solver_utils import safe_solve, solved_ok


def sample_validation(system, n_points, rng, radius, u_range=0.5,
                      w_gen=None, basis: Optional[Callable] = None,
                      state_dim=None):
    """Held-out transitions with states drawn uniformly from a box of `radius`."""
    n = state_dim or system.n
    m = system.B.shape[1] if hasattr(system, "B") else system.m_in
    Phi, U, Xp, Xs = [], [], [], []
    for _ in range(n_points):
        x = rng.uniform(-radius, radius, size=n)
        u = rng.uniform(-u_range, u_range, size=m)
        w = w_gen @ rng.uniform(-1, 1, size=w_gen.shape[1]) if w_gen is not None \
            else np.zeros(n)
        Xs.append(x)
        U.append(u)
        Xp.append(system.step(x, u, w))
        Phi.append(x if basis is None else basis(x))
    return (np.array(Phi).T, np.array(U).T, np.array(Xp).T, np.array(Xs).T)


def coverage(model, Phi, U, Xp):
    """Fraction of transitions contained in the model's one-step set."""
    if model is None or not getattr(model, "feasible", False):
        return np.nan
    ok = 0
    N = Phi.shape[1]
    for k in range(N):
        try:
            ok += int(model.covers(Phi[:, k], U[:, k], Xp[:, k]))
        except Exception:
            pass
    return ok / N


def enclosure_size(model, Phi, U):
    """Mean L1 generator mass of the certified one-step enclosure (tightness)."""
    if model is None or not getattr(model, "feasible", False):
        return np.nan
    vals = []
    for k in range(Phi.shape[1]):
        Z, _ = model.enclosure(Phi[:, k], U[:, k])
        vals.append(float(np.sum(np.abs(Z.G))))
    return float(np.mean(vals))


def enclosure_by_source(model, Phi, U):
    """Mean per-source contribution (state / input / additive) -- Lemma 5(2)."""
    if model is None or not getattr(model, "feasible", False):
        return {}
    acc = {}
    for k in range(Phi.shape[1]):
        _, parts = model.enclosure(Phi[:, k], U[:, k])
        for key, v in parts.items():
            acc.setdefault(key, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def contains_true_system(model, A_tr, B_tr, tol=1e-6):
    """Is the true (A_tr, B_tr) inside the identified matrix zonotopes?"""
    if model is None or not getattr(model, "feasible", False):
        return False
    try:
        return bool(model.A_set.contains_matrix(A_tr, tol)
                    and model.B_set.contains_matrix(B_tr, tol))
    except Exception:
        return False


def representability_gap(model, A_tr, B_tr, Phi, U, solver="CLARABEL"):
    """
    Direct test of Assumption 2.  The assumption asks for the existence of a
    *single* pair (A^1, B^1) in the identified sets such that

        (A_tr - A^1) phi_k + (B_tr - B^1) u_k  in  What      for all k,

    so it must be checked by optimizing over the sets, not by plugging in the
    centres.  This solves

        min t  s.t.  A^1 in Ahat, B^1 in Bhat,
                     (A_tr-A^1) phi_k + (B_tr-B^1) u_k - c^w = G^w_scaled lam_k,
                     |lam_k| <= t,

    and returns t*.  t* <= 1 means Assumption 2 holds with the identified sets;
    t* > 1 quantifies by how much the disturbance set would have to be inflated;
    np.inf means the mismatch leaves the range of G^w and no scaling suffices.
    """
    import cvxpy as cp

    if model is None or not getattr(model, "feasible", False):
        return np.nan
    A_tr = np.atleast_2d(A_tr)
    B_tr = np.atleast_2d(B_tr)
    GA = getattr(model.A_set, "blocks", [])
    GB = getattr(model.B_set, "blocks", [])
    Gw = model.W_set.G
    cw = model.W_set.c
    sA, sB, sw = len(GA), len(GB), Gw.shape[1]
    N = Phi.shape[1]

    t = cp.Variable(nonneg=True)
    lamk = cp.Variable((sw, N))
    cons = [cp.abs(lamk) <= cp.reshape(t * np.ones(sw), (sw, 1), order="C")
            @ np.ones((1, N))]
    lamA = cp.Variable(sA) if sA else None
    lamB = cp.Variable(sB) if sB else None
    if sA:
        cons.append(cp.abs(lamA) <= 1)
    if sB:
        cons.append(cp.abs(lamB) <= 1)

    dA0 = A_tr - model.CA
    dB0 = B_tr - model.CB
    for k in range(N):
        resid = dA0 @ Phi[:, k] + dB0 @ U[:, k] - cw
        if sA:
            MA = np.column_stack([G @ Phi[:, k] for G in GA])
            resid = resid - MA @ lamA
        if sB:
            MB = np.column_stack([G @ U[:, k] for G in GB])
            resid = resid - MB @ lamB
        cons.append(resid == Gw @ lamk[:, k])

    prob = cp.Problem(cp.Minimize(t), cons)
    status, _ = safe_solve(prob, solver=solver)
    if not solved_ok(status):
        return np.inf
    return float(t.value)
