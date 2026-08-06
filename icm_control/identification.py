"""
Set-based identification: the ICM convex program of Lemma 2 / Lemma 6 and the
DCM program of Eq. (12).

A single solver `solve_conformant_model` covers all three arms of the ablation
requested by the reviewers, so that the objective, disturbance template,
operating region and solver tolerances are *identical* across arms and only the
feature under test changes:

    zonotopic=False, side_info=False   ->  DCM              (Eq. 12)
    zonotopic=True,  side_info=False   ->  ICM w/o side info
    zonotopic=True,  side_info=True    ->  full ICM         (Eq. 10)

Linear (Lemma 2) and nonlinear (Lemma 6) identification differ only in the
regressor passed as `Phi`: Phi = X for (5), Phi = Z(X) for (6).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cvxpy as cp

from .solver_utils import safe_solve, solved_ok
import numpy as np

from .sets import MatrixZonotope, Polytope, Zonotope


# --------------------------------------------------------------------------- #
#  Result container
# --------------------------------------------------------------------------- #
@dataclass
class IdentificationResult:
    status: str
    feasible: bool
    method: str
    CA: Optional[np.ndarray] = None
    CB: Optional[np.ndarray] = None
    cw: Optional[np.ndarray] = None
    muA: Optional[np.ndarray] = None
    muB: Optional[np.ndarray] = None
    muw: Optional[np.ndarray] = None
    sX: float = np.nan
    objective: float = np.nan
    solve_time: float = np.nan
    n_variables: int = 0
    n_constraints: int = 0
    A_set: Optional[MatrixZonotope] = None
    B_set: Optional[MatrixZonotope] = None
    W_set: Optional[Zonotope] = None
    meta: dict = field(default_factory=dict)

    # -- derived quantities -------------------------------------------------
    def enclosure(self, phi, u):
        """
        Certified one-step error enclosure E_ICM(x,u) of Lemma 5, Eq. (18):
        origin-centred, decomposed by source, and state/input dependent.
        Returns (Zonotope, dict of per-source radii).
        """
        phi = np.asarray(phi, float).reshape(-1)
        u = np.asarray(u, float).reshape(-1)
        gens, parts = [], {}
        gA = np.column_stack([m * (B @ phi) for m, B in
                              zip(self.muA, self.A_set_template)]) \
            if self.A_set_template else np.zeros((self.cw.size, 0))
        gB = np.column_stack([m * (B @ u) for m, B in
                              zip(self.muB, self.B_set_template)]) \
            if self.B_set_template else np.zeros((self.cw.size, 0))
        gW = self.Gw * self.muw[None, :]
        for nm, g in (("state", gA), ("input", gB), ("additive", gW)):
            if g.size:
                gens.append(g)
            parts[nm] = float(np.sum(np.abs(g))) if g.size else 0.0
        G = np.hstack(gens) if gens else np.zeros((self.cw.size, 1))
        return Zonotope(np.zeros(self.cw.size), G), parts

    def predict(self, phi, u):
        """Nominal (centre) one-step prediction, Definition 5."""
        return self.CA @ np.asarray(phi, float).reshape(-1) \
            + self.CB @ np.asarray(u, float).reshape(-1) + self.cw

    def next_state_set(self, phi, u):
        """Full one-step reachable set  A phi + B u + W  (a zonotope)."""
        E, _ = self.enclosure(phi, u)
        return Zonotope(self.predict(phi, u), E.G)

    def covers(self, phi, u, x_next, tol=1e-6):
        """Does the identified model explain the transition (phi,u) -> x_next?"""
        Zn = self.next_state_set(phi, u)
        lo, hi = Zn.interval()
        x_next = np.asarray(x_next, float).reshape(-1)
        if np.any(x_next < lo - 1e-6) or np.any(x_next > hi + 1e-6):
            return False                     # cheap necessary test first
        return Zn.contains(x_next, tol=tol)


# --------------------------------------------------------------------------- #
#  Main solver
# --------------------------------------------------------------------------- #
def solve_conformant_model(Phi: np.ndarray,
                           U: np.ndarray,
                           Xp: np.ndarray,
                           Gw: np.ndarray,
                           X_region: Polytope,
                           GA_blocks: Optional[Sequence[np.ndarray]] = None,
                           GB_blocks: Optional[Sequence[np.ndarray]] = None,
                           A_side: Optional[MatrixZonotope] = None,
                           B_side: Optional[MatrixZonotope] = None,
                           zonotopic: bool = True,
                           side_info: bool = True,
                           weights=(0.10, 0.10, 0.10, 1.0),
                           mu_floor: float = 0.0,
                           solver: str = "CLARABEL",
                           verbose: bool = False,
                           method_name: Optional[str] = None) -> IdentificationResult:
    """
    Solve the conformance program.

    Parameters
    ----------
    Phi  : (p, N) regressor   -- x_k (linear, Lemma 2) or Z(x_k) (nonlinear, Lemma 6)
    U    : (m, N) inputs
    Xp   : (n, N) successors  x_{k+1}
    Gw   : (n, s_w) disturbance generator template (fixed a priori)
    GA_blocks / GB_blocks : fixed generator directions, blocks of shape (n,p)/(n,m)
    A_side / B_side       : prior sets  A_side, B_side  (matrix zonotopes)
    zonotopic : if False the dynamics collapse to point estimates -> DCM
    side_info : if False the containment constraints (10e)-(10f) are dropped
    weights   : (nu_A, nu_B, nu_w, nu_s) in the objective J_M
    mu_floor  : lower bound on mu_w.  The minimum-volume objective otherwise
                drives individual mu_w entries to zero, which makes the
                identified What lower dimensional; the representability
                condition of Assumption 2 can then never hold, because a
                mismatch with a component outside range(G^w diag(mu_w)) is not
                absorbed at any scaling.  A small floor keeps What full
                dimensional at negligible cost in volume.

    Objective
    ---------
        J_M = nu_A 1'mu_A + nu_B 1'mu_B + nu_w 1'mu_w + nu_s s_X
    convex and entrywise nondecreasing in (s_X, mu_A, mu_B, mu_w), as required
    by Lemma 2.
    """
    Phi = np.atleast_2d(Phi)
    U = np.atleast_2d(U)
    Xp = np.atleast_2d(Xp)
    Gw = np.atleast_2d(Gw)
    p, N = Phi.shape
    n = Xp.shape[0]
    m = U.shape[0]
    sw = Gw.shape[1]
    nuA, nuB, nuw, nus = weights

    if not zonotopic:
        GA_blocks, GB_blocks = [], []
    GA_blocks = list(GA_blocks or [])
    GB_blocks = list(GB_blocks or [])
    sA, sB = len(GA_blocks), len(GB_blocks)

    # ---- decision variables ------------------------------------------------
    CA = cp.Variable((n, p), name="CA")
    CB = cp.Variable((n, m), name="CB")
    cw = cp.Variable(n, name="cw")
    sX = cp.Variable(nonneg=True, name="sX")
    lam = cp.Variable((sw, N), name="lam")
    muw = cp.Variable(sw, nonneg=True, name="muw")
    lamA = cp.Variable((sA, N), name="lamA") if sA else None
    lamB = cp.Variable((sB, N), name="lamB") if sB else None
    muA = cp.Variable(sA, nonneg=True, name="muA") if sA else None
    muB = cp.Variable(sB, nonneg=True, name="muB") if sB else None

    cons = []

    # ---- (10b) data conformance -------------------------------------------
    # sum_i lamA[i,k] G_i phi_k  =  M_k lamA[:,k]  with M_k = [G_1 phi_k ... G_sA phi_k]
    for k in range(N):
        expr = CA @ Phi[:, k] + CB @ U[:, k] + cw + Gw @ lam[:, k]
        if sA:
            Mk = np.column_stack([G @ Phi[:, k] for G in GA_blocks])
            expr = expr + Mk @ lamA[:, k]
        if sB:
            Nk = np.column_stack([G @ U[:, k] for G in GB_blocks])
            expr = expr + Nk @ lamB[:, k]
        cons.append(Xp[:, k] == expr)

    # ---- (10c)-(10d) envelope and box constraints --------------------------
    ones_N = np.ones((1, N))
    cons += [cp.abs(lam) <= cp.reshape(muw, (sw, 1), order="C") @ ones_N,
             muw <= 1]
    if mu_floor > 0:
        cons.append(muw >= mu_floor)
    if sA:
        cons += [cp.abs(lamA) <= cp.reshape(muA, (sA, 1), order="C") @ ones_N,
                 muA <= 1]
    if sB:
        cons += [cp.abs(lamB) <= cp.reshape(muB, (sB, 1), order="C") @ ones_N,
                 muB <= 1]

    # ---- (10e)-(10f) zonotope containment in the side-information sets -----
    # Sadraddini-Tedrake encoding (3) applied to the vectorized matrix zonotopes.
    if side_info and A_side is not None and sA:
        cons += _containment_constraints(CA, muA, GA_blocks, A_side)
    if side_info and B_side is not None and sB:
        cons += _containment_constraints(CB, muB, GB_blocks, B_side)
    if side_info and not zonotopic:
        # point estimates must still lie inside the prior sets
        if A_side is not None:
            cons += _point_containment(CA, A_side)
        if B_side is not None:
            cons += _point_containment(CB, B_side)

    # ---- (10g) disturbance containment in s_X X ----------------------------
    HX, hX = X_region.H, X_region.h
    cons.append(HX @ cw + np.abs(HX @ Gw) @ muw <= sX * hX)

    # ---- objective ---------------------------------------------------------
    obj = nus * sX + nuw * cp.sum(muw)
    if sA:
        obj = obj + nuA * cp.sum(muA)
    if sB:
        obj = obj + nuB * cp.sum(muB)

    prob = cp.Problem(cp.Minimize(obj), cons)
    t0 = time.perf_counter()
    safe_solve(prob, solver=solver, verbose=verbose)
    t_solve = time.perf_counter() - t0

    if method_name is None:
        method_name = ("DCM" if not zonotopic else
                       ("ICM" if side_info else "ICM-noSI"))

    feasible = prob.status in ("optimal", "optimal_inaccurate")
    res = IdentificationResult(
        status=prob.status, feasible=feasible, method=method_name,
        solve_time=t_solve,
        n_variables=int(sum(v.size for v in prob.variables())),
        n_constraints=int(sum(np.prod(c.shape) if c.shape else 1 for c in cons)),
    )
    if not feasible:
        return res

    res.CA = np.array(CA.value).reshape(n, p)
    res.CB = np.array(CB.value).reshape(n, m)
    res.cw = np.array(cw.value).reshape(-1)
    res.muA = np.clip(np.array(muA.value).reshape(-1), 0, 1) if sA else np.zeros(0)
    res.muB = np.clip(np.array(muB.value).reshape(-1), 0, 1) if sB else np.zeros(0)
    res.muw = np.clip(np.array(muw.value).reshape(-1), 0, 1)
    res.sX = float(sX.value)
    res.objective = float(prob.value)
    res.A_set = MatrixZonotope(res.CA, [m_ * G for m_, G in zip(res.muA, GA_blocks)]) \
        if sA else MatrixZonotope(res.CA, [])
    res.B_set = MatrixZonotope(res.CB, [m_ * G for m_, G in zip(res.muB, GB_blocks)]) \
        if sB else MatrixZonotope(res.CB, [])
    res.W_set = Zonotope(res.cw, Gw * res.muw[None, :])
    # templates kept for the enclosure computation
    res.A_set_template = GA_blocks
    res.B_set_template = GB_blocks
    res.Gw = Gw
    return res


def _containment_constraints(C, mu, blocks, side: MatrixZonotope):
    """Eq. (3): G1 = G2 Gamma, c2 - c1 = G2 gamma, ||[Gamma gamma]||_inf <= 1."""
    G2 = side.vec_generators()                 # (n*p, r)
    c2 = side.vec_center()
    r = G2.shape[1]
    s = len(blocks)
    Gamma = cp.Variable((r, s))
    gamma = cp.Variable(r)
    Gvec = np.column_stack([B.reshape(-1, order="F") for B in blocks])
    G1 = Gvec @ cp.diag(mu)
    c1 = cp.vec(C, order="F")
    return [G1 == G2 @ Gamma,
            c2 - c1 == G2 @ gamma,
            cp.norm(cp.hstack([Gamma, cp.reshape(gamma, (r, 1), order="C")]), "inf") <= 1]


def _point_containment(C, side: MatrixZonotope):
    """Membership of a point estimate in a matrix zonotope (DCM + side info)."""
    G2 = side.vec_generators()
    c2 = side.vec_center()
    gamma = cp.Variable(G2.shape[1])
    return [c2 - cp.vec(C, order="F") == G2 @ gamma, cp.norm(gamma, "inf") <= 1]


# --------------------------------------------------------------------------- #
#  Convenience wrappers matching the paper's naming
# --------------------------------------------------------------------------- #
def solve_icm(**kwargs):
    """Full ICM, Eq. (10) / Lemma 2 (linear) or Lemma 6 (nonlinear)."""
    kwargs.setdefault("zonotopic", True)
    kwargs.setdefault("side_info", True)
    return solve_conformant_model(**kwargs)


def solve_icm_no_side_info(**kwargs):
    """ICM with the containment constraints removed (ablation arm 2)."""
    kwargs["zonotopic"] = True
    kwargs["side_info"] = False
    return solve_conformant_model(**kwargs)


def solve_dcm(**kwargs):
    """DCM, Eq. (12): point dynamics, additive disturbance set only."""
    kwargs["zonotopic"] = False
    kwargs["side_info"] = False
    return solve_conformant_model(**kwargs)
