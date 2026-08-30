"""
Compliance checking for Assumption 2 (linear) / Assumption 5 (nonlinear).

The assumption has two clauses:

  (i)  side-information consistency:  A_tr in A_side,  B_tr in B_side
  (ii) representability:  there exist A^1 in Ahat, B^1 in Bhat such that
       (A_tr - A^1) x + (B_tr - B^1) u  in  What  for all admissible (x, u)

"""

from __future__ import annotations

import numpy as np

from .evaluation import representability_gap

DEFAULT_MU_FLOOR = 0.05


CALIBRATED_GAMMA = {"pendulum": 27.26, "scalar_linear": 2.65}
CALIBRATION_TARGET = 0.25


def check_side_info(A_tr, B_tr, A_side, B_side, tol=1e-9):
    """
    Clause (i): A_tr in A_side and B_tr in B_side.

    Membership in a matrix zonotope is a linear feasibility problem, but for the
    axis-aligned (entrywise) priors used throughout the package the interval
    over-approximation is exact, so an elementwise interval test suffices and is
    exact here.  `margin` is the smallest slack across all entries; negative
    means the true matrix falls outside, and its magnitude says by how much.
    """
    out = {}
    for name, M_tr, S in (("A", A_tr, A_side), ("B", B_tr, B_side)):
        if S is None:
            out[f"{name}_ok"], out[f"{name}_margin"] = True, np.inf
            continue
        lo, hi = S.interval()
        M = np.atleast_2d(np.asarray(M_tr, float))
        margin = float(np.min(np.minimum(M - lo, hi - M)))
        out[f"{name}_ok"] = bool(margin >= -tol)
        out[f"{name}_margin"] = margin
    out["ok"] = bool(out["A_ok"] and out["B_ok"])
    return out


def check_assumption(model, A_tr, B_tr, Phi, U, A_side=None, B_side=None,
                     solver="CLARABEL"):
    """
    Both clauses.  Returns a dict with `holds` True only if the side
    information contains the true matrices AND t* <= 1.

    t* is the factor by which mu_w would have to be inflated for the true
    dynamics to be explained; t* = inf means the mismatch leaves range(G^w) and
    no scaling suffices, so the disturbance template itself is too narrow.
    """
    side = check_side_info(A_tr, B_tr, A_side, B_side)
    t_star = representability_gap(model, A_tr, B_tr, Phi, U, solver=solver)
    repr_ok = bool(np.isfinite(t_star) and t_star <= 1.0 + 1e-6)
    return {"side_info_ok": side["ok"],
            "side_margin_A": side["A_margin"], "side_margin_B": side["B_margin"],
            "t_star": float(t_star), "repr_ok": repr_ok,
            "holds": bool(side["ok"] and repr_ok)}


def calibrate_template_scale(identify_fn, pilot_seeds=(-1, -2, -3, -4, -5),
                             lo=1.0, hi=256.0, tol=1e-2, quantile=1.0,
                             target=0.5, verbose=False):
    """
    Smallest template scale gamma for which clause (ii) holds on the pilot seeds.

    """
    def ok_at(gamma):
        holds = []
        for s in pilot_seeds:
            res = identify_fn(int(s), float(gamma))
            if res is None:
                continue
            holds.append(bool(np.isfinite(res["t_star"])
                              and res["t_star"] <= target))
        if not holds:
            return False
        return float(np.mean(holds)) >= quantile - 1e-9

    if not ok_at(hi):
        if verbose:
            print(f"  [calibrate] t* > 1 even at gamma={hi}: G^w directions "
                  f"are deficient, not merely its magnitude")
        return float("nan")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if ok_at(mid):
            hi = mid
        else:
            lo = mid
        if verbose:
            print(f"  [calibrate] bracket [{lo:.3f}, {hi:.3f}]")
    return float(hi)


def summarize_compliance(records, name="Assumption 2"):
    """
    Aggregate per-trial `check_assumption` dicts into a reportable summary.

    `t_star_max` is the number to quote: the assumption is a for-all statement
    over trials, so the worst case is what governs, not the mean.
    """
    recs = [r for r in records if r is not None]
    if not recs:
        return {"n": 0, "holds_frac": np.nan}
    t = np.array([r["t_star"] for r in recs], float)
    finite = t[np.isfinite(t)]
    return {"name": name, "n": len(recs),
            "holds_frac": float(np.mean([r["holds"] for r in recs])),
            "side_info_frac": float(np.mean([r["side_info_ok"] for r in recs])),
            "t_star_median": float(np.median(finite)) if finite.size else np.inf,
            "t_star_max": float(np.max(finite)) if finite.size else np.inf,
            "t_star_infinite": int(np.sum(~np.isfinite(t)))}
