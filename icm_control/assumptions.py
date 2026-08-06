"""
Compliance checking for Assumption 2 (linear) / Assumption 5 (nonlinear).

The assumption has two clauses:

  (i)  side-information consistency:  A_tr in A_side,  B_tr in B_side
  (ii) representability:  there exist A^1 in Ahat, B^1 in Bhat such that
       (A_tr - A^1) x + (B_tr - B^1) u  in  What  for all admissible (x, u)

Clause (i) is a property of the problem data and holds by construction whenever
`linear_side_info` / `pendulum_side_info` are called with `bias=0`, since those
centre the prior on the true matrices.  `check_side_info` verifies it anyway.

Clause (ii) is a property of the COP *output*, and the minimum-envelope
objective works against it: the program returns the smallest sets that explain
the training data, which generically do not contain the true system.  Measured
by `representability_gap`, the index t* ran from 1.16 to 17.8 over eight
pendulum seeds with no floor -- i.e. clause (ii) failed every time.

MAKING THE BENCHMARK SATISFY THE ASSUMPTION
-------------------------------------------
A mu_w floor does NOT suffice, and it is worth being precise about why.
Constraint (10d) caps mu_w <= 1, so the identified set always satisfies

    What  =  <c^w, G^w diag(mu_w)>  subseteq  <c^w, G^w>,

whatever the floor.  The floor can drive mu_w up to 1 and no further, so the
reachable ceiling is the TEMPLATE itself.  Measured on the pendulum with
G^w = 0.02 I, the required inflation was t* ~ 14 (median) with three seeds at
infinity, and no floor in [0, 4] made clause (ii) hold.

Assumption 2/5 is therefore a condition on the disturbance TEMPLATE G^w, chosen
a priori, and not on any quantity the COP selects.  `calibrate_template_scale`
finds the smallest gamma such that G^w <- gamma G^w yields t* <= 1 on a PILOT
set of seeds.  The mu_w floor remains useful for a different purpose --- keeping
What full dimensional so that t* is finite rather than infinite --- and the two
are applied together.

Two rules govern how that floor is then used, and both matter for the honesty of
the reported numbers:

  1. Calibrate ONCE on pilot seeds, then FIX gamma for all reported trials.
     Re-tuning per trial against that trial's t* would guarantee compliance by
     construction and conceal exactly the failures the check exists to surface.

  2. The pilot seeds must be DISJOINT from the reported seeds.  `calibrate_mu_floor`
     defaults to negative seeds for this reason.

Calibration uses A_tr and B_tr, so it is benchmark design, not part of the
method: it configures a simulation that satisfies a standing assumption.  A
practitioner without the true dynamics cannot run it, and gamma must therefore
be reported as a stated experimental constant, not presented as something the
algorithm selects.  In deployment the template is chosen from physical
knowledge of the disturbance magnitude, and Assumption 2/5 is the formal
statement that this choice was generous enough.

Enlarging the template is not free.  It raises the state-set factor s_X and
loosens every downstream bound, so gamma should be reported alongside its cost
in s_X, and the achieved t* distribution reported alongside gamma.
"""

from __future__ import annotations

import numpy as np

from .evaluation import representability_gap

# Keeps What full dimensional so that t* is finite; does NOT by itself make
# clause (ii) hold -- see the module docstring.
DEFAULT_MU_FLOOR = 0.05

# Template scales calibrated on pilot seeds (-1..-4) at target t* <= 0.25, via
# `scripts/calibrate_assumptions.py`.  Verified on 10 disjoint reported seeds:
#
#   benchmark        gamma   Assum. holds   t* median / max   s_X (gamma=1 -> gamma)
#   pendulum         27.26      100%          0.129 / 0.239     0.0019 -> 0.0055
#   scalar linear     2.65      100%          0.076 / 0.240     0.0008 -> 0.0078
#
# These are STATED EXPERIMENTAL CONSTANTS, not quantities the method selects.
# Enlarging the template is what buys compliance, and s_X is what it costs: a
# 2.9x and 9.8x increase respectively.  Report both.
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

    `identify_fn(seed, gamma)` must run one identification with the disturbance
    template scaled by gamma and return the dict produced by `check_assumption`
    (or None if identification failed).

    `quantile` selects how strict the requirement is: 1.0 demands t* <= 1 on
    every pilot seed, 0.8 on 80% of them.  Use 1.0 unless a single outlier seed
    is driving gamma to a value that cripples the rest, and say which was used
    when reporting.

    `target` is the t* level demanded on the PILOT seeds, and should be well
    below 1.  Calibrating to t* <= 1 exactly leaves no headroom: t* varies from
    seed to seed, so a gamma that just clears 1 on the pilot fails on roughly
    half the reported seeds.  target=0.5 buys a 2x margin.  Report the achieved
    t* distribution on the reported seeds, not the pilot target.

    Monotonicity: enlarging the template can only make the representability LP
    easier, so t* is non-increasing in gamma and bisection is valid.  Returns
    NaN if even `hi` fails, which indicates that G^w has the wrong DIRECTIONS
    rather than the wrong magnitude -- add generators instead of scaling.
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
