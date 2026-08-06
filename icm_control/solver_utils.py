"""
One place to call `prob.solve` from.

WHY THIS EXISTS
---------------
CLARABEL is a Rust extension. When its eigendecomposition fails inside a PSD
cone it does not raise a Python exception -- it panics, and PyO3 surfaces the
panic as `pyo3_runtime.PanicException`:

    thread '<unnamed>' panicked at src/solver/core/cones/psdtrianglecone.rs:453
    Eigval error: Eigen(1)

`PanicException` derives from `BaseException`, NOT from `Exception`, so every
`except Exception:` in this package used to walk straight past it and the run
died mid-sweep. Catching it requires `BaseException`, which in turn means
re-raising `KeyboardInterrupt` and `SystemExit` explicitly so Ctrl-C still
works.

A panic also leaves the solver's internal state unusable, so the same problem
object is re-solved with a different solver rather than retried on the same one.

WHERE THE PSD CONE COMES FROM
-----------------------------
Constraint (19d) is written `||K|| <= rho`. Read as the induced 2-norm,
`cp.norm(K, 2)` lowers to a PSD cone, so the "linear program" of Theorem 1 is
in fact an SDP -- which is also what exposes it to this panic. See
`control_linear.lambda_contractive_lp`, whose `norm_type` argument defaults to
the Frobenius norm: since `||K||_2 <= ||K||_F`, bounding the Frobenius norm is a
*sufficient* condition for (19d), keeps the program conic-quadratic, and removes
the PSD cone that triggers the failure.
"""

from __future__ import annotations

import cvxpy as cp

OK_STATUS = ("optimal", "optimal_inaccurate")

# Order matters: CLARABEL first for speed, SCS as the fallback because it is
# pure-C and does not panic.
DEFAULT_CHAIN = ("CLARABEL", "SCS")


def safe_solve(prob: cp.Problem, solver: str = "CLARABEL",
               fallbacks=("SCS",), verbose: bool = False, **kwargs):
    """
    Solve `prob`, tolerating solver crashes as well as solver failures.

    Returns (status, used_solver).  `status` is cvxpy's status string, or
    "solver_error" if every solver in the chain failed.  Never raises for a
    numerical failure; `KeyboardInterrupt` and `SystemExit` still propagate.
    """
    chain = [solver] + [s for s in fallbacks if s != solver]
    last = "solver_error"
    for name in chain:
        try:
            prob.solve(solver=getattr(cp, name), verbose=verbose, **kwargs)
            last = prob.status
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            # Includes pyo3_runtime.PanicException from CLARABEL, which is a
            # BaseException and would otherwise escape an `except Exception`.
            last = "solver_error"
            continue
        if last in OK_STATUS:
            return last, name
    return last, None


def solved_ok(status) -> bool:
    return status in OK_STATUS
