# Changelog

Changes relative to the original `icm_simulations` package.

## Scope: Theorem 4 removed

The ISS / ultimate-boundedness results are no longer part of the paper, so the
package certifies the **disturbance-free** closed loop only.

* Deleted `experiments/exp6_augmented_disturbance.py` and its entry in
  `run_all.py`.
* Deleted `iss_certificate`, `max_certified_wbar`, `augmented_disturbance_bound`
  and the helper `_g_sup` from `icm_control/control_nonlinear.py`.
* `exp2_misspecified_basis.py` reported `wbar_max`; it now reports `c_delta`
  and `delta_star`.

**Non-obvious consequence.** `delta` must be chosen in `(0, delta*)`, and the two
quantities it controls move in opposite directions: `r*` grows with `delta`
while the decay rate `c(delta)` worsens toward 1. Theorem 4's `wbar_max`
balanced the two and so had an interior maximum, which is what
`best_delta_certificate` used to optimize. Maximizing `c*` alone instead drives
`delta -> delta*` and the certified convergence becomes arbitrarily slow.
`best_delta_certificate` now maximizes `c*` subject to an explicit decay budget
`c(delta) <= c_max` (default 0.99). **`c_max` is a reporting choice and belongs
in the figure/table caption**, not buried in a function signature.

## Correctness fixes in `control_nonlinear.py`

1. **`Theta` uses squared radii.** The Young-type bound
   `f (u v^T + v u^T) >= -(tau D^2 u u^T + tau^-1 v v^T)` is valid iff the
   product of the two coefficients is at least `D^2`. The previous
   `tau*abs(D)` / `1/tau` pairing gives product `abs(D)`, which exceeds `D^2`
   only while `abs(D) <= 1` — silently unsound above that, needlessly
   conservative below. `tau*D**2` / `1/tau` gives exactly `D^2` at every
   magnitude, with no division and no blow-up when a radius is zero.

2. **`contraction_factor` bisects.** It previously minimized `kappa` as a free
   variable. The `(1,1)` block `kappa*Pbar` goes singular at the optimum and the
   interior-point method stalls short of it: on the pendulum, CLARABEL returned
   0.9903 and SCS 0.9336 for a problem verifiably feasible at 0.90. Since
   `delta*`, `r*` and `c*` all scale with `(1-kappa)`, the stalled value
   collapsed every Theorem 3 constant. Fixed-`kappa` feasibility is well
   conditioned; ~10 solves give a reliable bound.

3. **`verify_contraction` added.** Monte-Carlo audit of
   `A_cl^T D A_cl <= kappa D` over the identified intervals, sampling box
   vertices as well as interior points (the worst case of a multilinear
   expression over a box sits at a vertex). `sweep_contraction_targets` runs it
   per row under `audit=True`. Necessary check, not a proof — the LMI is the
   proof.

4. **Log-spaced `_g_profile` grid.** `r*` is typically orders of magnitude below
   `r_cap`, so a linear grid over `[0, r_cap]` resolved it with a handful of
   points. `r*` quantized coarsely, `c*` went flat across the `delta` sweep, and
   the selected `delta` was decided by tie-breaking. On the pendulum this
   *overstated* `r*` by roughly 2x (0.030 vs 0.015) and pinned the winning
   `delta_frac` at the bottom of the sweep. Log spacing puts resolution where
   `r*` lives; the corrected `r*` is smaller and the `delta` trade-off is real.

5. **`Pbar >= I` normalization.** W.l.o.g. by homogeneity of the LMI in
   `(Pbar, Ybar, lam, eps2)`; `Kbar = Ybar Pbar^{-1}` is invariant. Fixes the
   scale, without which a solver may return an arbitrarily scaled Lyapunov
   matrix and the certified constants carry no information.

6. **Split `B`-slacks.** The two LMIs bound the `B` uncertainty on independent
   paths (`Khat` and `Kbar`); sharing `V_2` forced one multiplier to serve both.
   Now `eps1` / `eps2`. This is also what makes the homogeneity argument in (5)
   work, since the scaling must leave the cancellation LMI untouched.

## Assumption 2 / 5 compliance

New module `icm_control/assumptions.py` and script
`scripts/calibrate_assumptions.py`.

The assumption has two clauses. Clause (i), side-information consistency, holds
by construction: `linear_side_info` and `pendulum_side_info` centre the prior on
the true matrices when `bias=0`, and `check_side_info` verifies it (100% in every
run). Clause (ii), representability, **failed in every trial as originally
configured** -- `representability_gap` returned t* with median 31.4 and max 155.6
on the pendulum, where the assumption needs t* <= 1.

**A mu_w floor does not fix this, and it is worth being precise why.**
Constraint (10d) caps mu_w <= 1, so the identified set always satisfies
`What = <c^w, G^w diag(mu_w)> subseteq <c^w, G^w>` regardless of the floor. The
floor can drive mu_w to 1 and no further; the reachable ceiling is the TEMPLATE.
No floor in [0, 4] made clause (ii) hold. Assumption 2/5 is therefore a
condition on the disturbance template `G^w`, chosen a priori, not on anything
the COP selects. The floor is still applied, for a different purpose: it keeps
`What` full dimensional so t* is finite rather than infinite (three pendulum
seeds returned t* = inf without it).

`calibrate_template_scale` bisects on a scale gamma with `G^w <- gamma G^w`.
Calibrated on pilot seeds and verified on 10 disjoint reported seeds:

| benchmark | gamma | holds | t* median / max | s_X (1 -> gamma) |
|---|---|---|---|---|
| pendulum | 27.26 | 100% | 0.129 / 0.239 | 0.0019 -> 0.0055 |
| scalar linear | 2.65 | 100% | 0.076 / 0.240 | 0.0008 -> 0.0078 |

Three points on methodology:

* **Calibrate to t\* <= 0.25, not <= 1.** t* varies across seeds, so a gamma that
  just clears 1 on the pilot fails on about half the reported seeds (measured:
  50%). The 4x margin is what gets compliance to 100%.
* **Pilot seeds are disjoint from reported seeds** (negative vs non-negative), and
  gamma is FIXED across reported trials. Re-tuning per trial would guarantee
  compliance by construction and hide the failures the check exists to surface.
* **Calibration uses A_tr, B_tr**, so it is benchmark design, not method. A
  practitioner cannot run it. Report gamma as a stated experimental constant.

Compliance is not free: s_X rises 2.9x on the pendulum and 9.8x on the scalar
system. Report gamma and the achieved t* alongside every result that depends on
the assumption.

## Experiment set reduced

* **`exp1_ablation.py` trimmed to two figures** (`exp1_feasibility_vs_noise`,
  `exp1_coverage_vs_extrapolation`). Three were dropped; the underlying columns
  are still written to `exp1_raw_*.csv` so any can be regenerated.
  - *contraction margin*: its axis read "lower is better" and DCM was lowest
    everywhere, so as drawn it said the proposed method was worst. It was also
    biased toward DCM -- lambda* was averaged only over trials where each arm
    stayed feasible, i.e. the easiest draws for the arm that failed most. The
    table's lambda* column is now restricted to the `n_common` trials on which
    all three arms are feasible.
  - *feasibility vs lambda*: essentially flat; one number per arm.
  - *tightness vs coverage*: superseded by the enclosure-size column, which makes
    the same point numerically (all arms certify near-identical enclosure size,
    so ICM's coverage is not bought by inflating the sets).

* **Two template scales, on purpose.** `--gamma-feas 1.0` and `--gamma-cov 8.0`;
  the sweep runs once per gamma. The figures probe opposite regimes and cannot
  share one template: at gamma = 8 every arm stays feasible at every alpha_1 and
  Fig 1 says nothing, while at gamma = 1 coverage is 7% even with a correctly
  sized template, because the residual is dominated by dynamics mismatch rather
  than process noise. Each caption must state its gamma.

* **Fig 2 axis in manuscript notation.** The ordinate is the one-step set
  `Xhat+(x,u) := Ahat x (+) Bhat u (+) What`, written with a single symbol
  because the arms instantiate it differently: DCM returns zero generator blocks
  (verified: interval width exactly 0), so Ahat, Bhat are singletons and Xhat+
  collapses to the translate of Lemma 5 Part 3, whose width is independent of
  (x,u); the ICM arms are non-degenerate and scale with (x,u), Lemma 5 Part 2.
  The decay of the DCM curve with r/r_id is therefore the predicted consequence
  of a constant-width enclosure, not an incidental finding. The script prints a
  ready caption.

* **`exp4_baselines.py` re-run with the corrected settings, and its second
  figure deleted.** `exp4_baselines_violations` plotted 0.00 for all seven
  methods on a +/-0.04 axis -- a figure asserting that nothing happened. State it
  in the text instead. exp4 also did not expose `gamma` or `mu_floor`, so its
  numbers carried the same stale configuration that put exp1 coverage at 36%
  before calibration: ICM read 33% coverage against SMI's 100%, i.e. the figure
  argued against the method. Defaults are now gamma = 16, mu_floor = 0.20 and
  alpha_1 = 1, the last matching exp1 Fig. 2 so the two comparisons are read on
  the same footing. ICM then reaches 100% coverage at enclosure 0.096, against
  SMI(beta=1) at 100% and 0.183 -- while SMI is handed the TRUE disturbance
  bound, which Assumption 1 states is unknown. The `needs $\bar w$` column
  already records that asymmetry and should be kept in the caption.

  Two further changes to that figure. **LS is excluded**: its model is a point
  estimate with `W_set = {0}`, so the one-step set is the singleton
  `{A_LS x + B_LS u}` and containment of x_{k+1} requires exact equality --
  probability zero under a continuous disturbance. Its 0% is definitional, not
  measured, and plotting it at the origin of both axes invites the reading that
  a straw man was beaten. LS stays in the table and remains a meaningful
  zero-cost reference in exp5's timing plot. **Open markers** now denote the
  methods handed the true bound (the SMI family); filled markers denote methods
  that learn it. The axis label is the explicit formula `sum_j ||G_{:,j}||_1`
  rather than "l1 generator mass", which invited readers to look for an l1 ball
  that is not there. The script prints a ready caption.

* **`exp2_misspecified_basis.py` replaced by `exp2_roa.py`.** The old experiment
  studied basis misspecification, which the system model (6)-(7) excludes by
  assuming the nonlinear basis functions known -- the same scope argument that
  retires exp3. Its figures also showed failure rather than graceful
  degradation: dropping `1-cos x1` diverged (x1 reached 4.1 rad and was still
  climbing at k=60) and `linear only` stalled at x1 ~ 0.65 without reaching the
  origin, so two of four configurations broke the controller. The `bounded`
  column reported 1.0 for all four because its threshold was `max|traj| < 10`,
  which the diverging run passed -- the table contradicted the figure.

  `exp2_roa.py` instead tests what the paper proves. It draws the certified
  sublevel set `Omega_c* = {x : x' D x <= c*}` on the phase plane with
  closed-loop trajectories of the TRUE plant started ON ITS BOUNDARY -- the
  worst case for invariance, since any escape must cross there -- alongside
  uncontrolled orbits from the same initial conditions. A trajectory leaving the
  ellipse would falsify Theorem 3. The script counts escapes and prints the
  tally. Defaults use the calibrated gamma = 27.26 and cap r* at pi, the physical
  range of the pendulum angle; the table reports what fraction of trials hit that
  cap, since a capped r* is the region boundary rather than the certificate.

  The test rollout is disturbance-free (`--w-test 0`). Theorem 3 is stated for
  x_{k+1} = (A_cl + Delta_k) x_k with no w, so injecting disturbance tests a claim
  the paper does not make -- with Theorem 4 removed, an escape driven by w would
  not falsify Theorem 3, and no-escape under w is therefore not evidence for it.
  A second check verifies the decay claim V(x_k) <= c(delta)^k V(x_0) directly,
  which is sharper than invariance alone and was previously untested.

* **`exp3_biased_side_info.py` deleted**, with its `run_all.py` entry and README
  row. Assumption 2 asserts `A_tr in A_side` and `B_tr in B_side`, so a biased
  prior lies outside the assumption's scope and the paper is not obliged to
  characterise it. The representability half of the assumption is still measured:
  `representability_gap` feeds the `Repr. gap` column of `exp1_ablation`, and
  `icm_control/assumptions.py` with `scripts/calibrate_assumptions.py` covers it
  more directly than `exp3_assumption2` did.

## Solver robustness (crash fix)

CLARABEL is a Rust extension. When its eigendecomposition fails inside a PSD
cone it does not raise a Python exception -- it **panics**, and PyO3 surfaces
that as `pyo3_runtime.PanicException`:

    thread '<unnamed>' panicked at src/solver/core/cones/psdtrianglecone.rs:453
    Eigval error: Eigen(1)

`PanicException` derives from `BaseException`, **not** from `Exception`, so every
`except Exception:` in the package walked straight past it and killed the run
mid-sweep. Reported from `min_feasible_lambda`'s bisection during `exp1`, at
N=30.

New module `icm_control/solver_utils.py` with `safe_solve`, which catches
`BaseException` (re-raising `KeyboardInterrupt` and `SystemExit` so Ctrl-C still
works) and falls back CLARABEL -> SCS on a fresh solver rather than retrying the
panicked one. All eleven `prob.solve` call sites across `evaluation.py`,
`identification.py`, `control_linear.py`, `control_nonlinear.py`, `sets.py` and
`baselines.py` now route through it; none remain.

**Where the PSD cone came from.** Constraint (19d) is `||K|| <= rho`. Read as the
induced 2-norm, `cp.norm(K, 2)` lowers to a PSD cone -- so Theorem 1's "linear
program" is in fact an SDP, which is both a claim to fix in the manuscript and
the reason this particular failure was reachable. Since `||K||_2 <= ||K||_F`,
bounding the Frobenius norm is a *sufficient* condition for (19d): the
certificate stays valid, the program stays conic-quadratic, and the PSD cone
disappears. `lambda_contractive_lp` and `min_feasible_lambda` take
`norm_type="fro"` by default; pass `norm_type="spectral"` for the tight version.
Frobenius is conservative, so a feasibility rate computed with it is a lower
bound on the spectral one.

## Other

* `exp5_computation.py`: the `slope in N` / `slope in n` columns were silently
  blank whenever a sweep had fewer than 3 points (i.e. always, in `--quick`).
  `loglog_slope` now returns `(slope, n_points)`, accepts 2 points, and the
  table carries `pts (N)` / `pts (n)` so an under-determined fit is visible.
* `results/` cleared. The shipped tables were `--quick` output (2-3 trials,
  truncated sweeps) produced by the pre-fix code, so every number in them is
  superseded. **Re-run at paper settings before quoting anything.**

## Correspondence with the manuscript

    Pbar  <->  S     Lyapunov variable; D = Pbar^{-1}
    Ybar  <->  Y     = Kbar Pbar; Kbar = Ybar Pbar^{-1} recovered after solve
    Theta_*          SUBTRACTED from the (2,2) blocks
    V_1, V_2, V_3    POSITIVE on the diagonal
    kappa            design parameter in the (1,1) block, not read off
