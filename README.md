# Simulation package for "Information-Conformant System Modeling and Control"

Replacement simulation section addressing the reviewer requests. Every
experiment is a standalone script; all figures are IEEE single/double column
PDFs and all tables are emitted both as CSV and as ready-to-`\input` LaTeX.

```bash
pip install -r requirements.txt
python run_all.py --quick     # ~3 min, checks the pipeline
python run_all.py             # paper settings
python experiments/exp1_ablation.py --help   # per-experiment options
```

Outputs: `results/figures/*.pdf`, `results/tables/*.tex` and `*.csv`.

---

## 1. Reviewer comment → deliverable

| Reviewer request | Script | Figures | Tables |
|---|---|---|---|
| Three-way ablation DCM / ICM-noSI / ICM (R4.3, R5.3, R5.7) | `exp1_ablation.py` | `exp1_feasibility_vs_noise`, `exp1_contraction_margin`, `exp1_feasibility_vs_lambda`, `exp1_coverage_vs_extrapolation`, `exp1_tightness_vs_coverage` | `exp1_ablation`, `exp1_error_sources` |
| Misspecified-basis experiment on the pendulum (R5.6) | `exp2_roa.py` | `exp2_misspecified_trajectories`, `exp2_misspecified_phase` | `exp2_roa` |
| Stronger baselines: set-membership [13]–[14], tube-MPC [24] (R4 minor 2, R5.7) | `exp4_baselines.py` | `exp4_baselines_tradeoff`, `exp4_baselines_violations` | `exp4_baselines` |
| Computation time and complexity vs least squares (R1.1, R5.7) | `exp5_computation.py` | `exp5_computation_time`, `exp5_problem_size` | `exp5_complexity`, `exp5_generator_cost` |
| Generator / initial-set sensitivity (R5.2, R1.7) | `exp7_sensitivity.py` | `exp7_template_heatmap`, `exp7_sensitivity_eps`, `exp7_initial_set` | `exp7_template_sensitivity`, `exp7_generator_count`, `exp7_initial_set` |

The three ablation arms come from one solver with two flags, so objective,
disturbance template, operating region, solver and data are identical across
arms and every difference is attributable to the single feature switched:

```python
solve_conformant_model(..., zonotopic=False, side_info=False)  # DCM,      Eq. (12)
solve_conformant_model(..., zonotopic=True,  side_info=False)  # ICM-noSI, Eq. (10) w/o (10e)-(10f)
solve_conformant_model(..., zonotopic=True,  side_info=True)   # ICM,      Eq. (10)
```

---

## 2. Things to fix in the theory before resubmitting

These came out of implementing the results, and a careful reviewer will find
them too.

**(a) Eq. (32b)–(32c) is not satisfiable as printed.** The quadratic bounding
step `f(uvᵀ + vuᵀ) ⪰ −(τ|Δ| vvᵀ + τ⁻¹ uuᵀ)` *subtracts* from the diagonal, so
the robust LMI must carry `P − Θ_A − Θ_B̃` in the (2,2) block and `+V₃`, `+V₂`
in the Schur blocks. As printed, `−V₃` and `−V₂` sit on the diagonal, and no
positive-definite matrix has negative-definite diagonal blocks.

**(b) Eq. (32d) points the wrong way.** `X − P⁻¹ ⪰ 0` makes `X` an *upper*
bound on `P⁻¹`, but `X` occupies the (1,1) block where enlarging it *relaxes*
the LMI, so a feasible point does not imply the Lyapunov decrease. The code
uses the congruence `S = P⁻¹`, `Y = KS`, recovering `K = YS⁻¹` and `D = S⁻¹`;
this removes the inverse entirely and is an exact LMI in `(S, Y, λ, ε)`. The
LMI is homogeneous of degree one in `(S, Y, λ, ε)` and `K` is invariant under
that scaling, so `S ⪰ I` is imposed without loss of generality — this also
fixes the conditioning, which the certificates of Theorem 3 are sensitive to.

**(c) The contraction rate must be designed, not read off.** With the (1,1)
block equal to `S`, the LMI only certifies non-strict decrease, so the
recovered `κ` sits at `≈0.999` and `δ*`, `r*`, `c*` collapse to nothing.
Putting `κ S` in the (1,1) block with `κ` a design parameter and sweeping it
(`sweep_contraction_targets`) is what makes Theorem 3 produce non-trivial
numbers.

**(d) Assumption 2 is stronger than what the identification delivers.** It
demands a *single* pair `(A¹, B¹)` absorbing the mismatch for all admissible
`(x,u)`, whereas Definition 4 explicitly lets the realization vary with `k`.
Worse, the minimum-volume objective drives individual `µ*_{w,i}` to zero, so
`Ŵ` becomes lower dimensional and the representability condition fails at
*any* scaling. `scripts/calibrate_assumptions.py` quantifies this: with no floor the required
inflation `t*` is unbounded; a floor `µ_w ≥ 0.05` makes it finite at
negligible cost in `s_X` and coverage. Two options: state Assumption 2
pointwise in `k` to match Definition 4, or add the floor to (10) and keep the
uniform version. The solver supports `mu_floor=` for the latter.

**(e) The Fig. 2 argument is hard to defend.** Presenting a *larger* `s_X` as
"necessary conservatism" invites the reply that any method can inflate its
sets. The replacement metric throughout this package is held-out **coverage**:
the fraction of true transitions the identified model actually explains,
measured inside the identification region and at 2× and 3× extrapolation.
That is what a safety guarantee rests on, and it separates the arms cleanly.

---

## 3. Findings worth putting in the text

**The α₁ sweep only reproduces the reported behaviour when the disturbance
template `G^w` is held fixed while the true noise scales.** That is the regime
Lemma 3 is actually about: the cap `µ_w ≤ 1` in (10d) limits how much a purely
additive explanation can absorb, so DCM becomes infeasible while ICM keeps
explaining the data through the zonotopic dynamics. This gives the feasibility
collapse a mechanism instead of leaving it unexplained. Representative numbers
(third-order plant, template `0.01 I`):

| α₁ | DCM | ICM-noSI | ICM |
|---|---|---|---|
| 1.0 | 100% | 100% | 100% |
| 2.0 | 33% | 100% | 100% |
| 3.0 | 0% | 100% | 100% |
| 5.0 | 0% | 100% | 66% (N=30) |

The ICM drop at α₁ = 5 is the "infeasibility as diagnostic signal" of Remark 2
showing up empirically, and the non-monotonicity in `N` reproduces the
observation already in the paper.

**Biased priors produce infeasibility, not confident errors.** In
With prior half-width 0.15, identification is 100% feasible while the
prior contains the truth and drops to 0% once the bias exceeds the half-width.
ICM does not silently return a wrong model — this is the concrete evidence for
Remark 2 that the paper currently asserts without support.

**The honest baseline comparison.** Set-membership with an *oracle* disturbance
bound attains 100% coverage but with ~1.8× the enclosure size of ICM, and it
becomes infeasible as soon as the assumed bound is optimistic (β = 0.5). Tube
ZPC covers the training data but degrades to ~60% out of sample. Stating this
plainly is stronger than claiming dominance: ICM's case is that it needs no
disturbance bound, stays feasible where DCM does not, and is tighter than SMI.

**Lemma 5 is visible in the data.** ICM's coverage at 2× extrapolation is
*higher* than at 1× in several configurations, because the state-multiplicative
part of the enclosure (18) widens with `‖x‖` exactly where extrapolation makes
prediction less reliable. Worth pointing at explicitly.

**The Theorem 3/4 constants are conservative.** Certified `w̄_max` lands around
0.02–0.04 on the pendulum while simulation stays bounded at 10× that, and the
predicted ultimate bound exceeds the observed one by one to two orders of
magnitude. The binding term is `a = sup‖D A_cl‖`; the code tightens it using
elementwise monotonicity of the spectral norm (`‖DΔA‖₂ ≤ ‖|D| ΔA‖₂`), which
helps substantially but does not close the gap. Report it as
sufficient-but-conservative with the observed values alongside — silence here
is what draws fire.

**Sensitivity is to the generator magnitude, not the directions.** At fixed
`ε`, the choice among per-entry / per-row / random / residual-PCA templates
changes coverage by a few percent; changing `ε` over a decade moves everything.
`ε` is effectively the prior width and should be presented as such.

---

## 4. Package layout

```
icm_control/
  sets.py                zonotopes, matrix zonotopes, polytopes, generator templates
  systems.py             plants, data collection, physically motivated priors
  identification.py      Eq. (10) / Lemma 2 / Lemma 6 and Eq. (12), one solver
  baselines.py           least squares, set-membership, tube-based ZPC
  control_linear.py      Theorem 1 LP, contraction margin, closed-loop checks
  control_nonlinear.py   Theorem 2 SDP, Theorem 3 ROA, Theorem 4 ISS, L_Z / ā
  evaluation.py          coverage, enclosure size, Assumption 2 test
  reporting.py           IEEE plot style, CSV + LaTeX table export
experiments/             exp1 ... exp7
results/figures, results/tables
```

Reproducibility: every script takes an explicit seed sequence, all randomness
goes through `numpy.random.default_rng`, and raw per-trial records are written
to `results/tables/exp*_raw*.csv` so the aggregates in the paper can be
re-derived without re-running the solvers.
