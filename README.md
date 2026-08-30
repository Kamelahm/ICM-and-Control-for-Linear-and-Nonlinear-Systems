# Simulation package for "Information-Conformant System Modeling and Control"

Every experiment is a standalone script; all figures are IEEE single/double column
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
| Nonlinear coverage (Lemma 9), interval cost (Cor. 2), nonlinear ablation, ROA vs a point model | `exp3_nonlinear.py` | `exp3_nonlinear_coverage`, `exp3_roa_by_arm` | `exp3_nonlinear`, `exp3_interval_cost` |
| Prior mis-specification (Remark 1) and the excitation claim (Lemma 4-2, Remark 2) | `exp6_prior.py` | `exp6_prior_bias`, `exp6_excitation` | `exp6_prior_bias`, `exp6_excitation` |
| Three-way ablation DCM / ICM-noSI / ICM (R4.3, R5.3, R5.7) | `exp1_ablation.py` | `exp1_feasibility_vs_noise`, `exp1_contraction_margin`, `exp1_feasibility_vs_lambda`, `exp1_coverage_vs_extrapolation`, `exp1_tightness_vs_coverage` | `exp1_ablation`, `exp1_error_sources` |
| Misspecified-basis experiment on the pendulum (R5.6) | `exp2_roa.py` | `exp2_misspecified_trajectories`, `exp2_misspecified_phase` | `exp2_roa` |
| Stronger baselines: set-membership [13]–[14], tube-MPC [24] (R4 minor 2, R5.7) | `exp4_baselines.py` | `exp4_baselines_tradeoff`, `exp4_baselines_violations` | `exp4_baselines` |
| Computation time and complexity vs least squares (R1.1, R5.7) | `exp5_computation.py` | `exp5_computation_time`, `exp5_problem_size` | `exp5_complexity`, `exp5_generator_cost` |
| Generator / initial-set sensitivity (R5.2, R1.7) | `exp7_sensitivity.py` | `exp7_template_heatmap`, `exp7_sensitivity_eps`, `exp7_initial_set` | `exp7_template_sensitivity`, `exp7_generator_count`, `exp7_initial_set` |

The three ablation arms come from one solver with two flags, so objective,
disturbance template, operating region, solver, and data are identical across
arms and every difference is attributable to the single feature switched:

```python
solve_conformant_model(..., zonotopic=False, side_info=False)  # DCM,      Eq. (12)
solve_conformant_model(..., zonotopic=True,  side_info=False)  # ICM-noSI, Eq. (10) w/o (10e)-(10f)
solve_conformant_model(..., zonotopic=True,  side_info=True)   # ICM,      Eq. (10)
```

---

## 2. Findings worth putting in the text

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

**Corollary 2 is free in the regime the paper operates in, and that is worth
saying rather than hiding.** With the entrywise template of Section 5.1 every
generator is a single-entry matrix, so the identified matrix zonotope already
*is* an interval matrix and Lemma 1 loses nothing -- the inflation ratio is
1.000 to machine precision. The relaxation only costs something when the
generator directions are not axis-aligned *and* `mu_A > 0`; the rotated-template
row of `exp3_interval_cost` gives 1.34-1.37 and is the only row where the
comparison measures anything.

**At the headline operating point, the zonotopic dynamics are inactive.** The
`[diagnostic]` line of `exp1` reports `1'mu_A`. At the generous template
(`gamma = 8`, the upper block of Table 1) it is `~1e-8`: the minimum-volume
objective explains everything additively and collapses `Ahat`, `Bhat` to
singletons, so the identified model is a point model plus an additive set --
structurally a DCM with a better centre. At `gamma = 1` it is `~0.5` and the
sets are genuinely active. This is consistent with the paper's own sentence
that the gain comes from shifting the centres, but it means the enclosure
growth of Lemma 5 is only observable at the tight template: the growth ratio
`||E||(3x)/||E||(1x)` is 2.0-2.4 for ICM and exactly 1.00 for DCM at
`gamma = 1`, and 1.00 for everything at `gamma = 8`. Draw the Lemma 5 figure at
`gamma = 1` or it shows nothing.

**Infeasibility really is the diagnostic Remark 1 claims.** Sweeping the prior
centre bias at fixed half-width 0.15: identification stays 100% feasible while
the prior contains the truth, degrades in coverage (100% -> 49% -> 27%) as the
bias grows past the half-width, and hits 0% feasibility at bias 0.4. The
prior-free arm is flat across the whole sweep, which is the control that makes
the attribution clean.

**The identified sets never contain the true matrices.** `contains_true` is 0%
across every configuration, including bias 0. This is the empirical form of the
caveat the paper already states after the motivating example, and it is the
reason coverage rather than containment has to be the headline metric. State it
rather than leaving a reader to discover it.

**The zonotopic slack does not reduce the data requirement; the prior does.**
Below the excitation threshold `n + m = 6`, with an identical zonotopic
template in all three arms, the prior-free arm has centre error 0.71 and 0%
coverage at `N = 3`, while the dense and structure-aware priors sit at 0.002 to
0.005 and 100%. This is the concrete evidence for Lemma 4-2 and Remark 2, which
the paper currently argues only analytically.

**Sensitivity is to the generator magnitude, not the directions.** At fixed
`ε`, the choice among per-entry / per-row / random / residual-PCA templates
changes coverage by a few percent; changing `ε` over a decade moves everything.
`ε` is effectively the prior width and should be presented as such.

---

## 3. Package layout

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
experiments/             exp1 (linear ablation), exp2 (ROA figure),
                         exp3 (nonlinear validation), exp4 (baselines),
                         exp5 (computation), exp6 (prior + excitation),
                         exp7 (generator sensitivity)
results/figures, results/tables
```

Reproducibility: every script takes an explicit seed sequence, all randomness
goes through `numpy.random.default_rng`, and raw per-trial records are written
to `results/tables/exp*_raw*.csv` so the aggregates in the paper can be
re-derived without re-running the solvers.
