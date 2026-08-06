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

## Package layout

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
