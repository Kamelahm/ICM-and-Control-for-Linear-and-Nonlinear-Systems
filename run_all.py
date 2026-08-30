#!/usr/bin/env python3
"""
Run every experiment and regenerate all figures and tables.

    python run_all.py            # paper settings (~20-40 min)
    python run_all.py --quick    # reduced settings for a smoke test (~3 min)
    python run_all.py --only 1 4 # a subset

Figures land in results/figures (pdf + png), tables in results/tables
(csv + ready-to-\\input tex).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"

JOBS = {
    1: ("exp1_ablation.py",
        "Three-way ablation: DCM / ICM-noSI / ICM  (R4.3, R5.3, R5.7)",
        [], ["--trials", "3", "--alphas", "1.0", "3.0", "--data-lengths", "10"]),
    2: ("exp2_roa.py",
        "Misspecified basis on the pendulum  (R5.6)",
        [], ["--trials", "2"]),
    3: ("exp3_nonlinear.py",
        "Nonlinear validation: coverage (Lemma 9), interval cost (Cor. 2), ROA",
        [], ["--trials", "2", "--n-val", "15", "--n-ic", "6"]),
    4: ("exp4_baselines.py",
        "Stronger baselines: SMI and tube-based ZPC  (R4 minor 2, R5.7)",
        [], ["--trials", "3", "--data-lengths", "30"]),
    5: ("exp5_computation.py",
        "Computation time and complexity vs least squares  (R1.1, R5.7)",
        [], ["--repeats", "2", "--Ns", "10", "50", "--dims", "2", "4"]),
    6: ("exp6_prior.py",
        "Prior mis-specification and the excitation claim of Lemma 4",
        [], ["--trials", "3", "--bias-grid", "0.0", "0.2", "0.4",
             "--N-grid", "3", "6", "12", "--n-val", "15"]),
    7: ("exp7_sensitivity.py",
        "Generator and initial-set sensitivity  (R5.2, R1.7)",
        [], ["--trials", "3", "--eps-grid", "0.02", "0.1", "--gen-counts",
             "1", "3", "--x0-radii", "0.25", "0.5"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="reduced settings, for checking the pipeline runs")
    ap.add_argument("--only", type=int, nargs="+", default=sorted(JOBS))
    args = ap.parse_args()

    total = time.time()
    failures = []
    for k in args.only:
        if k not in JOBS:
            continue
        script, desc, full_args, quick_args = JOBS[k]
        extra = quick_args if args.quick else full_args
        print(f"\n{'=' * 72}\n[{k}] {desc}\n{'=' * 72}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(EXP / script)] + extra)
        dt = time.time() - t0
        if r.returncode != 0:
            failures.append(k)
            print(f"  !! experiment {k} failed (exit {r.returncode})")
        else:
            print(f"  experiment {k} finished in {dt:.1f}s")

    print(f"\n{'=' * 72}")
    print(f"total {time.time() - total:.1f}s")
    print(f"figures -> {ROOT / 'results' / 'figures'}")
    print(f"tables  -> {ROOT / 'results' / 'tables'}")
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
