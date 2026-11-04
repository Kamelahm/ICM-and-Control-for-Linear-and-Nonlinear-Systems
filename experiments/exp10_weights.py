"""
Experiment 10 -- sensitivity of the ICM split to the objective weights
(Section 5.7, review Comment 4).

nu_A = nu_B = 0.1, nu_s = 1 fixed; nu_w = 0.1 * r for r in {0.1, 1, 10}.
Third-order plant, alpha_1 = 1, gamma = 1 (matrix part active), independent
one-step data, regularized ICM, N in {10, 30}, 20 trials each.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import warnings
warnings.filterwarnings("ignore")

from icm_control.control_linear import min_feasible_lambda, to_uncertain_model
from icm_control.evaluation import coverage, enclosure_size, sample_validation
from icm_control.identification import solve_icm
from icm_control.protocol import REG, U_MAX_LIN, X_RADIUS_LIN
from icm_control.reporting import save_table
from icm_control.systems import (box_region, collect_independent,
                                 linear_side_info, safe_set_box,
                                 third_order_system)

SIGMA = 0.01


def main(args):
    plant = third_order_system(); n = plant.n
    safe = safe_set_box(n, 1.0); Mx = safe.max_norm_bound(p=np.inf)
    A_side, GA, B_side, GB = linear_side_info(plant.A, plant.B, rel_err=0.15)
    Gw_true = args.alpha * SIGMA * np.eye(n)
    rows = []
    for N in args.Ns:
        for t in range(args.trials):
            seed = 15000 + 29 * t + N
            rng = np.random.default_rng(seed)
            Phi, U, Xp, Xs = collect_independent(plant, N, rng, x_radius=X_RADIUS_LIN,
                                                 u_range=0.5, w_gen=Gw_true)
            r_data = float(np.max(np.abs(Xs)))
            val = {f: sample_validation(plant, args.n_val,
                                        np.random.default_rng(seed + 10_000),
                                        radius=f * r_data, w_gen=Gw_true)
                   for f in (1.0, 3.0)}
            for ratio in args.ratios:
                w = (0.1, 0.1, 0.1 * ratio, 1.0)
                res = solve_icm(Phi=Phi, U=U, Xp=Xp, Gw=args.gamma * SIGMA * np.eye(n),
                                X_region=box_region(n, X_RADIUS_LIN), GA_blocks=GA,
                                GB_blocks=GB, A_side=A_side, B_side=B_side,
                                mu_floor=args.mu_floor, reg=REG, weights=w)
                row = {"N": N, "seed": seed, "ratio": ratio, "id_feasible": res.feasible}
                if res.feasible:
                    row.update({
                        "muA": float(np.sum(res.muA)), "muB": float(np.sum(res.muB)),
                        "muw": float(np.sum(res.muw)),
                        "cov1": 100 * coverage(res, *val[1.0][:3]),
                        "cov3": 100 * coverage(res, *val[3.0][:3]),
                        "encl1": enclosure_size(res, *val[1.0][:2]),
                        "encl3": enclosure_size(res, *val[3.0][:2]),
                        "lambda_min": min_feasible_lambda(to_uncertain_model(res), safe,
                                                          Mx, u_max=U_MAX_LIN)})
                rows.append(row)
            print(f"  N={N} trial {t} done", flush=True)
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "results" / "tables"
    df.to_csv(out / "exp10_raw.csv", index=False)
    tab = (df[df.id_feasible].groupby("ratio")
           [["muA", "muB", "muw", "cov1", "cov3", "encl1", "encl3", "lambda_min"]]
           .mean().reset_index())
    tab.insert(1, "ID feas.", 100 * df.groupby("ratio").id_feasible.mean().values)
    save_table(tab, "exp10_weights")
    print(tab.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+", default=[10, 30])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--mu-floor", type=float, default=0.20)
    ap.add_argument("--n-val", type=int, default=40)
    main(ap.parse_args())
