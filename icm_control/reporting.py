"""IEEE-column plotting defaults and results IO (CSV + LaTeX tables)."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# Single-column IEEE figure: 3.4in wide.
COL_W = 3.4
COL_H = 2.3

METHOD_STYLE = {
    "DCM":        dict(color="#1f77b4", marker="o", ls="-"),
    "ICM-noSI":   dict(color="#2ca02c", marker="^", ls="--"),
    "ICM":        dict(color="#d62728", marker="s", ls="-"),
    "SMI":        dict(color="#9467bd", marker="v", ls=":"),
    "Tube-MPC":   dict(color="#8c564b", marker="D", ls="-."),
    "LS":         dict(color="#7f7f7f", marker="x", ls=":"),
    "ICM-rot":    dict(color="#e377c2", marker="*", ls="--"),
}

LABELS = {
    "DCM": "DCM [22]",
    "ICM-noSI": "ICM w/o side info.",
    "ICM": "ICM (proposed)",
    "SMI": "Set-membership [13], [14]",
    "Tube-MPC": "Tube-based ZPC [24]",
    "LS": "Least squares",
    "ICM-rot": "ICM, rotated template",
}


def use_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "legend.framealpha": 0.9,
        "legend.handlelength": 1.8,
    })


def style(method):
    return METHOD_STYLE.get(method, dict(marker="o", ls="-"))


def label(method):
    return LABELS.get(method, method)


def save_fig(fig, name, exts=("pdf", "png")):
    paths = []
    for e in exts:
        p = FIG_DIR / f"{name}.{e}"
        fig.savefig(p)
        paths.append(str(p))
    plt.close(fig)
    print(f"  [fig] {name}: " + ", ".join(os.path.basename(p) for p in paths))
    return paths


def save_table(df: pd.DataFrame, name, caption="", label_tex=None,
               float_fmt="%.3f", index=False):
    csv = TAB_DIR / f"{name}.csv"
    df.to_csv(csv, index=index)
    tex = TAB_DIR / f"{name}.tex"
    body = df.to_latex(index=index, float_format=lambda v: float_fmt % v,
                       escape=False, column_format="l" + "c" * (df.shape[1] - 1))
    label_tex = label_tex or f"tab:{name}"
    with open(tex, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write(f"\\caption{{{caption}}}\n\\label{{{label_tex}}}\n")
        f.write("\\resizebox{\\columnwidth}{!}{%\n")
        f.write(body)
        f.write("}\n\\end{table}\n")
    print(f"  [tab] {name}.csv / {name}.tex")
    return str(csv), str(tex)


def summarize(df: pd.DataFrame, by, cols, aggs=("mean", "std")):
    return df.groupby(by)[list(cols)].agg(list(aggs)).reset_index()


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a success rate (error bars on feasibility)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
