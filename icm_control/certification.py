"""
Finite-sample certification of the true dynamics (Section 3.3).

* `sps_row`          -- Sign-Perturbed Sums confidence region for one regression
                        row (Csaji, Campi and Weyer, IEEE TSP 2015), with the
                        guaranteed ellipsoidal outer approximation of their
                        Section V (S-procedure); exact coverage 1 - q/m.
* `certified_set`    -- row-wise SPS boxes intersected with the prior box.
                        Entries with zero prior radius (known entries,
                        structural zeros) are fixed and subtracted.
* `disturbance_radius` -- tau_hat_N = max_i sup_{theta in Theta_c}
                        || x_i^+ - theta z_i - c0 ||_inf   (fixed center c0).
* `coverage_level`   -- epsilon with (1 - epsilon)^N = beta.

Simultaneous confidence over the n rows follows by the union bound:
P(theta_tr in Theta_c) >= 1 - n q / m (given Assumption 2 for the prior).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def sps_row(Phi, y, m_sps, q, rng):
    """
    Phi : (N, p) regressors, y : (N,) responses.
    Returns theta_hat (LS estimate), R = Phi'Phi/N and radius r such that the
    SPS region is contained in {theta : (theta - theta_hat)' R (theta - theta_hat) <= r}.
    """
    N, p = Phi.shape
    R = Phi.T @ Phi / N
    w_, V_ = np.linalg.eigh(R)
    if w_[0] <= 1e-12:
        raise ValueError("regressor matrix is rank deficient")
    R_mhalf = V_ @ np.diag(w_ ** -0.5) @ V_.T
    theta_hat = np.linalg.solve(Phi.T @ Phi, Phi.T @ y)
    res = y - Phi @ theta_hat
    alphas = rng.choice([-1.0, 1.0], size=(m_sps - 1, N))
    gammas = []
    for a in alphas:
        s = R_mhalf @ (Phi.T @ (a * res)) / N
        Qm = R_mhalf @ ((Phi * a[:, None]).T @ Phi / N) @ R_mhalf
        IQ = np.eye(p) - Qm.T @ Qm
        lmin = np.linalg.eigvalsh(IQ)[0]
        if lmin <= 1e-12:
            gammas.append(np.inf)
            continue
        lam0 = 1.0 / lmin
        Qts = Qm.T @ s

        def g(off):
            lam = lam0 * (1.0 + np.exp(off))
            M = lam * IQ - np.eye(p)
            return lam * s @ s + lam ** 2 * Qts @ np.linalg.solve(M, Qts)

        r_ = minimize_scalar(g, bounds=(-20, 20), method="bounded",
                             options={"xatol": 1e-10})
        gammas.append(float(r_.fun))
    gammas = np.sort(np.array(gammas))[::-1]
    return theta_hat, R, float(gammas[q - 1])


def certified_set(Phi, U, Xp, C_prior, R_prior, m_sps, q, rng, intercept=True):
    """
    Phi : (S, N) regressors (states, or Z(x)), U : (m, N), Xp : (n, N).
    C_prior, R_prior : (n, S + m) prior center and entrywise radius of [A B].
    Returns center, radius of Theta_c (n x (S+m)), or None if empty.
    """
    n = Xp.shape[0]
    N = Xp.shape[1]
    full = np.vstack([Phi, U])
    C, Rd = C_prior.copy(), R_prior.copy()
    for i in range(n):
        free = np.where(R_prior[i] > 1e-12)[0]
        fixed = np.where(R_prior[i] <= 1e-12)[0]
        y = Xp[i] - C_prior[i, fixed] @ full[fixed]
        if free.size == 0:
            continue
        cols = [full[free]] + ([np.ones((1, N))] if intercept else [])
        Ph = np.vstack(cols).T
        th, Rm, r = sps_row(Ph, y, m_sps, q, rng)
        hw = np.sqrt(r * np.diag(np.linalg.inv(Rm)))
        k = free.size
        lo = np.maximum(th[:k] - hw[:k], C_prior[i, free] - R_prior[i, free])
        hi = np.minimum(th[:k] + hw[:k], C_prior[i, free] + R_prior[i, free])
        if np.any(lo > hi):
            return None
        C[i, free] = 0.5 * (lo + hi)
        Rd[i, free] = 0.5 * (hi - lo)
    return C, Rd


def disturbance_radius(Phi, U, Xp, C, Rd, c0=None):
    """tau_hat_N = max_i sup_{|theta - C| <= Rd} || x_i^+ - theta z_i - c0 ||_inf."""
    n = Xp.shape[0]
    c0 = np.zeros(n) if c0 is None else np.asarray(c0, float)
    Z = np.vstack([Phi, U])
    cen = Xp - C @ Z - c0[:, None]
    rad = Rd @ np.abs(Z)
    return float(np.max(np.abs(cen) + rad))


def coverage_level(N, beta):
    """epsilon such that (1 - epsilon)^N = beta."""
    return 1.0 - beta ** (1.0 / N)
