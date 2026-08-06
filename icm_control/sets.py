"""
Set representations used throughout the ICM framework.

Conventions
-----------
Zonotope        Z = <c, G>_Z,  c in R^v,  G in R^{v x s}   (generators are COLUMNS)
MatrixZonotope  M = <C, [G_1 ... G_s]>_M, C, G_i in R^{v x p}
                (generators are stored as a list of v-by-p BLOCKS, matching Def. 3)

The identified sets produced by the convex program (Lemma 2) are always of the
"envelope-scaled" form  <C, {mu_i * G_i}>  where the directions G_i are fixed a
priori and only the non-negative scalars mu_i are optimized.
"""

from __future__ import annotations

import itertools

import numpy as np

from .solver_utils import safe_solve, solved_ok


# --------------------------------------------------------------------------- #
#  Zonotope
# --------------------------------------------------------------------------- #
class Zonotope:
    """Zonotope <c, G>_Z = {c + G lam : |lam|_inf <= 1}."""

    def __init__(self, c, G):
        self.c = np.atleast_1d(np.asarray(c, dtype=float)).reshape(-1)
        self.G = np.atleast_2d(np.asarray(G, dtype=float))
        if self.G.shape[0] != self.c.shape[0]:
            raise ValueError(f"center dim {self.c.shape} vs generator dim {self.G.shape}")

    # -- basic properties ---------------------------------------------------
    @property
    def dim(self):
        return self.c.shape[0]

    @property
    def n_gen(self):
        return self.G.shape[1]

    def interval(self):
        """Axis-aligned interval hull [lo, hi] (Lemma 1)."""
        r = np.sum(np.abs(self.G), axis=1)
        return self.c - r, self.c + r

    def support(self, direction):
        """max_{x in Z} <direction, x>."""
        d = np.asarray(direction, dtype=float).reshape(-1)
        return float(d @ self.c + np.sum(np.abs(d @ self.G)))

    def contains(self, x, tol=1e-8):
        """
        Exact membership test  x in <c,G>  <=>  exists |lam|<=1 with G lam = x-c.
        Solved with HiGHS (a feasibility LP); fast enough for Monte-Carlo loops.
        """
        from scipy.optimize import linprog

        b = np.asarray(x, float).reshape(-1) - self.c
        lo, hi = self.interval()
        if np.any(np.asarray(x, float).reshape(-1) < lo - 1e-9) or \
           np.any(np.asarray(x, float).reshape(-1) > hi + 1e-9):
            return False                      # cheap necessary condition
        res = linprog(c=np.zeros(self.n_gen), A_eq=self.G, b_eq=b,
                      bounds=[(-1 - tol, 1 + tol)] * self.n_gen, method="highs")
        return bool(res.success)

    def volume_proxy(self):
        """Cheap size proxy: volume of the interval hull (log-scale friendly)."""
        lo, hi = self.interval()
        return float(np.prod(np.maximum(hi - lo, 1e-16)))

    def vertices(self):
        """Vertices (2^s enumeration, then convex hull). Use for plots only."""
        s = self.n_gen
        if s > 14:
            raise ValueError("too many generators for exhaustive vertex enumeration")
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=s)))
        pts = (self.c[None, :] + signs @ self.G.T)
        if self.dim == 1:
            return np.array([[pts.min()], [pts.max()]])
        from scipy.spatial import ConvexHull

        hull = ConvexHull(pts)
        return pts[hull.vertices]

    # -- algebra ------------------------------------------------------------
    def __add__(self, other):  # Minkowski sum
        if isinstance(other, Zonotope):
            return Zonotope(self.c + other.c, np.hstack([self.G, other.G]))
        return Zonotope(self.c + np.asarray(other, float).reshape(-1), self.G)

    def __rmul__(self, M):  # M @ Z
        M = np.atleast_2d(M)
        return Zonotope(M @ self.c, M @ self.G)

    def scale(self, a):
        return Zonotope(a * self.c, a * self.G)

    def __repr__(self):
        return f"Zonotope(dim={self.dim}, n_gen={self.n_gen})"


# --------------------------------------------------------------------------- #
#  Matrix zonotope
# --------------------------------------------------------------------------- #
class MatrixZonotope:
    """Matrix zonotope <C, {G_i}>_M = {C + sum_i lam_i G_i : |lam|_inf <= 1}."""

    def __init__(self, C, blocks):
        self.C = np.atleast_2d(np.asarray(C, dtype=float))
        self.blocks = [np.atleast_2d(np.asarray(B, dtype=float)) for B in blocks]
        for B in self.blocks:
            if B.shape != self.C.shape:
                raise ValueError(f"block shape {B.shape} != center shape {self.C.shape}")

    @property
    def shape(self):
        return self.C.shape

    @property
    def n_gen(self):
        return len(self.blocks)

    def interval(self):
        """Interval over-approximation I = [C - dG, C + dG] (Lemma 1)."""
        dG = sum((np.abs(B) for B in self.blocks), np.zeros_like(self.C))
        return self.C - dG, self.C + dG

    def radius(self):
        """Elementwise generator radius dG (used for Delta A, Delta B)."""
        return sum((np.abs(B) for B in self.blocks), np.zeros_like(self.C))

    def vec_generators(self):
        """Column-stacked vectorization -> zonotope in R^{v*p} (for containment)."""
        return np.column_stack([B.reshape(-1, order="F") for B in self.blocks])

    def vec_center(self):
        return self.C.reshape(-1, order="F")

    def sample(self, rng, n_samples=1):
        out = []
        for _ in range(n_samples):
            lam = rng.uniform(-1, 1, size=self.n_gen)
            out.append(self.C + sum(l * B for l, B in zip(lam, self.blocks)))
        return out if n_samples > 1 else out[0]

    def contains_matrix(self, M, tol=1e-7):
        """Exact test M in <C, {G_i}> via LP."""
        import cvxpy as cp

        lam = cp.Variable(self.n_gen)
        expr = self.C + sum(lam[i] * self.blocks[i] for i in range(self.n_gen))
        prob = cp.Problem(cp.Minimize(0),
                          [expr == np.asarray(M, float),
                           cp.norm(lam, "inf") <= 1 + tol])
        status, _ = safe_solve(prob)
        return solved_ok(status)

    def scaled(self, mu):
        """Envelope-scaled copy <C, {mu_i G_i}>."""
        mu = np.asarray(mu, float).reshape(-1)
        return MatrixZonotope(self.C, [m * B for m, B in zip(mu, self.blocks)])

    def __repr__(self):
        return f"MatrixZonotope(shape={self.shape}, n_gen={self.n_gen})"


# --------------------------------------------------------------------------- #
#  Polytope
# --------------------------------------------------------------------------- #
class Polytope:
    """P = <H, h>_P = {x : H x <= h}."""

    def __init__(self, H, h):
        self.H = np.atleast_2d(np.asarray(H, float))
        self.h = np.asarray(h, float).reshape(-1)

    @property
    def dim(self):
        return self.H.shape[1]

    def contains(self, x, tol=1e-9):
        return bool(np.all(self.H @ np.asarray(x, float).reshape(-1) <= self.h + tol))

    def contains_zonotope(self, Z, tol=1e-9):
        """Eq. (2): H c + |H G| 1 <= h."""
        return bool(np.all(self.H @ Z.c + np.sum(np.abs(self.H @ Z.G), axis=1)
                           <= self.h + tol))

    def max_norm_bound(self, p=np.inf):
            """
            p = inf  pairs with the induced infinity norm on K and the ell_1 row
                     norms in (19b); this is the linear-program form and the default.
            p = 2    pairs with the Frobenius or induced 2-norm on K and ell_2 row
                     norms (the earlier SOCP/SDP forms).
            """
            import cvxpy as cp

            x = cp.Variable(self.dim)
            box = np.zeros(self.dim)
            for i in range(self.dim):
                vals = []
                for sgn in (+1.0, -1.0):
                    prob = cp.Problem(cp.Maximize(sgn * x[i]), [self.H @ x <= self.h])
                    safe_solve(prob)
                    vals.append(abs(prob.value) if prob.value is not None else np.inf)
                box[i] = max(vals)
            return float(np.linalg.norm(box, ord=p))


# --------------------------------------------------------------------------- #
#  Generator template factories
# --------------------------------------------------------------------------- #
def identity_blocks(n, p, eps):
    """One generator per matrix entry: G_(i,j) = eps * e_i e_j^T  (n*p blocks)."""
    blocks = []
    for i in range(n):
        for j in range(p):
            B = np.zeros((n, p))
            B[i, j] = eps
            blocks.append(B)
    return blocks


def entrywise_blocks(R, tol=1e-12):
    """
    One generator per *uncertain* entry of the radius matrix R.
    Entries with R[i,j] == 0 are structural zeros / exactly known values and get
    no generator, which is how sparsity and known entries enter as prior
    information (the r independent affine relations of Lemma 4-2).
    """
    R = np.atleast_2d(np.asarray(R, float))
    blocks = []
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if abs(R[i, j]) > tol:
                B = np.zeros_like(R)
                B[i, j] = abs(R[i, j])
                blocks.append(B)
    return blocks


def structured_prior(C, R):
    """Matrix zonotope <C, entrywise_blocks(R)> plus its generator template."""
    blocks = entrywise_blocks(R)
    return MatrixZonotope(C, blocks), blocks


def row_blocks(n, p, eps):
    """One generator per row (n blocks) - coarse but cheap template."""
    blocks = []
    for i in range(n):
        B = np.zeros((n, p))
        B[i, :] = eps
        blocks.append(B)
    return blocks


def random_blocks(n, p, eps, n_blocks, rng):
    """Random unit-Frobenius directions scaled by eps."""
    blocks = []
    for _ in range(n_blocks):
        B = rng.normal(size=(n, p))
        blocks.append(eps * B / np.linalg.norm(B, "fro"))
    return blocks


def residual_pca_blocks(Phi, U, Xp, eps, n_blocks):
    """
    Data-driven directions (Remark 1): least-squares residuals are regressed onto
    the dominant right-singular directions of the regressor, giving generator
    blocks aligned with the directions the data cannot resolve.
    """
    Theta = np.vstack([Phi, U])
    Theta_pinv = np.linalg.pinv(Theta)
    M_ls = Xp @ Theta_pinv                     # [A_ls  B_ls]
    R = Xp - M_ls @ Theta                      # residuals, n x N
    _, _, Vt = np.linalg.svd(Theta, full_matrices=False)
    n, p = Xp.shape[0], Phi.shape[0]
    blocks = []
    for k in range(min(n_blocks, Vt.shape[0])):
        w = Vt[k]                              # direction in sample space
        B = (R @ w).reshape(-1, 1) @ np.ones((1, p))
        nrm = np.linalg.norm(B, "fro")
        blocks.append(eps * B / nrm if nrm > 1e-12 else eps * np.ones((n, p)) / np.sqrt(n * p))
    return blocks